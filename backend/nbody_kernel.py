"""CUDA-N-Body-Kernel (f64) fuer den Sonnensystem-Simulator.

Spiegelt exakt die Physik des JS-Workers in index.html:
- Yoshida-Integrator 4. Ordnung (3 gewichtete Velocity-Verlets pro Schritt)
- adaptiver Substep in Hybrid-Worker-Semantik (ASTAD): massive Paare UND
  Asteroid-x-massiv druecken dt (tEnc/20, Floor MAX_SUB_DT/1000)
- Kraefte: massive×massive + massive×Asteroid (symmetrisch, Softening 1e-6),
  Asteroid×Asteroid vernachlaessigt
- ausgeschaltete Koerper (visible=0) sind eingefroren

MULTI-GPU (hardwareagnostisch): Die Asteroiden werden gewichtet nach
f64-Score auf beliebig viele Karten geshardet; die massiven Koerper sind
auf jeder Karte repliziert und werden von JEDER Karte deterministisch
identisch integriert (die Gegenkraft-Partialsummen aller Shards werden in
fester Reihenfolge summiert — bit-identische Bahnen, exakte Physik, kein
Naeherungs-Fallback). Die Substep-Synchronisation laeuft ueber eine
System-Scope-Spin-Barrier auf gemapptem Host-Speicher (PCIe reicht, kein
NVLink noetig). Mit EINER Karte degeneriert dieselbe Codebahn zu reinen
grid.sync()s — null Zusatz-Overhead.

BATCH-SAMPLING: Ein Launch rechnet K Raster-Samples am Stueck und schreibt
pro Raster einen kompakten f32-Snapshot (Launch-/Sync-Overhead nur noch
1x pro K Samples).

Der komplette Batch laeuft in EINEM cooperative Launch je Karte
(Grid-Sync); Asteroiden liegen in globalen Arrays (Grid-Stride),
massive Koerper verwaltet Thread 0, Gegenkraefte werden blockweise in
Shared Memory reduziert.
"""
from __future__ import annotations

# Alle CUDA-Abhaengigkeiten (Header, NVRTC, libcudadevrt) kommen aus den
# nvidia-*-cu12-Wheels des venv (cupy-cuda12x[ctk]) — kein System-Toolkit
# noetig. CuPys Pfadsuche kennt beim gesplitteten cu12-Wheel-Layout aber
# keinen CUDA-Root und findet libcudadevrt.a (Cooperative-Groups-Linking)
# nicht — der memoisierte Pfad wird daher direkt gesetzt. (Interne API,
# gilt fuer cupy 14.x; bei einem CuPy-Upgrade pruefen.)
import ctypes
import os

import cupy as cp
import cupy.cuda.compiler as _cupy_compiler
import numpy as np
import nvidia.cuda_runtime

_cupy_compiler._cudadevrt = os.path.join(
    nvidia.cuda_runtime.__path__[0], "lib", "libcudadevrt.a")

# CuPy klemmt das Grid bei Cooperative Launches selbst auf das Residenz-
# Limit (der Grid-Stride-Loop im Kernel deckt den Rest) — die zugehoerige
# Warnung ist Absicht und wuerde nur das Server-Log fluten.
import warnings  # noqa: E402

warnings.filterwarnings("ignore", message="The grid size will be reduced")

M_MAX = 64
G_MAX = 8                      # max. Physik-GPUs (GSync-Layout)
K_MAX = 16                     # max. Samples pro Batch-Launch
G_AU = 4 * np.pi * np.pi
SOFTENING = 1e-6
MAX_SUB_DT_YEARS = 0.5 / 365.25
MAX_SUB_STEPS_PER_FRAME = 100_000
YOSHIDA_W1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
YOSHIDA_W0 = -(2.0 ** (1.0 / 3.0)) / (2.0 - 2.0 ** (1.0 / 3.0))

_SRC = r"""
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define M_MAX 64
#define G_MAX 8

// Geteilter Sync-Bereich aller Physik-GPUs. Bei nGpus > 1 liegt er in
// GEMAPPTEM Host-Speicher (jede Karte sieht dieselben Bytes ueber PCIe),
// bei nGpus == 1 in normalem Device-Speicher (gleiche Codebahn, volle
// Geschwindigkeit).
// ATOMICS-FREI: System-Atomics auf gemapptem Host-Speicher sind ueber
// PCIe nicht unterstuetzt (nur mit hostNativeAtomicSupported, d. h.
// NVLink-Host-Kopplung) — der erste Wurf hing deshalb in der Barrier.
// Stattdessen schreibt jede Karte AUSSCHLIESSLICH in ihre eigenen Slots
// (Rundenzaehler, Min-Wert, Partialsummen) und liest die der anderen:
// reine Loads/Stores, die jede PCIe-Plattform beherrscht.
#define PAD 16   // eigene Cacheline pro Karte gegen False Sharing
struct GSync {
    unsigned int round_[G_MAX * PAD];       // Barrier-Rundenzaehler je GPU
    unsigned long long minEnc[G_MAX * PAD]; // lokales tEnc/20-Min je GPU
    double backX[G_MAX * M_MAX];            // Gegenkraft-Partialsummen
    double backY[G_MAX * M_MAX];
};

struct Ctrl {            // GPU-lokaler, replizierter Loop-Zustand
    double remaining;    // verbleibende Raster-Zeit (identisch auf allen)
    double subDt;
    int done;
    int guard;
    unsigned long long minEncDev;   // device-lokale Min-Reduktion
};

// System-weite Barrier zwischen allen Physik-GPUs (Sense ueber
// monoton wachsende Rundenzaehler). Ein Thread pro Karte macht den
// PCIe-Handshake, der Rest haengt im grid.sync. Bei nGpus == 1
// degeneriert sie zum reinen grid.sync.
__device__ void sys_barrier(GSync* gs, const int gpuId, const int nGpus,
                            unsigned int* barRound,
                            cg::grid_group& grid, const int tid)
{
    grid.sync();
    if (nGpus > 1 && tid == 0) {
        // Alle vorherigen Writes dieser Karte (Partialsummen, minEnc)
        // muessen VOR dem Rundenzaehler systemweit sichtbar sein.
        __threadfence_system();
        const unsigned int r = ++(*barRound);
        ((volatile unsigned int*)gs->round_)[gpuId * PAD] = r;
        for (int g = 0; g < nGpus; g++) {
            volatile unsigned int* p =
                &((volatile unsigned int*)gs->round_)[g * PAD];
            while (*p < r) { __nanosleep(256); }
        }
    }
    grid.sync();
}

extern "C" __global__ void frame_kernel(
    // Asteroiden-SHARD dieser Karte [nAst]
    double* __restrict__ ax, double* __restrict__ ay,
    double* __restrict__ avx, double* __restrict__ avy,
    double* __restrict__ aAccX, double* __restrict__ aAccY,
    const double* __restrict__ am,
    const unsigned char* __restrict__ aVis,
    // massive Koerper (REPLIK, wird von jeder Karte identisch integriert)
    double* mx, double* my, double* mvx, double* mvy,
    const double* __restrict__ mm,
    const unsigned char* __restrict__ mVis,
    double* mAccX, double* mAccY,
    double* backX, double* backY,        // device-lokale Gegenkraft-Akkus
    unsigned char* __restrict__ hot,     // Klassifikation je Shard-Asti
    double* mx0, double* my0,            // Massiv-Pos am Segment-Anfang
    GSync* gs, Ctrl* ctrl,
    // Batch-Snapshots: kompakt [K][4][nAst] f32; GPU 0 schreibt die
    // Massiven zusaetzlich nach snapM [K][4][M].
    float* __restrict__ snap, float* __restrict__ snapM,
    const int nAst, const int M,
    const int gpuId, const int nGpus,
    const double G, const double soft,
    const double dtRaster, const int K,
    const double maxSubDt, const int maxSubSteps,
    const double w1, const double w0)
{
    cg::grid_group grid = cg::this_grid();
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    // Rundenzaehler dieser Karte — ueber Launches hinweg fortlaufend
    // (gs->round_ behaelt den Stand, alle Karten starten konsistent).
    unsigned int barRound = (nGpus > 1)
        ? ((volatile unsigned int*)gs->round_)[gpuId * PAD] : 0u;

    __shared__ double s_mx[M_MAX], s_my[M_MAX], s_mm[M_MAX];
    __shared__ unsigned char s_mv[M_MAX];
    __shared__ double s_fx[M_MAX], s_fy[M_MAX];
    __shared__ unsigned long long s_minEnc;

    // ---- Initiale Beschleunigungen (wie computeAccel() im JS-Worker) ----
    if (tid == 0) {
        ctrl->guard = 0;
        for (int j = 0; j < M; j++) { backX[j] = 0.0; backY[j] = 0.0; }
    }
    grid.sync();
    if (threadIdx.x < M) {
        s_mx[threadIdx.x] = mx[threadIdx.x];
        s_my[threadIdx.x] = my[threadIdx.x];
        s_mm[threadIdx.x] = mm[threadIdx.x];
        s_mv[threadIdx.x] = mVis[threadIdx.x];
        s_fx[threadIdx.x] = 0.0;
        s_fy[threadIdx.x] = 0.0;
    }
    __syncthreads();
    for (int i = tid; i < nAst; i += stride) {
        double acx = 0.0, acy = 0.0;
        if (aVis[i]) {
            const double px = ax[i], py = ay[i], pm = am[i];
            for (int kk = 0; kk < M; kk++) {
                if (!s_mv[kk]) continue;
                const double dx = s_mx[kk] - px, dy = s_my[kk] - py;
                const double r2 = dx * dx + dy * dy + soft;
                const double f = G / (r2 * sqrt(r2));
                acx += f * s_mm[kk] * dx;
                acy += f * s_mm[kk] * dy;
                atomicAdd(&s_fx[kk], -f * pm * dx);
                atomicAdd(&s_fy[kk], -f * pm * dy);
            }
        }
        aAccX[i] = acx; aAccY[i] = acy;
    }
    __syncthreads();
    if (threadIdx.x < M) {
        atomicAdd(&backX[threadIdx.x], s_fx[threadIdx.x]);
        atomicAdd(&backY[threadIdx.x], s_fy[threadIdx.x]);
    }
    grid.sync();
    if (tid == 0) {
        for (int j = 0; j < M; j++) {           // eigener Slot: normale
            gs->backX[gpuId * M_MAX + j] = backX[j];   // Writes reichen
            gs->backY[gpuId * M_MAX + j] = backY[j];
        }
    }
    sys_barrier(gs, gpuId, nGpus, &barRound, grid, tid);
    if (tid == 0) {
        for (int i = 0; i < M; i++) {
            double acx = 0.0, acy = 0.0;
            for (int g = 0; g < nGpus; g++) {       // feste Reihenfolge ->
                acx += gs->backX[g * M_MAX + i];    // deterministisch auf
                acy += gs->backY[g * M_MAX + i];    // allen Karten identisch
            }
            if (mVis[i]) {
                for (int j = 0; j < M; j++) {
                    if (j == i || !mVis[j]) continue;
                    const double dx = mx[j] - mx[i], dy = my[j] - my[i];
                    const double r2 = dx * dx + dy * dy + soft;
                    const double f = G / (r2 * sqrt(r2));
                    acx += f * mm[j] * dx;
                    acy += f * mm[j] * dy;
                }
            }
            mAccX[i] = acx; mAccY[i] = acy;
        }
    }
    sys_barrier(gs, gpuId, nGpus, &barRound, grid, tid);

    const double wDts[3] = { w1, w0, w1 };

    // ================= Batch: K Raster-Samples pro Launch =================
    // HIERARCHISCHE ZEITSCHRITTE: dtH (grob, max. maxSubDt) wird nur von
    // massiv-x-massiv-Paaren gedrueckt — jede Karte berechnet ihn aus den
    // replizierten Massiven IDENTISCH, ohne Austausch. Pro dtH werden die
    // Asteroiden klassifiziert: tEnc/20 < dtH -> "heiss" (enge Begegnung).
    // Ruhige + Massive machen EIN Yoshida(dtH) wie bisher; jeder heisse
    // Asteroid integriert sein Segment in einer PRIVATEN, synchronisations-
    // freien Feinschleife (eigenes adaptives dt, Yoshida, Floor
    // maxSubDt/1000), die Massiv-Positionen linear im Segment
    // interpoliert. Bewusste Naeherungen ggue. dem Live-Algo: Massiv-
    // Interpolation statt Fein-Mitintegration, Gegenkraefte heisser Astis
    // nur an Segmentgrenzen (~1e-12 M_sun), Ruhige bleiben beim groben
    // Schritt — dafuer bricht EIN Sonnentaucher nicht mehr die Rate aller.
    for (int ks = 0; ks < K; ks++) {
        if (tid == 0) {
            ctrl->remaining = dtRaster;
            ctrl->done = 0;
        }
        grid.sync();

        for (;;) {
            // dtH: massiv-Paare (identisch auf jeder Karte) + Klemmen
            if (tid == 0) {
                if (ctrl->remaining <= 1e-12 ||
                    ctrl->guard++ >= maxSubSteps) {
                    ctrl->done = 1;
                } else {
                    double dtH = maxSubDt;
                    for (int i = 0; i < M; i++) {
                        if (!mVis[i]) continue;
                        for (int j = i + 1; j < M; j++) {
                            if (!mVis[j]) continue;
                            const double dx = mx[j] - mx[i];
                            const double dy = my[j] - my[i];
                            double dist = sqrt(dx * dx + dy * dy);
                            if (dist < 1e-12) dist = 1e-12;
                            const double dvx = mvx[j] - mvx[i];
                            const double dvy = mvy[j] - mvy[i];
                            const double vrel =
                                sqrt(dvx * dvx + dvy * dvy);
                            if (vrel > 1e-9) {
                                const double tEnc = dist / vrel / 20.0;
                                if (tEnc < dtH) dtH = tEnc;
                            }
                        }
                    }
                    if (dtH < maxSubDt / 1000.0) dtH = maxSubDt / 1000.0;
                    if (dtH > ctrl->remaining) dtH = ctrl->remaining;
                    ctrl->subDt = dtH;
                    ctrl->remaining -= dtH;
                }
            }
            grid.sync();
            if (ctrl->done) break;
            const double dtH = ctrl->subDt;

            // Klassifikation: heiss, wenn eine Begegnung mit einem
            // massiven Koerper das alte adaptDt unter dtH gezogen haette.
            for (int i = tid; i < nAst; i += stride) {
                unsigned char h = 0;
                if (aVis[i]) {
                    const double px = ax[i], py = ay[i];
                    const double pvx = avx[i], pvy = avy[i];
                    for (int kk = 0; kk < M; kk++) {
                        if (!mVis[kk]) continue;
                        const double dx = mx[kk] - px, dy = my[kk] - py;
                        double dist = sqrt(dx * dx + dy * dy);
                        if (dist < 1e-12) dist = 1e-12;
                        const double dvx = mvx[kk] - pvx;
                        const double dvy = mvy[kk] - pvy;
                        const double vrel = sqrt(dvx * dvx + dvy * dvy);
                        if (vrel > 1e-9 &&
                            dist / vrel / 20.0 < dtH) { h = 1; break; }
                    }
                }
                hot[i] = h;
            }
            // (kein sync noetig: Phase A liest hot[i] im selben Thread)

            for (int w = 0; w < 3; w++) {
                const double dt = wDts[w] * dtH;
                // Segment-Anfang der Massiven fuer die Interpolation
                if (tid == 0) {
                    for (int j = 0; j < M; j++) {
                        mx0[j] = mx[j]; my0[j] = my[j];
                    }
                }
                grid.sync();
                // Phase A: Positionen — nur RUHIGE Astis + Massive
                for (int i = tid; i < nAst; i += stride) {
                    if (!aVis[i] || hot[i]) continue;
                    ax[i] += avx[i] * dt + 0.5 * aAccX[i] * dt * dt;
                    ay[i] += avy[i] * dt + 0.5 * aAccY[i] * dt * dt;
                }
                if (tid == 0) {
                    for (int i = 0; i < M; i++) {
                        if (!mVis[i]) continue;
                        mx[i] += mvx[i] * dt + 0.5 * mAccX[i] * dt * dt;
                        my[i] += mvy[i] * dt + 0.5 * mAccY[i] * dt * dt;
                    }
                    for (int j = 0; j < M; j++) {
                        backX[j] = 0.0; backY[j] = 0.0;
                    }
                }
                grid.sync();
                // Phase B: Beschleunigung + Velocity fuer RUHIGE;
                // ALLE Astis (auch heisse, an ihrer Segment-Position)
                // liefern Gegenkraefte auf die Massiven.
                if (threadIdx.x < M) {
                    s_mx[threadIdx.x] = mx[threadIdx.x];
                    s_my[threadIdx.x] = my[threadIdx.x];
                    s_fx[threadIdx.x] = 0.0;
                    s_fy[threadIdx.x] = 0.0;
                }
                __syncthreads();
                for (int i = tid; i < nAst; i += stride) {
                    if (!aVis[i]) continue;
                    const double px = ax[i], py = ay[i], pm = am[i];
                    double acx = 0.0, acy = 0.0;
                    for (int kk = 0; kk < M; kk++) {
                        if (!s_mv[kk]) continue;
                        const double dx = s_mx[kk] - px, dy = s_my[kk] - py;
                        const double r2 = dx * dx + dy * dy + soft;
                        const double f = G / (r2 * sqrt(r2));
                        acx += f * s_mm[kk] * dx;
                        acy += f * s_mm[kk] * dy;
                        atomicAdd(&s_fx[kk], -f * pm * dx);
                        atomicAdd(&s_fy[kk], -f * pm * dy);
                    }
                    if (!hot[i]) {
                        avx[i] += 0.5 * (aAccX[i] + acx) * dt;
                        avy[i] += 0.5 * (aAccY[i] + acy) * dt;
                        aAccX[i] = acx; aAccY[i] = acy;
                    }
                }
                __syncthreads();
                if (threadIdx.x < M) {
                    atomicAdd(&backX[threadIdx.x], s_fx[threadIdx.x]);
                    atomicAdd(&backY[threadIdx.x], s_fy[threadIdx.x]);
                }
                grid.sync();
                if (tid == 0) {
                    for (int j = 0; j < M; j++) {
                        gs->backX[gpuId * M_MAX + j] = backX[j];
                        gs->backY[gpuId * M_MAX + j] = backY[j];
                    }
                }
                sys_barrier(gs, gpuId, nGpus, &barRound, grid, tid);
                // Phase C: Massiv-Beschleunigung aus allen Partialsummen
                if (tid == 0) {
                    for (int i = 0; i < M; i++) {
                        double acx = 0.0, acy = 0.0;
                        for (int g = 0; g < nGpus; g++) {
                            acx += gs->backX[g * M_MAX + i];
                            acy += gs->backY[g * M_MAX + i];
                        }
                        if (mVis[i]) {
                            for (int j = 0; j < M; j++) {
                                if (j == i || !mVis[j]) continue;
                                const double dx = mx[j] - mx[i];
                                const double dy = my[j] - my[i];
                                const double r2 = dx * dx + dy * dy + soft;
                                const double f = G / (r2 * sqrt(r2));
                                acx += f * mm[j] * dx;
                                acy += f * mm[j] * dy;
                            }
                            mvx[i] += 0.5 * (mAccX[i] + acx) * dt;
                            mvy[i] += 0.5 * (mAccY[i] + acy) * dt;
                        }
                        mAccX[i] = acx; mAccY[i] = acy;
                    }
                }
                sys_barrier(gs, gpuId, nGpus, &barRound, grid, tid);

                // ---- Private Feinschleife der HEISSEN Astis ueber das
                // Segment [0, dt] (dt kann negativ sein: Yoshida-w0).
                // Massiv-Positionen linear interpoliert, kein Sync.
                for (int i = tid; i < nAst; i += stride) {
                    if (!aVis[i] || !hot[i]) continue;
                    double px = ax[i], py = ay[i];
                    double pvx = avx[i], pvy = avy[i];
                    const double sgn = dt >= 0.0 ? 1.0 : -1.0;
                    const double span = fabs(dt);
                    double tau = 0.0;
                    int guardF = 0;
                    while (tau < span - 1e-15 && guardF++ < 8000) {
                        // Feines dt aus tEnc gegen interpolierte Massive
                        const double al = tau / span;
                        double dtF = maxSubDt;
                        for (int kk = 0; kk < M; kk++) {
                            if (!s_mv[kk]) continue;
                            const double mxi = mx0[kk] +
                                (mx[kk] - mx0[kk]) * al;
                            const double myi = my0[kk] +
                                (my[kk] - my0[kk]) * al;
                            const double dx = mxi - px, dy = myi - py;
                            double dist = sqrt(dx * dx + dy * dy);
                            if (dist < 1e-12) dist = 1e-12;
                            const double dvx = mvx[kk] - pvx;
                            const double dvy = mvy[kk] - pvy;
                            const double vrel =
                                sqrt(dvx * dvx + dvy * dvy);
                            if (vrel > 1e-9) {
                                const double tE = dist / vrel / 20.0;
                                if (tE < dtF) dtF = tE;
                            }
                        }
                        if (dtF < maxSubDt / 1000.0)
                            dtF = maxSubDt / 1000.0;
                        if (dtF > span - tau) dtF = span - tau;
                        // Yoshida (3 Verlets) mit interpolierten Massiven
                        double tSub = tau;
                        for (int wf = 0; wf < 3; wf++) {
                            const double df = wDts[wf] * dtF * sgn;
                            const double a0 = tSub / span;
                            double acx = 0.0, acy = 0.0;
                            for (int kk = 0; kk < M; kk++) {
                                if (!s_mv[kk]) continue;
                                const double mxi = mx0[kk] +
                                    (mx[kk] - mx0[kk]) * a0;
                                const double myi = my0[kk] +
                                    (my[kk] - my0[kk]) * a0;
                                const double dx = mxi - px, dy = myi - py;
                                const double r2 = dx * dx + dy * dy + soft;
                                const double f = G / (r2 * sqrt(r2));
                                acx += f * s_mm[kk] * dx;
                                acy += f * s_mm[kk] * dy;
                            }
                            px += pvx * df + 0.5 * acx * df * df;
                            py += pvy * df + 0.5 * acy * df * df;
                            tSub += wDts[wf] * dtF;
                            const double a1 = tSub / span;
                            double ncx = 0.0, ncy = 0.0;
                            for (int kk = 0; kk < M; kk++) {
                                if (!s_mv[kk]) continue;
                                const double mxi = mx0[kk] +
                                    (mx[kk] - mx0[kk]) * a1;
                                const double myi = my0[kk] +
                                    (my[kk] - my0[kk]) * a1;
                                const double dx = mxi - px, dy = myi - py;
                                const double r2 = dx * dx + dy * dy + soft;
                                const double f = G / (r2 * sqrt(r2));
                                ncx += f * s_mm[kk] * dx;
                                ncy += f * s_mm[kk] * dy;
                            }
                            pvx += 0.5 * (acx + ncx) * df;
                            pvy += 0.5 * (acy + ncy) * df;
                        }
                        tau += dtF;
                    }
                    ax[i] = px; ay[i] = py;
                    avx[i] = pvx; avy[i] = pvy;
                    // Beschleunigung am Segment-Ende fuer die naechste
                    // grobe Phase A (falls der Asti wieder ruhig wird)
                    double acx = 0.0, acy = 0.0;
                    for (int kk = 0; kk < M; kk++) {
                        if (!s_mv[kk]) continue;
                        const double dx = mx[kk] - px, dy = my[kk] - py;
                        const double r2 = dx * dx + dy * dy + soft;
                        const double f = G / (r2 * sqrt(r2));
                        acx += f * s_mm[kk] * dx;
                        acy += f * s_mm[kk] * dy;
                    }
                    aAccX[i] = acx; aAccY[i] = acy;
                }
                grid.sync();
            }
        }

        // ---- Snapshot dieses Rasters: kompakt f32, Shard-Reihenfolge ----
        {
            const long long b = (long long)ks * 4 * nAst;
            for (int i = tid; i < nAst; i += stride) {
                snap[b + i] = (float)ax[i];
                snap[b + nAst + i] = (float)ay[i];
                snap[b + 2 * nAst + i] = (float)avx[i];
                snap[b + 3 * nAst + i] = (float)avy[i];
            }
            if (gpuId == 0 && tid == 0) {
                const long long bm = (long long)ks * 4 * M;
                for (int j = 0; j < M; j++) {
                    snapM[bm + j] = (float)mx[j];
                    snapM[bm + M + j] = (float)my[j];
                    snapM[bm + 2 * M + j] = (float)mvx[j];
                    snapM[bm + 3 * M + j] = (float)mvy[j];
                }
            }
        }
        grid.sync();
    }
}
"""

_FP64_RATIO = {
    (6, 0): 0.5, (7, 0): 0.5, (8, 0): 0.5, (9, 0): 0.5,
    (7, 5): 1 / 32, (8, 6): 1 / 64, (8, 9): 1 / 64,
}


def _f64_score(i: int) -> float:
    p = cp.cuda.runtime.getDeviceProperties(i)
    ratio = _FP64_RATIO.get((p["major"], p["minor"]), 1 / 32)
    return p["multiProcessorCount"] * p["clockRate"] * ratio


def pick_device() -> int:
    """GPU mit der hoechsten f64-Leistung waehlen (V100 vor RTX 8000)."""
    n = cp.cuda.runtime.getDeviceCount()
    return max(range(n), key=_f64_score)


def pick_devices() -> list[int]:
    """Physik-GPUs hardwareagnostisch waehlen: alle Karten, deren
    f64-Score mindestens 25% der besten erreicht — schwaechere Karten
    wuerden den synchronen Substep-Verbund nur ausbremsen (die Barrier
    wartet auf den langsamsten Shard). Absteigend nach Score sortiert;
    Karte [0] integriert die Massiven."""
    n = cp.cuda.runtime.getDeviceCount()
    scores = {i: _f64_score(i) for i in range(n)}
    best = max(scores.values())
    devs = sorted((i for i in range(n) if scores[i] >= 0.25 * best),
                  key=lambda i: -scores[i])
    return devs[:G_MAX]


def pick_detect_devices(exclude, rank: int = 0,
                        count: int = 1) -> list[int]:
    """GPUs AUSSERHALB der Physik-Karten fuer die ausgelagerte
    Kollisions-/Bounce-Erkennung. Die Erkennung laeuft in f32, da reicht
    auch eine f64-schwache Karte.

    Es werden bis zu `count` Karten zurueckgegeben — die Bounce-Suche
    teilt die Szene raeumlich unter ihnen auf. rank verteilt parallele
    Sessions round-robin ueber die freien Karten, damit zwei gleichzeitige
    Nutzer nicht mit derselben Karte anfangen. Leere Liste, wenn es keine
    freie Karte gibt (dann analysiert die Physik-Karte selbst)."""
    excl = set(exclude) if isinstance(exclude, (list, tuple, set)) \
        else {exclude}
    n = cp.cuda.runtime.getDeviceCount()
    cands = sorted((i for i in range(n) if i not in excl),
                   key=lambda i: -_f64_score(i))
    if not cands:
        return []
    count = min(count, len(cands))
    return [cands[(rank + i) % len(cands)] for i in range(count)]


class NBodyCuda:
    """Haelt Kernel + GPU-Buffer fuer einen Verbund aus 1..G_MAX Karten.

    devices: Liste der Physik-GPUs (devices[0] integriert die Massiven).
    Ein einzelnes int wird als 1-Karten-Verbund akzeptiert."""

    def __init__(self, devices):
        if isinstance(devices, int):
            devices = [devices]
        if not devices or len(devices) > G_MAX:
            raise ValueError(f"1..{G_MAX} Devices erwartet: {devices}")
        self.devices = list(devices)
        self.device = self.devices[0]      # Kompatibilitaet (Erkennung etc.)
        self._mods = {}
        self._kerns = {}
        self._block = 256
        for d in self.devices:
            with cp.cuda.Device(d):
                # Register-Limit: die private Feinschleife der heissen
                # Asteroiden wuerde sonst den Registerbedarf des GANZEN
                # Kernels aufblaehen und die Occupancy aller Phasen
                # halbieren; Spills treffen nur die wenigen Heissen.
                mod = cp.RawModule(
                    code=_SRC,
                    options=("--std=c++17", "--maxrregcount=64"),
                    enable_cooperative_groups=True)
                self._mods[d] = mod
                self._kerns[d] = mod.get_function("frame_kernel")

    def name(self) -> str:
        names = []
        for d in self.devices:
            p = cp.cuda.runtime.getDeviceProperties(d)
            names.append(p["name"].decode())
        return " + ".join(names)

    # ---------------- Zustand laden (gewichtete Shards) ----------------

    def load_state(self, x: np.ndarray, y: np.ndarray,
                   vx: np.ndarray, vy: np.ndarray,
                   mass: np.ndarray, visible: np.ndarray,
                   is_ast: np.ndarray) -> dict:
        """Vollzustand uebernehmen — bleibt danach GPU-resident.

        Asteroiden werden nach f64-Score gewichtet auf die Karten
        verteilt (contiguous Slices der Originalreihenfolge); massive
        Koerper sind auf jeder Karte repliziert. Rueckgabe ist ein
        Zustands-Dict PRO SESSION."""
        ast = is_ast != 0
        m_idx = np.flatnonzero(~ast)
        a_idx = np.flatnonzero(ast)
        m = len(m_idx)
        if m > M_MAX:
            raise ValueError(f"zu viele massive Koerper: {m} > {M_MAX}")
        n = len(x)
        n_ast = len(a_idx)
        ng = len(self.devices)

        # Gewichtete Partition der Asteroiden-Indizes — INTERLEAVED
        # (Round-Robin ueber 16 Gewichtsquanten pro Karte) statt
        # zusammenhaengender Slices: injizierte Wolken (zusammenhaengende
        # Indexbloecke) verteilen sich so gleichmaessig auf alle Karten,
        # sonst traegt EIN Shard alle heissen Feinschleifen und der Rest
        # wartet an der Segment-Barrier.
        weights = np.array([_f64_score(d) for d in self.devices])
        quanta = np.maximum(1, np.round(
            weights / weights.max() * 16).astype(np.int64))
        wheel = np.concatenate([np.full(q, g, np.int64)
                                for g, q in enumerate(quanta)])
        assign = wheel[np.arange(n_ast) % len(wheel)]
        shard_aidx = [a_idx[assign == g] for g in range(ng)]

        # GSync-Bereich: gemappter Host-Speicher bei >1 Karte (alle sehen
        # dieselben Bytes), sonst normaler Device-Speicher.
        # struct GSync: round_[G_MAX*16] u32 | minEnc[G_MAX*16] u64 |
        # backX/backY[G_MAX*M_MAX] f64 (+ Alignment-Polster)
        gs_bytes = 4 * G_MAX * 16 + 8 * G_MAX * 16 + 2 * 8 * G_MAX * M_MAX
        gs_bytes += (-gs_bytes) % 8 + 64
        if ng > 1:
            # cudaHostAllocPortable(1) | cudaHostAllocMapped(2): der
            # Bereich ist in ALLEN Karten-Kontexten zero-copy erreichbar.
            gs_host_ptr = cp.cuda.runtime.hostAlloc(gs_bytes, 3)
            ctypes.memset(gs_host_ptr, 0, gs_bytes)
        else:
            gs_host_ptr = None

        shards = []
        for g, d in enumerate(self.devices):
            sa = shard_aidx[g]
            with cp.cuda.Device(d):
                dd = cp.float64
                f64_host = np.concatenate([
                    x[sa], y[sa], vx[sa], vy[sa], mass[sa],
                    x[m_idx], y[m_idx], vx[m_idx], vy[m_idx], mass[m_idx]])
                vis_host = np.concatenate([visible[sa], visible[m_idx]])
                sh = {"dev": d, "gpu_id": g, "n_ast": len(sa),
                      "a_idx_h": sa,
                      "f64": cp.asarray(f64_host, dd),
                      "vis": cp.asarray(vis_host, cp.uint8),
                      "aaccx": cp.zeros(len(sa), dd),
                      "aaccy": cp.zeros(len(sa), dd),
                      "maccx": cp.zeros(m, dd), "maccy": cp.zeros(m, dd),
                      "backx": cp.zeros(m, dd), "backy": cp.zeros(m, dd),
                      "hot": cp.zeros(max(len(sa), 1), cp.uint8),
                      "mx0": cp.zeros(m, dd), "my0": cp.zeros(m, dd),
                      "ctrl": cp.zeros(5, dtype=cp.float64),
                      "snap": cp.empty(K_MAX * 4 * max(len(sa), 1),
                                       cp.float32)}
                if g == 0:
                    sh["snapM"] = cp.empty(K_MAX * 4 * max(m, 1),
                                           cp.float32)
                if ng > 1:
                    sh["gs"] = _device_view_of_host(gs_host_ptr, gs_bytes, d)
                else:
                    sh["gs"] = cp.zeros(gs_bytes // 8, dtype=cp.float64)
                shards.append(sh)

        # Inverse Abbildung Originalindex -> (Shard, Position) fuer
        # punktuelle Updates ohne FULL-Upload. Massive: Shard -1.
        inv_shard = np.full(n, -1, np.int8)
        inv_pos = np.zeros(n, np.int64)
        for g, sa in enumerate(shard_aidx):
            inv_shard[sa] = g
            inv_pos[sa] = np.arange(len(sa))
        inv_pos[m_idx] = np.arange(m)

        # gs_host_ptr wird bewusst nicht freigegeben: der Producer lebt
        # genau eine Film-Session, der Live-Pfad laedt selten neu (8 KB
        # pro FULL-Upload bei Multi-GPU sind vernachlaessigbar).
        st = {"N": n, "m": m, "n_ast": n_ast, "shards": shards,
              "m_idx_h": m_idx, "a_idx_h": a_idx,
              "inv_shard": inv_shard, "inv_pos": inv_pos,
              "gs_host_ptr": gs_host_ptr}
        return st

    # ---------------- Frame rechnen (Batch) ----------------

    def step_batch(self, st: dict, dt_raster_years: float,
                   k: int) -> np.ndarray:
        """K Raster-Samples in EINEM cooperative Launch je Karte rechnen.

        Rueckgabe: f32-Array (k, 4n) [x|y|vx|vy] in Originalreihenfolge.
        Die f64-Wahrheit bleibt auf den Karten. Erst werden ALLE Karten
        gelaunct (sie warten in der System-Barrier aufeinander!), dann
        eingesammelt."""
        if st is None:
            raise ValueError("kein Zustand geladen — FULL-Frame noetig")
        if not 1 <= k <= K_MAX:
            raise ValueError(f"k ausserhalb 1..{K_MAX}: {k}")
        m = st["m"]
        ng = len(st["shards"])
        for sh in st["shards"]:
            d = sh["dev"]
            n_ast = sh["n_ast"]
            with cp.cuda.Device(d):
                f64 = sh["f64"]
                o = 0
                views = []
                for ln in (n_ast,) * 5 + (m,) * 5:
                    views.append(f64[o:o + ln])
                    o += ln
                (g_ax, g_ay, g_avx, g_avy, _g_am,
                 g_mx, g_my, g_mvx, g_mvy, g_mm) = views
                grid = max(1, (n_ast + self._block - 1) // self._block)
                self._kerns[d](
                    (grid,), (self._block,),
                    (g_ax, g_ay, g_avx, g_avy,
                     sh["aaccx"], sh["aaccy"], _g_am,
                     sh["vis"][:n_ast],
                     g_mx, g_my, g_mvx, g_mvy, g_mm,
                     sh["vis"][n_ast:],
                     sh["maccx"], sh["maccy"],
                     sh["backx"], sh["backy"],
                     sh["hot"], sh["mx0"], sh["my0"],
                     sh["gs"], sh["ctrl"],
                     sh["snap"],
                     sh.get("snapM", sh["snap"]),
                     cp.int32(n_ast), cp.int32(m),
                     cp.int32(sh["gpu_id"]), cp.int32(ng),
                     cp.float64(G_AU), cp.float64(SOFTENING),
                     cp.float64(dt_raster_years), cp.int32(k),
                     cp.float64(MAX_SUB_DT_YEARS),
                     cp.int32(MAX_SUB_STEPS_PER_FRAME),
                     cp.float64(YOSHIDA_W1), cp.float64(YOSHIDA_W0)))
        # Einsammeln + Host-Scatter in Originalreihenfolge
        n = st["N"]
        out = np.empty((k, 4 * n), dtype=np.float32)
        for sh in st["shards"]:
            n_ast = sh["n_ast"]
            with cp.cuda.Device(sh["dev"]):
                snap = cp.asnumpy(sh["snap"][:k * 4 * n_ast]) \
                    .reshape(k, 4, n_ast)
                if sh["gpu_id"] == 0:
                    snapm = cp.asnumpy(sh["snapM"][:k * 4 * m]) \
                        .reshape(k, 4, m)
            sa = sh["a_idx_h"]
            for f in range(4):
                out[:, f * n + sa] = snap[:, f, :]
            if sh["gpu_id"] == 0:
                mi = st["m_idx_h"]
                for f in range(4):
                    out[:, f * n + mi] = snapm[:, f, :]
        return out

    def step(self, st: dict, dt_years: float) -> np.ndarray:
        """Ein einzelnes Sample (Kompatibilitaets-API fuer den
        Live-CUDA-Pfad des Servers)."""
        return self.step_batch(st, dt_years, 1)[0]

    # ---------------- Punktuelle Zustands-Updates ----------------

    def _flat(self, st: dict, sh: dict, field: int, pos):
        n_ast = sh["n_ast"]
        return field * n_ast + pos

    def apply_updates(self, st: dict, idx: np.ndarray,
                      vals: np.ndarray) -> None:
        """x/y/vx/vy ABSOLUT setzen (Scatter). idx: Originalindizes."""
        self._scatter(st, idx, vals, add=False)

    def apply_deltas(self, st: dict, idx: np.ndarray,
                     deltas: np.ndarray) -> None:
        """Additive x/y/vx/vy-Korrekturen (Bounce-Impulse + Push).
        idx MUSS eindeutig sein (Fancy-Index-Add)."""
        self._scatter(st, idx, deltas, add=True)

    def _scatter(self, st: dict, idx: np.ndarray, vals: np.ndarray,
                 add: bool) -> None:
        inv_shard = st["inv_shard"][idx]
        inv_pos = st["inv_pos"][idx]
        m = st["m"]
        for sh in st["shards"]:
            g = sh["gpu_id"]
            n_ast = sh["n_ast"]
            base = 5 * n_ast
            sel = inv_shard == g
            # Massive (Shard -1) wohnen auf ALLEN Karten -> ueberall
            # anwenden, damit die Repliken identisch bleiben.
            msel = inv_shard == -1
            flats = []
            values = []
            for f in range(4):
                if sel.any():
                    flats.append(f * n_ast + inv_pos[sel])
                    values.append(vals[sel, f])
                if msel.any():
                    flats.append(base + f * m + inv_pos[msel])
                    values.append(vals[msel, f])
            if not flats:
                continue
            with cp.cuda.Device(sh["dev"]):
                gi = cp.asarray(np.concatenate(flats))
                gv = cp.asarray(np.concatenate(values).astype(np.float64))
                if add:
                    sh["f64"][gi] = sh["f64"][gi] + gv
                else:
                    sh["f64"][gi] = gv

    def apply_body_state(self, st: dict, idx: int,
                         x: float, y: float, vx: float, vy: float,
                         mass: float) -> None:
        """Einen Koerper komplett setzen (x, y, vx, vy, Masse) — fuer
        Kollisions-Merges im Film-Producer."""
        g = int(st["inv_shard"][idx])
        pos = int(st["inv_pos"][idx])
        m = st["m"]
        vals = np.asarray([x, y, vx, vy, mass])
        for sh in st["shards"]:
            n_ast = sh["n_ast"]
            base = 5 * n_ast
            if g >= 0 and sh["gpu_id"] == g:
                flat = [f * n_ast + pos for f in range(5)]
            elif g == -1:
                flat = [base + f * m + pos for f in range(5)]
            else:
                continue
            with cp.cuda.Device(sh["dev"]):
                sh["f64"][cp.asarray(np.asarray(flat, np.int64))] = \
                    cp.asarray(vals)

    def deactivate_body(self, st: dict, idx: int) -> None:
        """Koerper einfrieren und kraftlos machen: Masse 0, visible 0."""
        g = int(st["inv_shard"][idx])
        pos = int(st["inv_pos"][idx])
        m = st["m"]
        for sh in st["shards"]:
            n_ast = sh["n_ast"]
            with cp.cuda.Device(sh["dev"]):
                if g >= 0 and sh["gpu_id"] == g:
                    sh["f64"][4 * n_ast + pos] = 0.0
                    sh["vis"][pos] = 0
                elif g == -1:
                    sh["f64"][5 * n_ast + 4 * m + pos] = 0.0
                    sh["vis"][n_ast + pos] = 0

    def export_f64(self, st: dict) -> np.ndarray:
        """Exakten f64-Zustand [x|y|vx|vy] (4n) in Originalreihenfolge
        exportieren — fuer die Engine-Uebergabe (Film-Stop/Dump)."""
        n = st["N"]
        m = st["m"]
        out4 = np.empty(4 * n, dtype="<f8")
        for sh in st["shards"]:
            n_ast = sh["n_ast"]
            with cp.cuda.Device(sh["dev"]):
                f64 = cp.asnumpy(sh["f64"])
            sa = sh["a_idx_h"]
            base = 5 * n_ast
            for f in range(4):
                out4[f * n + sa] = f64[f * n_ast:(f + 1) * n_ast]
            if sh["gpu_id"] == 0:
                mi = st["m_idx_h"]
                for f in range(4):
                    out4[f * n + mi] = f64[base + f * m:base + (f + 1) * m]
        return out4


def _device_view_of_host(host_ptr: int, nbytes: int, device: int):
    """Gemappten Host-Speicher als Device-Pointer der jeweiligen Karte
    ansprechen (zero-copy ueber PCIe) — Traeger der System-Barrier.
    Unter UVA (64-bit, alle modernen Karten) ist der Host-Pointer eines
    cudaHostAlloc(Portable|Mapped)-Bereichs direkt als Device-Pointer
    gueltig — cudaHostGetDevicePointer waere ein No-op."""
    with cp.cuda.Device(device):
        mem = cp.cuda.UnownedMemory(host_ptr, nbytes, owner=None)
        return cp.ndarray((nbytes // 8,), dtype=cp.float64,
                          memptr=cp.cuda.MemoryPointer(mem, 0))
