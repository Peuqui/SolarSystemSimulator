"""CUDA-N-Body-Kernel (f64) fuer den Sonnensystem-Simulator.

Spiegelt exakt die Physik des JS-Workers in index.html:
- Yoshida-Integrator 4. Ordnung (3 gewichtete Velocity-Verlets pro Schritt)
- adaptiver Substep aus den massiven Paaren (tEnc/20, Floor MAX_SUB_DT/1000)
- Kraefte: massive×massive + massive×Asteroid (symmetrisch, Softening 1e-6),
  Asteroid×Asteroid vernachlaessigt
- ausgeschaltete Koerper (visible=0) sind eingefroren

Der komplette Frame (alle Substeps) laeuft in EINEM Kernel-Launch mit
Cooperative-Groups-Grid-Sync — dadurch faellt der Launch-Overhead weg,
der eine naive GPU-Portierung auf JS-Worker-Niveau ausbremst.

Asteroiden liegen in globalen Arrays (Grid-Stride, beliebiges N).
Massive Koerper (bis M_MAX) verwaltet Thread 0
ihre Gegenkraefte werden
blockweise in Shared Memory reduziert und per atomicAdd aufsummiert.
"""
from __future__ import annotations

# Alle CUDA-Abhaengigkeiten (Header, NVRTC, libcudadevrt) kommen aus den
# nvidia-*-cu12-Wheels des venv (cupy-cuda12x[ctk]) — kein System-Toolkit
# noetig. CuPys Pfadsuche kennt beim gesplitteten cu12-Wheel-Layout aber
# keinen CUDA-Root und findet libcudadevrt.a (Cooperative-Groups-Linking)
# nicht — der memoisierte Pfad wird daher direkt gesetzt. (Interne API,
# gilt fuer cupy 14.x; bei einem CuPy-Upgrade pruefen.)
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

struct Ctrl {           // Steuerzustand des adaptiven Frame-Loops
    double remaining;   // verbleibende Frame-Zeit (Jahre)
    double subDt;       // aktueller Substep
    int done;
    int guard;
};

// Verlet-Teilschritt fuer die massiven Koerper — nur Thread 0.
// mAcc haelt die aktuelle Beschleunigung (Eingang: gueltig, Ausgang: neu).
__device__ void massivePositions(
    double* mx, double* my, const double* mvx, const double* mvy,
    const double* mAccX, const double* mAccY, const unsigned char* mVis,
    const int M, const double dt)
{
    for (int i = 0; i < M; i++) {
        if (!mVis[i]) continue;
        mx[i] += mvx[i] * dt + 0.5 * mAccX[i] * dt * dt;
        my[i] += mvy[i] * dt + 0.5 * mAccY[i] * dt * dt;
    }
}

extern "C" __global__ void frame_kernel(
    // Asteroiden [nAst]
    double* __restrict__ ax, double* __restrict__ ay,
    double* __restrict__ avx, double* __restrict__ avy,
    double* __restrict__ aAccX, double* __restrict__ aAccY,
    const double* __restrict__ am,
    const unsigned char* __restrict__ aVis,
    // massive Koerper [M]
    double* mx, double* my, double* mvx, double* mvy,
    const double* __restrict__ mm,
    const unsigned char* __restrict__ mVis,
    double* mAccX, double* mAccY,        // aktuelle Beschleunigung massive
    double* backX, double* backY,        // Gegenkraft-Akkumulatoren [M]
    Ctrl* ctrl,
    const int nAst, const int M,
    const double G, const double soft,
    const double dtFrame, const double maxSubDt,
    const int maxSubSteps,
    const double w1, const double w0)
{
    cg::grid_group grid = cg::this_grid();
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    __shared__ double s_mx[M_MAX], s_my[M_MAX], s_mm[M_MAX];
    __shared__ unsigned char s_mv[M_MAX];
    __shared__ double s_fx[M_MAX], s_fy[M_MAX];

    // ---- Initiale Beschleunigungen (wie computeAccel() im JS-Worker) ----
    if (tid == 0) {
        ctrl->remaining = dtFrame;
        ctrl->done = 0;
        ctrl->guard = 0;
        for (int k = 0; k < M; k++) { backX[k] = 0.0; backY[k] = 0.0; }
    }
    grid.sync();
    if (threadIdx.x < M) {
        s_mx[threadIdx.x] = mx[threadIdx.x];
        s_my[threadIdx.x] = my[threadIdx.x];
        s_mm[threadIdx.x] = mm[threadIdx.x];
        s_mv[threadIdx.x] = mVis[threadIdx.x];
    }
    if (threadIdx.x < M) { s_fx[threadIdx.x] = 0.0; s_fy[threadIdx.x] = 0.0; }
    __syncthreads();
    for (int i = tid; i < nAst; i += stride) {
        double acx = 0.0, acy = 0.0;
        if (aVis[i]) {
            const double px = ax[i], py = ay[i], pm = am[i];
            for (int k = 0; k < M; k++) {
                if (!s_mv[k]) continue;
                const double dx = s_mx[k] - px, dy = s_my[k] - py;
                const double r2 = dx * dx + dy * dy + soft;
                const double f = G / (r2 * sqrt(r2));
                acx += f * s_mm[k] * dx;
                acy += f * s_mm[k] * dy;
                atomicAdd(&s_fx[k], -f * pm * dx);
                atomicAdd(&s_fy[k], -f * pm * dy);
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
        for (int i = 0; i < M; i++) {
            double acx = backX[i], acy = backY[i];
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
    grid.sync();

    // ---- Frame-Loop: adaptiver Substep, Yoshida = 3 Verlets ----
    const double wDts[3] = { w1, w0, w1 };
    for (;;) {
        // Substep bestimmen (Thread 0, Worker-adaptDt-Semantik: nur
        // massive Paare, tEnc/20, Floor maxSubDt/1000)
        if (tid == 0) {
            if (ctrl->remaining <= 1e-12 || ctrl->guard++ >= maxSubSteps) {
                ctrl->done = 1;
            } else {
                double dt = maxSubDt;
                for (int i = 0; i < M; i++) {
                    if (!mVis[i]) continue;
                    for (int j = i + 1; j < M; j++) {
                        if (!mVis[j]) continue;
                        const double dx = mx[j] - mx[i], dy = my[j] - my[i];
                        double dist = sqrt(dx * dx + dy * dy);
                        if (dist < 1e-12) dist = 1e-12;
                        const double dvx = mvx[j] - mvx[i], dvy = mvy[j] - mvy[i];
                        const double vrel = sqrt(dvx * dvx + dvy * dvy);
                        if (vrel > 1e-9) {
                            const double tEnc = dist / vrel;
                            if (tEnc / 20.0 < dt) dt = tEnc / 20.0;
                        }
                    }
                }
                if (dt < maxSubDt / 1000.0) dt = maxSubDt / 1000.0;
                if (dt > ctrl->remaining) dt = ctrl->remaining;
                ctrl->subDt = dt;
                ctrl->remaining -= dt;
            }
        }
        grid.sync();
        if (ctrl->done) break;
        const double subDt = ctrl->subDt;

        for (int w = 0; w < 3; w++) {
            const double dt = wDts[w] * subDt;
            // Phase A: Positionen (Asteroiden stride, massive Thread 0);
            // Gegenkraft-Akkus vorab nullen.
            for (int i = tid; i < nAst; i += stride) {
                if (!aVis[i]) continue;
                ax[i] += avx[i] * dt + 0.5 * aAccX[i] * dt * dt;
                ay[i] += avy[i] * dt + 0.5 * aAccY[i] * dt * dt;
            }
            if (tid == 0) {
                massivePositions(mx, my, mvx, mvy, mAccX, mAccY, mVis, M, dt);
                for (int k = 0; k < M; k++) { backX[k] = 0.0; backY[k] = 0.0; }
            }
            grid.sync();
            // Phase B: neue Asteroid-Beschleunigung + Velocity-Update
            // (haengt nur von neuen Positionen ab) + Gegenkraft-Reduktion.
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
                for (int k = 0; k < M; k++) {
                    if (!s_mv[k]) continue;
                    const double dx = s_mx[k] - px, dy = s_my[k] - py;
                    const double r2 = dx * dx + dy * dy + soft;
                    const double f = G / (r2 * sqrt(r2));
                    acx += f * s_mm[k] * dx;
                    acy += f * s_mm[k] * dy;
                    atomicAdd(&s_fx[k], -f * pm * dx);
                    atomicAdd(&s_fy[k], -f * pm * dy);
                }
                avx[i] += 0.5 * (aAccX[i] + acx) * dt;
                avy[i] += 0.5 * (aAccY[i] + acy) * dt;
                aAccX[i] = acx; aAccY[i] = acy;
            }
            __syncthreads();
            if (threadIdx.x < M) {
                atomicAdd(&backX[threadIdx.x], s_fx[threadIdx.x]);
                atomicAdd(&backY[threadIdx.x], s_fy[threadIdx.x]);
            }
            grid.sync();
            // Phase C: massive Beschleunigung + Velocity (Thread 0)
            if (tid == 0) {
                for (int i = 0; i < M; i++) {
                    double acx = backX[i], acy = backY[i];
                    if (mVis[i]) {
                        for (int j = 0; j < M; j++) {
                            if (j == i || !mVis[j]) continue;
                            const double dx = mx[j] - mx[i], dy = my[j] - my[i];
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
            grid.sync();
        }
    }
}
"""

_FP64_RATIO = {
    (6, 0): 0.5, (7, 0): 0.5, (8, 0): 0.5, (9, 0): 0.5,
    (7, 5): 1 / 32, (8, 6): 1 / 64, (8, 9): 1 / 64,
}


def pick_device() -> int:
    """GPU mit der hoechsten f64-Leistung waehlen (V100 vor RTX 8000)."""
    best, best_score = 0, -1.0
    for i in range(cp.cuda.runtime.getDeviceCount()):
        p = cp.cuda.runtime.getDeviceProperties(i)
        cc = (p["major"], p["minor"])
        ratio = _FP64_RATIO.get(cc, 1 / 32)
        score = p["multiProcessorCount"] * p["clockRate"] * ratio
        if score > best_score:
            best, best_score = i, score
    return best


class NBodyCuda:
    """Haelt Kernel + GPU-Buffer und rechnet einen Frame pro advance()."""

    def __init__(self, device: int):
        self.device = device
        with cp.cuda.Device(device):
            self._mod = cp.RawModule(
                code=_SRC, options=("--std=c++17",),
                enable_cooperative_groups=True)
            self._kern = self._mod.get_function("frame_kernel")
            self._block = 256

    def name(self) -> str:
        return cp.cuda.runtime.getDeviceProperties(self.device)["name"].decode()

    def load_state(self, x: np.ndarray, y: np.ndarray,
                   vx: np.ndarray, vy: np.ndarray,
                   mass: np.ndarray, visible: np.ndarray,
                   is_ast: np.ndarray) -> dict:
        """Vollzustand vom Client uebernehmen — bleibt danach GPU-resident.

        Rueckgabe ist ein Zustands-Dict, das der Server PRO VERBINDUNG
        haelt und an step() uebergibt — mehrere gleichzeitige Clients
        (z. B. lokal + remote) stoeren sich so nicht gegenseitig.

        Der Browser schickt den Vollzustand nur noch bei Mutationen
        (Kollisionen, Injects, Edits)
        normale Frames sind reine
        step()-Aufrufe ohne Upload (server-authoritativer Zustand).
        """
        ast = is_ast != 0
        m_idx = np.flatnonzero(~ast)
        a_idx = np.flatnonzero(ast)
        m = len(m_idx)
        if m > M_MAX:
            raise ValueError(f"zu viele massive Koerper: {m} > {M_MAX}")
        with cp.cuda.Device(self.device):
            d = cp.float64
            n_ast = len(a_idx)
            # EIN H2D-Transfer: alle f64-Felder in einem Block, auf der GPU
            # per View zerschnitten.
            f64_host = np.concatenate([
                x[a_idx], y[a_idx], vx[a_idx], vy[a_idx], mass[a_idx],
                x[m_idx], y[m_idx], vx[m_idx], vy[m_idx], mass[m_idx]])
            vis_host = np.concatenate([visible[a_idx], visible[m_idx]])
            g_f64 = cp.asarray(f64_host, d)
            st = {"N": len(x), "n_ast": n_ast, "m": m,
                  "f64": g_f64, "vis": cp.asarray(vis_host, cp.uint8),
                  "a_idx": cp.asarray(a_idx, cp.int32),
                  "m_idx": cp.asarray(m_idx, cp.int32),
                  "aaccx": cp.zeros(n_ast, d), "aaccy": cp.zeros(n_ast, d),
                  "maccx": cp.zeros(m, d), "maccy": cp.zeros(m, d),
                  "backx": cp.zeros(m, d), "backy": cp.zeros(m, d),
                  "out_f32": cp.empty(4 * len(x), cp.float32),
                  # Kernel-Steuerpuffer PRO SESSION — mehrere gleichzeitige
                  # Producer (mehrere Clients) duerfen sich den Loop-Zustand
                  # nicht teilen.
                  "ctrl": cp.zeros(4, dtype=cp.float64)}
            # Inverse Abbildung Originalindex -> (Kategorie, Position) fuer
            # punktuelle Delta-Updates (Bounces) ohne FULL-Upload.
            n = len(x)
            inv_kind = np.zeros(n, np.uint8)
            inv_kind[a_idx] = 1
            inv_pos = np.zeros(n, np.int64)
            inv_pos[a_idx] = np.arange(len(a_idx))
            inv_pos[m_idx] = np.arange(m)
            st["inv_kind"] = inv_kind
            st["inv_pos"] = inv_pos
            return st

    def apply_updates(self, st: dict, idx: np.ndarray,
                      vals: np.ndarray) -> None:
        """Punktuelle x/y/vx/vy-Updates (Bounces) in den residenten Zustand
        schreiben — ein Scatter statt FULL-Upload. idx: Originalindizes,
        vals: (k, 4) mit [x, y, vx, vy]."""
        if st is None:
            raise ValueError("kein Zustand geladen — FULL-Frame noetig")
        n_ast, m = st["n_ast"], st["m"]
        kind = st["inv_kind"][idx]
        pos = st["inv_pos"][idx]
        base = 5 * n_ast
        flats = []
        values = []
        for f in range(4):
            flats.append(np.where(kind == 1, f * n_ast + pos, base + f * m + pos))
            values.append(vals[:, f])
        with cp.cuda.Device(self.device):
            st["f64"][cp.asarray(np.concatenate(flats))] = \
                cp.asarray(np.concatenate(values))

    def step(self, st: dict, dt_years: float) -> np.ndarray:
        """Einen Frame auf dem residenten Zustand rechnen.

        Rueckgabe: f32-Array [x|y|vx|vy] in Originalreihenfolge des Clients
        (kompakt fuers Rendering
        die f64-Wahrheit bleibt auf der GPU).
        """
        if st is None:
            raise ValueError("kein Zustand geladen — FULL-Frame noetig")
        with cp.cuda.Device(self.device):
            n_ast, m, g_f64 = st["n_ast"], st["m"], st["f64"]
            o = 0
            g_ax  = g_f64[o:o + n_ast]
            o += n_ast
            g_ay  = g_f64[o:o + n_ast]
            o += n_ast
            g_avx = g_f64[o:o + n_ast]
            o += n_ast
            g_avy = g_f64[o:o + n_ast]
            o += n_ast
            g_am  = g_f64[o:o + n_ast]
            o += n_ast
            g_mx  = g_f64[o:o + m]
            o += m
            g_my  = g_f64[o:o + m]
            o += m
            g_mvx = g_f64[o:o + m]
            o += m
            g_mvy = g_f64[o:o + m]
            o += m
            g_mm  = g_f64[o:o + m]
            o += m
            g_avis = st["vis"][:n_ast]
            g_mvis = st["vis"][n_ast:]
            g_aaccx, g_aaccy = st["aaccx"], st["aaccy"]
            g_maccx, g_maccy = st["maccx"], st["maccy"]
            g_backx, g_backy = st["backx"], st["backy"]
            # CuPy klemmt das Grid bei Cooperative Launches selbst auf das
            # Residenz-Limit — der Grid-Stride-Loop im Kernel deckt den Rest.
            grid = max(1, (n_ast + self._block - 1) // self._block)
            self._kern(
                (grid,), (self._block,),
                (g_ax, g_ay, g_avx, g_avy, g_aaccx, g_aaccy, g_am, g_avis,
                 g_mx, g_my, g_mvx, g_mvy, g_mm, g_mvis,
                 g_maccx, g_maccy, g_backx, g_backy,
                 st["ctrl"],
                 cp.int32(n_ast), cp.int32(m),
                 cp.float64(G_AU), cp.float64(SOFTENING),
                 cp.float64(dt_years), cp.float64(MAX_SUB_DT_YEARS),
                 cp.int32(MAX_SUB_STEPS_PER_FRAME),
                 cp.float64(YOSHIDA_W1), cp.float64(YOSHIDA_W0)))

            # Ausgabe fuer den Client: f32 [x|y|vx|vy] in Originalreihenfolge,
            # auf der GPU per Scatter zusammengesetzt, EIN D2H-Transfer.
            # Die f64-Wahrheit bleibt resident auf der Karte.
            n = st["N"]
            out = st["out_f32"]
            ga, gm = st["a_idx"], st["m_idx"]
            f32 = cp.float32
            xs = out[0:n]
            ys = out[n:2*n]
            vxs = out[2*n:3*n]
            vys = out[3*n:4*n]
            xs[ga] = g_ax.astype(f32)
            xs[gm] = g_mx.astype(f32)
            ys[ga] = g_ay.astype(f32)
            ys[gm] = g_my.astype(f32)
            vxs[ga] = g_avx.astype(f32)
            vxs[gm] = g_mvx.astype(f32)
            vys[ga] = g_avy.astype(f32)
            vys[gm] = g_mvy.astype(f32)
            return cp.asnumpy(out)
