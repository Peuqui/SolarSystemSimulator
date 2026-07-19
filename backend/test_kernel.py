"""Korrektheits- und Leistungstest des CUDA-Kernels gegen eine NumPy-
Referenzimplementierung mit identischem Algorithmus (Yoshida + adaptDt).

Aufruf: ./venv/bin/python backend/test_kernel.py
"""
from __future__ import annotations

import time

import numpy as np

from nbody_kernel import (G_AU, MAX_SUB_DT_YEARS, SOFTENING, YOSHIDA_W0,
                          YOSHIDA_W1, NBodyCuda, pick_device)


def reference_advance(dt_years, x, y, vx, vy, mass, visible, is_ast):
    """CPU-Referenz: exakt die Worker-Physik (vektorisiert mit NumPy)."""
    x, y, vx, vy = (a.astype(np.float64).copy() for a in (x, y, vx, vy))
    vis = visible != 0
    ast = is_ast != 0
    mas_i = np.flatnonzero(~ast & vis)

    def accel():
        ax = np.zeros_like(x); ay = np.zeros_like(y)
        for i in mas_i:
            dx = x - x[i]; dy = y - y[i]
            r2 = dx * dx + dy * dy + SOFTENING
            f = G_AU / (r2 * np.sqrt(r2))
            f[i] = 0.0
            f[~vis] = 0.0
            # Ziel: alle sichtbaren Koerper (ausser sich selbst) spueren i;
            # i spuert die Gegenkraefte. Asteroid×Asteroid gibt es nicht,
            # weil nur massive i als Quelle auftreten. Doppelzaehlung
            # massiver Paare vermeiden: nur symmetrisch fuer j>i addieren.
            sel = np.ones_like(f, dtype=bool)
            sel[~vis] = False
            sel[i] = False
            sel[mas_i[mas_i < i]] = False    # Paar schon behandelt
            ax[sel] += f[sel] * mass[i] * (-dx[sel])
            ay[sel] += f[sel] * mass[i] * (-dy[sel])
            ax[i] += np.sum(f[sel] * mass[sel] * dx[sel])
            ay[i] += np.sum(f[sel] * mass[sel] * dy[sel])
        return ax, ay

    def adapt_dt():
        dt = MAX_SUB_DT_YEARS
        for a_i in range(len(mas_i)):
            for b_i in range(a_i + 1, len(mas_i)):
                i, j = mas_i[a_i], mas_i[b_i]
                dist = max(np.hypot(x[j] - x[i], y[j] - y[i]), 1e-12)
                vrel = np.hypot(vx[j] - vx[i], vy[j] - vy[i])
                if vrel > 1e-9:
                    dt = min(dt, dist / vrel / 20.0)
        return max(dt, MAX_SUB_DT_YEARS / 1000.0)

    ax, ay = accel()

    def verlet(dt):
        nonlocal ax, ay
        x[vis] += vx[vis] * dt + 0.5 * ax[vis] * dt * dt
        y[vis] += vy[vis] * dt + 0.5 * ay[vis] * dt * dt
        nax, nay = accel()
        vx[vis] += 0.5 * (ax[vis] + nax[vis]) * dt
        vy[vis] += 0.5 * (ay[vis] + nay[vis]) * dt
        ax, ay = nax, nay

    remaining, guard = dt_years, 0
    while remaining > 1e-12 and guard < 100_000:
        guard += 1
        sub = min(remaining, adapt_dt())
        for w in (YOSHIDA_W1, YOSHIDA_W0, YOSHIDA_W1):
            verlet(w * sub)
        remaining -= sub
    return x, y, vx, vy


def make_system(n_ast: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    n = 10 + n_ast
    x = np.zeros(n); y = np.zeros(n); vx = np.zeros(n); vy = np.zeros(n)
    mass = np.full(n, 1e-6); mass[0] = 1.0
    is_ast = np.zeros(n, np.uint8); is_ast[10:] = 1
    visible = np.ones(n, np.uint8)
    visible[3] = 0                      # ein eingefrorener Planet als Edge-Case
    rp = np.arange(1, 10, dtype=np.float64)
    x[1:10] = rp
    vy[1:10] = np.sqrt(G_AU / rp)
    r = 2 + 2 * rng.random(n_ast)
    th = rng.random(n_ast) * 2 * np.pi
    x[10:] = r * np.cos(th); y[10:] = r * np.sin(th)
    v = np.sqrt(G_AU / r)
    vx[10:] = -v * np.sin(th); vy[10:] = v * np.cos(th)
    mass[10:] = 1e-12
    return x, y, vx, vy, mass, visible, is_ast


def main() -> None:
    dev = pick_device()
    sim = NBodyCuda(dev)
    print(f"Device: {dev} ({sim.name()})")

    # --- Korrektheit: 5 Tage Frame, 200 Asteroiden, gegen CPU-Referenz
    state = make_system(200)
    dt = 5 / 365.25
    gx, gy, gvx, gvy = sim.advance(dt, *state)
    rx, ry, rvx, rvy = reference_advance(dt, *state)
    err_pos = np.max(np.abs(np.concatenate([gx - rx, gy - ry])))
    err_vel = np.max(np.abs(np.concatenate([gvx - rvx, gvy - rvy])))
    frozen_ok = gx[3] == state[0][3] and gy[3] == state[1][3]
    print(f"max |Δpos| = {err_pos:.3e} AU, max |Δvel| = {err_vel:.3e} AU/Jahr, "
          f"eingefrorener Koerper unveraendert: {frozen_ok}")
    assert err_pos < 1e-9 and err_vel < 1e-9 and frozen_ok, "Physik weicht ab!"

    # --- Energie-Sanity: 1 Jahr Sonnensystem ohne Asteroiden
    state = make_system(0)
    sx, sy, svx, svy = state[0].copy(), state[1].copy(), state[2].copy(), state[3].copy()
    for _ in range(73):                      # 73 × 5 Tage ≈ 1 Jahr
        sx, sy, svx, svy = sim.advance(5 / 365.25, sx, sy, svx, svy,
                                       state[4], state[5], state[6])
    r_earth = np.hypot(sx[3] - sx[0], sy[3] - sy[0])
    print(f"Planet r=3 AU nach 1 Jahr: {np.hypot(sx[3]-sx[0], sy[3]-sy[0]):.6f} "
          f"(eingefroren, soll exakt 3.0)")

    # --- Benchmark: Tage/s wie im Browser-Vergleich (dt=50 Tage/Frame)
    for n_ast in (7000, 50000, 200000):
        state = make_system(n_ast)
        dtf = 50 / 365.25
        sim.advance(dtf, *state)             # Warmup
        t0 = time.perf_counter()
        frames = 20
        for _ in range(frames):
            sim.advance(dtf, *state)
        secs = time.perf_counter() - t0
        days_per_sec = frames * 50 / secs
        ms = secs / frames * 1000
        print(f"N={n_ast:7d}: {ms:7.1f} ms/Frame (dt=50 Tage) "
              f"→ {days_per_sec:7.0f} Tage/s  (JS-Worker N=7000: ~290)")


if __name__ == "__main__":
    main()
