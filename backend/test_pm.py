"""Particle-Mesh-Kraefte gegen all-pairs validieren.

PM ist eine NAEHERUNG — Cloud-in-Cell hat Gitterfehler im Prozentbereich,
kein 1e-5 wie ein exakter Kernel. Die Tests pruefen deshalb:

  1. Impulserhaltung: Σ m·a ≈ 0 (Newtons drittes Gesetz). Haengt NICHT an
     der Aufloesung, sondern an der Symmetrie des Kerns — ein starker
     Korrektheits-Check. Muss sehr klein sein.
  2. Fernes Paar gegen die analytische 1/r²-Kraft (r ≫ Zelle): PM muss die
     echte Newton-Kraft grossraeumig treffen.
  3. Zufallswolke gegen all-pairs bei ε ≫ h: dann glaettet das Softening
     ueber viele Zellen, CiC ist genau, und PM ≈ all-pairs (Median-Fehler
     unter ein paar Prozent).
  4. Dasselbe bei ε ≈ h (der Betriebsfall Softening = Gitterskala): groesser,
     aber beschraenkt.

Aufruf: ./venv/bin/python backend/test_pm.py
"""
from __future__ import annotations

import sys
import time

import cupy as cp
import numpy as np

from nbody_kernel import G_AU
from pm_kernel import (NBodyPM, build_force_kernels, grid_fuer,
                       pm_accelerations)


def allpairs_accel(x, y, m, eps2, G=G_AU):
    """Exakte O(N²)-Referenz auf der GPU. a_i = Σ_j G m_j (r_j−r_i)/
    (|Δ|²+ε²)^1,5. Der Selbstterm (j=i) traegt 0 bei (Δ=0)."""
    dx = x[None, :] - x[:, None]        # dx[i,j] = x_j − x_i
    dy = y[None, :] - y[:, None]
    r2 = dx * dx + dy * dy + eps2
    inv = 1.0 / (r2 * cp.sqrt(r2))
    ax = G * (m[None, :] * dx * inv).sum(axis=1)
    ay = G * (m[None, :] * dy * inv).sum(axis=1)
    return ax, ay


def test_impuls(rng):
    n = 5000
    x = cp.asarray(rng.uniform(-40000, 40000, n))
    y = cp.asarray(rng.uniform(-40000, 40000, n))
    m = cp.asarray(rng.uniform(1e6, 2e7, n))
    grid_n = 512
    x0, y0, h = grid_fuer(x, y, grid_n)
    eps2 = (2 * h) ** 2
    ax, ay = pm_accelerations(x, y, m, grid_n, x0, y0, h, eps2)
    # Gesamtimpuls-Aenderung gegen die typische Kraftgroesse
    px = float(cp.sum(m * ax))
    py = float(cp.sum(m * ay))
    skala = float(cp.sum(m * cp.hypot(ax, ay)))
    rel = np.hypot(px, py) / max(skala, 1e-30)
    print(f"1) Impulserhaltung: |Σ m·a| / Σ m·|a| = {rel:.2e}")
    return rel < 1e-3


def test_fernes_paar():
    grid_n = 512
    h = 100.0                      # AE je Zelle
    x0 = y0 = 0.0
    eps2 = (2 * h) ** 2
    m_val = 1e10
    r = 60 * h                     # 6000 AE — weit ueber Zelle und Softening
    cx = grid_n * h / 2
    x = cp.asarray([cx - r / 2, cx + r / 2])
    y = cp.asarray([cx, cx])
    m = cp.asarray([m_val, m_val])
    ax, ay = pm_accelerations(x, y, m, grid_n, x0, y0, h, eps2)
    pm_kraft = float(abs(ax[0]))
    analytisch = G_AU * m_val / (r * r)      # 1/r² (3D-Gravitation)
    rel = abs(pm_kraft - analytisch) / analytisch
    print(f"2) Fernes Paar (r={r:.0f} AE): PM {pm_kraft:.4e} vs "
          f"analytisch {analytisch:.4e}  → {rel*100:.2f} %")
    return rel < 0.02


def test_wolke_vs_allpairs(rng, eps_in_h, schwelle, label):
    n = 4000
    x = cp.asarray(rng.uniform(-40000, 40000, n))
    y = cp.asarray(rng.uniform(-40000, 40000, n))
    m = cp.asarray(rng.uniform(1e6, 2e7, n))
    grid_n = 512
    x0, y0, h = grid_fuer(x, y, grid_n)
    eps2 = (eps_in_h * h) ** 2

    ax_pm, ay_pm = pm_accelerations(x, y, m, grid_n, x0, y0, h, eps2)
    ax_ref, ay_ref = allpairs_accel(x, y, m, eps2)

    betrag_ref = cp.hypot(ax_ref, ay_ref)
    fehler = cp.hypot(ax_pm - ax_ref, ay_pm - ay_ref)
    rel = cp.asnumpy(fehler / cp.maximum(betrag_ref, 1e-30))
    median = float(np.median(rel))
    p90 = float(np.percentile(rel, 90))
    print(f"{label}: Median rel. Kraftfehler {median*100:.2f} %, "
          f"90. Perzentil {p90*100:.2f} %  (ε = {eps_in_h}·h)")
    return median < schwelle


def energie_impuls(x, y, vx, vy, m, eps2):
    """Gesamtenergie (KE + PE) und Impuls auf der GPU. PE ueber all-pairs
    mit demselben Softening wie die Kraft."""
    x = cp.asarray(x)
    y = cp.asarray(y)
    vx = cp.asarray(vx)
    vy = cp.asarray(vy)
    m = cp.asarray(m)
    ke = 0.5 * float(cp.sum(m * (vx * vx + vy * vy)))
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    r = cp.sqrt(dx * dx + dy * dy + eps2)
    mm = m[:, None] * m[None, :]
    # −G Σ_{i<j} m_i m_j / r  = −G/2 · (Σ_{i≠j} …) ; Diagonale i=j abziehen
    pe_voll = -G_AU * float(cp.sum(mm / r))
    pe_diag = -G_AU * float(cp.sum(m * m) / np.sqrt(eps2))
    pe = 0.5 * (pe_voll - pe_diag)
    px = float(cp.sum(m * vx))
    py = float(cp.sum(m * vy))
    return ke + pe, (px, py)


def test_dynamik(rng, dev):
    """Leapfrog ueber 100 Schritte mit NBodyPM — Impuls- und
    Energieerhaltung. Ein sub-virialer Haufen, der ueber die kurze Zeit
    (~0,1 dynamische Zeiten) kaum wandert, sodass das adaptive Gitter stabil
    bleibt. Impuls muss maschinengenau bleiben (symmetrische Kraft + KDK),
    die Energie im Prozentbereich (PM-Gitterfehler)."""
    n = 10000
    R = 30000.0
    Mtot = 5.6e9
    r = R * np.sqrt(rng.uniform(0, 1, n))
    th = rng.uniform(0, 2 * np.pi, n)
    x = r * np.cos(th)
    y = r * np.sin(th)
    m = np.full(n, Mtot / n)
    vvir = np.sqrt(G_AU * Mtot / R)
    ang = rng.uniform(0, 2 * np.pi, n)
    speed = 0.7 * vvir
    vx = speed * np.cos(ang)
    vy = speed * np.sin(ang)
    vx -= np.sum(m * vx) / Mtot          # Gesamtimpuls auf 0
    vy -= np.sum(m * vy) / Mtot

    kern = NBodyPM([dev], softening_zellen=1.5)   # grid_n auto aus N
    st = kern.load_state(x, y, vx, vy, m)         # setzt kern.grid_n
    _, _, h = grid_fuer(cp.asarray(x), cp.asarray(y), kern.grid_n, 4.0)
    eps2 = (1.5 * h) ** 2
    e0, p0 = energie_impuls(x, y, vx, vy, m, eps2)
    t_dyn = R / vvir
    dt = t_dyn / 1000.0                   # 100 Schritte ≈ 0,1 t_dyn
    kern.step_batch(st, dt, steps=100)
    f = kern.export_f64(st)
    e1, p1 = energie_impuls(f[0:n], f[n:2 * n], f[2 * n:3 * n],
                            f[3 * n:4 * n], m, eps2)

    e_drift = abs(e1 - e0) / abs(e0)
    p_rel = np.hypot(*p1) / max(np.hypot(np.sum(m * np.abs(vx)),
                                         np.sum(m * np.abs(vy))), 1e-30)
    print(f"5) Dynamik (100 Leapfrog-Schritte): Energie-Drift "
          f"{e_drift*100:.3f} %, Rest-Impuls {p_rel:.2e}")
    return e_drift < 0.03 and p_rel < 1e-6


def _zeit(fn, wdh=5):
    fn()
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(wdh):
        fn()
    cp.cuda.Stream.null.synchronize()
    return (time.perf_counter() - t0) / wdh * 1000  # ms


def bench_tempo(rng):
    """PM gegen all-pairs — der eigentliche Sinn. all-pairs ist O(N²) und
    sprengt bei grossem N den Speicher (N² Matrix), PM ist O(N log N) bei
    festem Gitter. Gezeigt wird, dass PM bei HALBER Million Koerpern noch
    schneller ist als all-pairs bei 15.000."""
    grid_n = 512
    kernels = None
    print("\nTempo (eine V100, ms je Kraftauswertung):")
    print(f"  {'N':>9}  {'all-pairs':>12}  {'PM':>10}")
    for n in (5000, 15000, 50000, 200000, 500000):
        x = cp.asarray(rng.uniform(-40000, 40000, n))
        y = cp.asarray(rng.uniform(-40000, 40000, n))
        m = cp.asarray(rng.uniform(1e6, 2e7, n))
        x0, y0, h = grid_fuer(x, y, grid_n)
        eps2 = (2 * h) ** 2
        if kernels is None:
            kernels = build_force_kernels(grid_n, h, eps2)
        # all-pairs nur bis 15k — darueber platzt die N²-Matrix (>3 GB/Array)
        ap = _zeit(lambda: allpairs_accel(x, y, m, eps2)) if n <= 15000 else None
        # Kerne haengen an (h, eps2) → hier je N neu, weil grid_fuer h aendert
        k = build_force_kernels(grid_n, h, eps2)
        pm = _zeit(lambda: pm_accelerations(x, y, m, grid_n, x0, y0, h,
                                            eps2, kernels=k))
        ap_s = f"{ap:>10.1f}  " if ap is not None else f"{'—':>12}"
        print(f"  {n:>9}  {ap_s}  {pm:>8.2f}")
    print("  (all-pairs > 15k ausgelassen: N²-Matrix sprengt den Speicher)")


def main():
    dev = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cp.cuda.Device(dev).use()
    rng = np.random.default_rng(42)
    print(f"PM-Validierung auf GPU {dev} "
          f"({cp.cuda.runtime.getDeviceProperties(dev)['name'].decode()})\n")

    ergebnisse = [
        ("Impulserhaltung",        test_impuls(rng)),
        ("Fernes Paar (1/r²)",     test_fernes_paar()),
        ("Wolke ε=4h vs all-pairs", test_wolke_vs_allpairs(
            rng, 4.0, 0.03, "3) Wolke ε≫h")),
        ("Wolke ε=1h vs all-pairs", test_wolke_vs_allpairs(
            rng, 1.0, 0.12, "4) Wolke ε≈h")),
        ("Dynamik (Leapfrog)",     test_dynamik(rng, dev)),
    ]
    print()
    alle_ok = True
    for name, ok in ergebnisse:
        print(f"  {'BESTANDEN' if ok else 'FEHLGESCHLAGEN'}  {name}")
        alle_ok = alle_ok and ok
    print("\n" + ("ALLE PM-TESTS BESTANDEN — Kraft stimmt gegen all-pairs"
                  if alle_ok else "PM-VALIDIERUNG FEHLGESCHLAGEN"))
    if alle_ok:
        bench_tempo(rng)
    return 0 if alle_ok else 1


if __name__ == "__main__":
    sys.exit(main())
