"""Zwischenbilder (Sub-Samples) der heissen Asteroiden pruefen.

Der Kernel legt fuer eng begegnende Asteroiden M_sub Stuetzpunkte je
Raster ab. Dieser Test stellt sicher, dass sie
  a) ueberhaupt entstehen (Sonnentaucher ist heiss, ferner Asti nicht),
  b) zeitlich auf dem Raster j*dtRaster/M_sub liegen und
  c) auf der ECHTEN Bahn liegen — geprueft gegen eine feine
     NumPy-Referenzintegration derselben Physik.

Aufruf: ./venv/bin/python backend/test_subsamples.py
"""
from __future__ import annotations

import numpy as np

from nbody_kernel import (G_AU, SOFTENING, SUB_SAMPLES, NBodyCuda,
                          pick_device)


def referenz_bahn(x, y, vx, vy, mass, dt_jahre, schritte):
    """Feine Velocity-Verlet-Referenz (Sonne + Testteilchen), gibt die
    Position des Testteilchens an jedem Schritt zurueck."""
    px, py, pvx, pvy = float(x), float(y), float(vx), float(vy)
    mu = G_AU * mass
    h = dt_jahre / schritte

    def acc(ax_, ay_):
        r2 = ax_ * ax_ + ay_ * ay_ + SOFTENING
        f = -mu / (r2 * np.sqrt(r2))
        return f * ax_, f * ay_

    accx, accy = acc(px, py)
    bahn = []
    for _ in range(schritte):
        px += pvx * h + 0.5 * accx * h * h
        py += pvy * h + 0.5 * accy * h * h
        nax, nay = acc(px, py)
        pvx += 0.5 * (accx + nax) * h
        pvy += 0.5 * (accy + nay) * h
        accx, accy = nax, nay
        bahn.append((px, py))
    return bahn


def main():
    dev = pick_device()
    ms = SUB_SAMPLES
    sim = NBodyCuda(dev, m_sub=ms)
    print(f"Device: {dev}, M_sub = {ms}")

    # Sonne + zwei Asteroiden: einer taucht dicht an der Sonne vorbei
    # (heiss), einer zieht weit draussen ruhig seine Bahn (nicht heiss).
    m_sonne = 1.0
    taucher = (0.15, 0.004, -160.0, 0.0)      # schnell fast radial rein
    fern = (3.0, 0.0, 0.0, np.sqrt(G_AU / 3.0))
    x = np.array([0.0, taucher[0], fern[0]])
    y = np.array([0.0, taucher[1], fern[1]])
    vx = np.array([0.0, taucher[2], fern[2]])
    vy = np.array([0.0, taucher[3], fern[3]])
    mass = np.array([m_sonne, 0.0, 0.0])
    vis = np.ones(3, np.uint8)
    is_ast = np.array([0, 1, 1], np.uint8)

    dt_raster = 0.5 / 365.25                   # 0,5 Tage
    st = sim.load_state(x, y, vx, vy, mass, vis, is_ast)
    sim.step_batch(st, dt_raster, 1)
    sub = st["sub"]
    assert sub is not None, "keine Sub-Samples geliefert"
    idx, pos = sub[0]
    print(f"Koerper mit lueckenloser Sub-Bahn: {idx.tolist()} "
          f"(erwartet: nur der Taucher = Index 1)")
    assert 1 in idx.tolist(), "Taucher hat KEINE Sub-Bahn"
    assert 2 not in idx.tolist(), "ferner Asti sollte nicht heiss sein"
    assert pos.shape[0] == ms and pos.shape[1] == 2, f"Form {pos.shape}"

    # c) Stuetzpunkte gegen die feine Referenz. Die Referenz laeuft mit
    # 200k Schritten; an j*dtRaster/M_sub wird verglichen.
    spalte = idx.tolist().index(1)
    schritte = 200_000
    bahn = referenz_bahn(taucher[0], taucher[1], taucher[2], taucher[3],
                         m_sonne, dt_raster, schritte)
    print("\n j |  Sub-Sample (x, y)        |  Referenz (x, y)          |"
          " Abweichung")
    schlimmste = 0.0
    for j in range(ms):
        sx, sy = float(pos[j, 0, spalte]), float(pos[j, 1, spalte])
        rx, ry = bahn[int(round((j + 1) / ms * schritte)) - 1]
        d = float(np.hypot(sx - rx, sy - ry))
        schlimmste = max(schlimmste, d)
        print(f"{j + 1:2d} | {sx:11.6f} {sy:11.6f} | {rx:11.6f} {ry:11.6f}"
              f" | {d:.2e}")
    # f32-Ausgabe: ~1e-6 relativ. Grosszuegig, aber weit unter allem,
    # was im Bild sichtbar waere (1e-4 AU bei nahem Zoom).
    print(f"\nGroesste Abweichung: {schlimmste:.2e} AU")
    assert schlimmste < 1e-4, f"Sub-Bahn weicht ab: {schlimmste:.2e} AU"
    print("Sub-Samples liegen auf der echten Bahn — Test bestanden.")


if __name__ == "__main__":
    main()
