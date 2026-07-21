"""Enge Begegnung mit einem BEWEGTEN Stern.

Die Feinschleife der heissen Asteroiden (nbody_kernel) rechnet mit
linear ueber dtH interpolierten Positionen der massiven Koerper. Steht
der Stern still im Ursprung, faellt ein Fehler darin nicht auf — alle
bisherigen Tests (test_subsamples, test_kernel, test_film_golden) sind
genau so gebaut. Bewegt er sich, sieht ein nah vorbeifliegender Asteroid
die Masse an der falschen Stelle und wird zur Seite abgelenkt, statt
seiner Bahn zu folgen.

Geprueft wird gegen eine feine Velocity-Verlet-Referenz mit demselben
geradlinig bewegten Stern (masselose Asteroiden -> keine Rueckwirkung,
der Stern fliegt exakt gerade).

Aufruf: ./venv/bin/python backend/test_bewegter_stern.py
"""
from __future__ import annotations

import numpy as np

from nbody_kernel import G_AU, SOFTENING, SUB_SAMPLES, NBodyCuda, pick_device

RASTER_TAGE = 0.5
TAGE_PRO_JAHR = 365.25


def referenz(px, py, pvx, pvy, sx, sy, svx, svy, m_stern,
             dt_jahre, schritte):
    """Testteilchen im Feld eines geradlinig bewegten Sterns."""
    mu = G_AU * m_stern
    h = dt_jahre / schritte

    def acc(t):
        dx = (sx + svx * t) - px
        dy = (sy + svy * t) - py
        r2 = dx * dx + dy * dy + SOFTENING
        f = mu / (r2 * np.sqrt(r2))
        return f * dx, f * dy

    t = 0.0
    ax, ay = acc(t)
    for _ in range(schritte):
        px += pvx * h + 0.5 * ax * h * h
        py += pvy * h + 0.5 * ay * h * h
        t += h
        nax, nay = acc(t)
        pvx += 0.5 * (ax + nax) * h
        pvy += 0.5 * (ay + nay) * h
        ax, ay = nax, nay
    return px, py


def lauf(v_stern: float, raster_anzahl: int = 16):
    """Stern mit Eigengeschwindigkeit v_stern (AE/Jahr) in x-Richtung,
    ein Asteroid fliegt dicht an ihm vorbei. Liefert (gpu, referenz)."""
    dev = pick_device()
    sim = NBodyCuda(dev, m_sub=SUB_SAMPLES)

    m_stern = 1.0
    # Asteroid kommt von unten links, Perihel ~0,08 AE am Stern vorbei
    ax0, ay0 = -0.6, -0.25
    avx0, avy0 = 11.0, 3.0
    x = np.array([0.0, ax0])
    y = np.array([0.0, ay0])
    vx = np.array([v_stern, avx0])
    vy = np.array([0.0, avy0])
    mass = np.array([m_stern, 0.0])
    vis = np.ones(2, np.uint8)
    is_ast = np.array([0, 1], np.uint8)

    dt = RASTER_TAGE / TAGE_PRO_JAHR
    st = sim.load_state(x, y, vx, vy, mass, vis, is_ast)
    out = sim.step_batch(st, dt, raster_anzahl)
    # Layout je Sample: [x(n) | y(n) | vx(n) | vy(n)], hier n = 2
    gpu = (float(out[-1][1]), float(out[-1][3]))       # x[1], y[1]

    rx, ry = referenz(ax0, ay0, avx0, avy0, 0.0, 0.0, v_stern, 0.0,
                      m_stern, dt * raster_anzahl, 400_000)
    return gpu, (rx, ry)


def main():
    print(f"Enge Begegnung ueber {16 * RASTER_TAGE:.0f} Sim-Tage, "
          f"Perihel ~0,08 AE\n")
    print(f"{'v Stern':>10} {'GPU (x, y)':>26} {'Referenz (x, y)':>26} "
          f"{'Abweichung':>12}")
    ok = True
    for v_stern in (0.0, 0.5, 2.0, 5.0):
        (gx, gy), (rx, ry) = lauf(v_stern)
        d = float(np.hypot(gx - rx, gy - ry))
        print(f"{v_stern:>8.1f}   {gx:>12.6f} {gy:>12.6f} "
              f"{rx:>12.6f} {ry:>12.6f} {d:>12.2e}")
        # Der ruhende Fall ist die Kontrolle: er MUSS stimmen. Weicht nur
        # der bewegte ab, sitzt der Fehler in der Positions-Interpolation
        # der massiven Koerper.
        if d > 1e-3:
            ok = False
    print("\n" + ("BESTANDEN — bewegter Stern aendert nichts"
                  if ok else
                  "FEHLGESCHLAGEN — die Bahn haengt von der Sternbewegung ab"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
