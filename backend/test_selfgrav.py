"""Kernel A: selbstgravitierende Massen auf beliebig vielen Karten.

Fuenf Fragen, jede einzeln beantwortbar:

  A) Rechnet der Kernel ueberhaupt richtig? Zweikoerper-Kreisbahn ueber
     einen vollen Umlauf — sie muss geschlossen sein.
  B) Traegt f32 in der Kraftschleife? Energieerhaltung gegen eine
     f64-NumPy-Referenz, kollabierend UND virialisiert.
  C) Rechnet der Verbund kartenzahl-unabhaengig? 1 gegen N Karten muss
     BITGLEICH sein — sonst haengt das Ergebnis daran, welche Hardware
     gerade frei ist.
  D) Misst die Kalibrierung reproduzierbar? Schwankende Gewichte hiessen
     bei jedem Sessionstart andere Segmentgrenzen.
  E) Deckt `segmentiere` alle Koerper ab, luecken- und ueberlappungsfrei?
  F) Folgen die TRACER (Kernel B) dem Feld, ohne zurueckzuwirken?

Warum Energie und nicht Bahnen: N-Koerper ist chaotisch, zwei Laeufe mit
minimal verschiedener Arithmetik divergieren exponentiell. Das ist
Physik, kein Fehler. Die Gesamtenergie ist dagegen eine Erhaltungsgroesse
und misst die Qualitaet des Integrators unabhaengig davon.

Aufruf: ./venv/bin/python backend/test_selfgrav.py
"""
from __future__ import annotations

import sys

import numpy as np

import selfgrav_kernel as sg
from selfgrav_kernel import G_AU, NBodySelfGrav

BOX = 100_000.0        # AU, Ausdehnung wie im Strukturbildungs-Szenario
M_KOERPER = 1.12e7     # Sonnenmassen je Koerper
EPS = 100.0            # Plummer-Softening in AU

# Energiefehler, den wir dem Kernel zugestehen. Die f64-NumPy-Referenz
# liegt bei rund 5e-5; alles in derselben Groessenordnung ist der
# ZEITSCHRITT, nicht die Arithmetik.
E_SCHRANKE = 5e-4


def wolke(n: int, v_faktor: float, seed: int = 7):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-BOX / 2, BOX / 2, n)
    y = rng.uniform(-BOX / 2, BOX / 2, n)
    mass = np.full(n, M_KOERPER)
    v_typ = np.sqrt(G_AU * mass.sum() / (BOX / 2))
    th = rng.uniform(0, 2 * np.pi, n)
    return (x, y, v_typ * v_faktor * np.cos(th),
            v_typ * v_faktor * np.sin(th), mass)


def energie(x, y, vx, vy, mass, eps2):
    """Gesamtenergie mit Plummer-Potential. Der Selbstterm (i == j)
    liefert -G m^2/eps und wird wieder herausgerechnet."""
    kin = 0.5 * (mass * (vx * vx + vy * vy)).sum()
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    r = np.sqrt(dx * dx + dy * dy + eps2)
    pot = -0.5 * G_AU * (mass[None, :] * mass[:, None] / r).sum()
    pot += 0.5 * G_AU * (mass * mass / np.sqrt(eps2)).sum()
    return kin + pot


def referenz(x, y, vx, vy, mass, eps2, dt, schritte):
    """Leapfrog-KDK in f64 auf der CPU — dieselbe Integration wie im
    Kernel, damit nur die Arithmetik verglichen wird."""
    rx, ry = x.copy(), y.copy()
    rvx, rvy = vx.copy(), vy.copy()

    def kraft(px, py):
        dx = px[None, :] - px[:, None]
        dy = py[None, :] - py[:, None]
        inv3 = (dx * dx + dy * dy + eps2) ** -1.5
        return (G_AU * mass[None, :] * inv3 * dx).sum(1), \
               (G_AU * mass[None, :] * inv3 * dy).sum(1)

    ax, ay = kraft(rx, ry)
    for _ in range(schritte):
        rvx += ax * (0.5 * dt)
        rvy += ay * (0.5 * dt)
        rx += rvx * dt
        ry += rvy * dt
        ax, ay = kraft(rx, ry)
        rvx += ax * (0.5 * dt)
        rvy += ay * (0.5 * dt)
    return rx, ry, rvx, rvy


def laufe(sim, st, dt, schritte, block=2000):
    """In Bloecken rechnen: ein Launch ueber zehntausende Schritte
    machte den Snapshot-Puffer (steps * 4 * n floats) unnoetig gross."""
    rest = schritte
    while rest > 0:
        b = min(rest, block)
        sim.step_batch(st, dt, b)
        rest -= b


def a_kreisbahn(dev) -> bool:
    """Zwei gleiche Massen auf einer Kreisbahn, ein voller Umlauf."""
    print("\n--- A) Zweikoerper-Kreisbahn")
    m, r = 1.0, 1.0
    v = np.sqrt(G_AU * m / (4 * r))      # v^2/r = G m /(2r)^2
    periode = 2 * np.pi * r / v
    x = np.array([-r, r])
    y = np.zeros(2)
    schritte = 20000
    sim = NBodySelfGrav([dev], softening_au=1e-4)
    st = sim.load_state(x, y, np.zeros(2), np.array([-v, v]),
                        np.array([m, m]))
    sim.step_batch(st, periode / schritte, schritte)
    # (4n) = x|y|vx|vy; bei n=2 also z[0]=x0, z[2]=y0.
    z = sim.export_f64(st)
    fehler = float(np.hypot(z[0] - x[0], z[2] - y[0]))
    grenze = 1e-4 * 2 * r
    ok = fehler < grenze
    print(f"    Abweichung nach einem Umlauf: {fehler:.3e} AU "
          f"(Grenze {grenze:.1e})")
    print("    " + ("BESTANDEN" if ok else "FEHLGESCHLAGEN — die Bahn "
                    "schliesst sich nicht"))
    return ok


def b_energie(dev) -> bool:
    """Energieerhaltung gegen die f64-Referenz, beide Startregime."""
    print("\n--- B) Energieerhaltung gegen f64-NumPy")
    n, schritte, dt = 400, 8000, 0.0055
    eps2 = EPS ** 2
    alles_ok = True
    for v_faktor, etikett in ((0.4, "kollabierend"), (0.7, "virialisiert")):
        x, y, vx, vy, mass = wolke(n, v_faktor)
        e0 = energie(x, y, vx, vy, mass, eps2)
        sim = NBodySelfGrav([dev], softening_au=EPS)
        st = sim.load_state(x, y, vx, vy, mass)
        laufe(sim, st, dt, schritte)
        z = sim.export_f64(st).reshape(4, n)
        # Aus dem f64-ZUSTAND, nicht aus den f32-Snapshots: die
        # Gesamtenergie ist eine Differenz grosser, fast gleicher Zahlen.
        e_gpu = energie(z[0], z[1], z[2], z[3], mass, eps2)
        rx, ry, rvx, rvy = referenz(x, y, vx, vy, mass, eps2, dt, schritte)
        e_ref = energie(rx, ry, rvx, rvy, mass, eps2)
        d_gpu = abs((e_gpu - e0) / e0)
        d_ref = abs((e_ref - e0) / e0)
        ok = d_gpu < E_SCHRANKE
        alles_ok &= ok
        print(f"    {etikett:14s} GPU {d_gpu:.2e} | Referenz {d_ref:.2e}"
              f"  {'ok' if ok else 'ZU GROSS'}")
    print("    " + ("BESTANDEN" if alles_ok else
                    f"FEHLGESCHLAGEN — ueber {E_SCHRANKE:.0e}"))
    return alles_ok


def c_determinismus(karten) -> bool:
    """1 gegen N Karten muss bitgleich rechnen."""
    print(f"\n--- C) Kartenzahl-Unabhaengigkeit ({len(karten)} Karten)")
    if len(karten) < 2:
        print("    UEBERSPRUNGEN — nur eine Karte verfuegbar")
        return True
    n, schritte, dt = 3000, 300, 0.0055
    x, y, vx, vy, mass = wolke(n, 0.5, seed=11)
    ref, alles_ok = None, True
    for k in range(1, len(karten) + 1):
        sim = NBodySelfGrav(karten[:k], softening_au=EPS)
        st = sim.load_state(x, y, vx, vy, mass)
        sim.step_batch(st, dt, schritte)
        z = sim.export_f64(st)
        if ref is None:
            ref = z
            print(f"    {k} Karte(n) {karten[:k]}: Referenz")
            continue
        d = float(np.abs(z - ref).max())
        alles_ok &= (d == 0.0)
        print(f"    {k} Karte(n) {karten[:k]}: "
              + ("bitgleich" if d == 0.0 else f"ABWEICHUNG {d:.3e}"))
    print("    " + ("BESTANDEN" if alles_ok else
                    "FEHLGESCHLAGEN — das Ergebnis haengt an der "
                    "Kartenzahl"))
    return alles_ok


def d_kalibrierung(karten) -> bool:
    """Liefert die Kalibrierung bei Wiederholung dasselbe?

    Eine ruhende Karte misst zu langsam (Idle-Taktstufe). Schwankten die
    Gewichte, saessen die Segmentgrenzen bei jedem Start woanders."""
    print("\n--- D) Reproduzierbarkeit der Kalibrierung")
    runden = []
    for _ in range(3):
        sg._gewicht_cache.clear()
        runden.append([sg.miss_gewicht(d, True) for d in karten])
    sg._gewicht_cache.clear()
    reihen = np.array(runden)
    spanne = (reihen.max(0) - reihen.min(0)) / reihen.mean(0)
    ok = bool((spanne < 0.05).all())
    for i, d in enumerate(karten):
        print(f"    GPU {d}: " + " ".join(f"{r[i]:7.0f}" for r in runden)
              + f"   Spanne {spanne[i] * 100:.1f} %")
    print("    " + ("BESTANDEN" if ok else
                    "FEHLGESCHLAGEN — Gewichte schwanken ueber 5 %"))
    return ok


def e_segmente() -> bool:
    """Lueckenlos, ueberlappungsfrei, vollstaendig — auch bei schiefen
    Gewichten und mehr Karten als Koerpern."""
    print("\n--- E) Segmentierung")
    faelle = [(100_000, [1.0, 1.0, 1.0]), (1000, [3.0, 1.0]),
              (7, [1.0, 1.0, 1.0, 1.0, 1.0]), (1, [1.0, 2.0]),
              (12345, [4262.8, 3565.5, 3562.7, 4243.7, 3365.0])]
    ok = True
    for n, gew in faelle:
        segs = sg.segmentiere(n, gew)
        summe = sum(ln for _, ln in segs)
        lueckenlos = all(segs[i][0] + segs[i][1] == segs[i + 1][0]
                         for i in range(len(segs) - 1))
        gut = (summe == n and segs[0][0] == 0 and lueckenlos
               and len(segs) == len(gew))
        ok &= gut
        print(f"    n={n:6d}, {len(gew)} Karten: Summe {summe}"
              f"{'' if gut else '  FEHLER'}")
    print("    " + ("BESTANDEN" if ok else "FEHLGESCHLAGEN"))
    return ok


def f_tracer(dev) -> bool:
    """Kernel B: Folgen Tracer dem Feld — und wirken sie NICHT zurueck?

    Beides in einem Aufbau: eine ruhende Zentralmasse, ein Tracer auf
    exakter Kreisbahn. Nach einem vollen Umlauf muss er am Start stehen,
    und die Masse darf sich um NICHTS bewegt haben. Bewegt sie sich,
    uebt der Tracer eine Kraft aus — dann waere er keiner, und die
    ganze Rechnung waere wieder O(N^2) statt O(N x M)."""
    print("\n--- F) Tracer folgen dem Feld, ohne zurueckzuwirken")
    m, r = 1.0, 5.0
    v = np.sqrt(G_AU * m / r)          # Kreisbahn
    periode = 2 * np.pi * r / v
    schritte = 20000
    sim = NBodySelfGrav([dev], softening_au=1e-4)
    st = sim.load_state(np.array([0.0]), np.array([0.0]),
                        np.array([0.0]), np.array([0.0]), np.array([m]),
                        tracer=(np.array([r]), np.array([0.0]),
                                np.array([0.0]), np.array([v])))
    out = sim.step_batch(st, periode / schritte, schritte)
    n, mt = st["N"], st["T"]
    tx, ty = float(out[-1][4 * n]), float(out[-1][4 * n + mt])
    mx, my = float(out[-1][0]), float(out[-1][n])
    fehler = float(np.hypot(tx - r, ty))
    bahn_ok = fehler < 1e-3 * r
    ruhe_ok = abs(mx) + abs(my) == 0.0
    print(f"    Tracer nach einem Umlauf: Abweichung {fehler:.3e} AU "
          f"von {r} AU  {'ok' if bahn_ok else 'ZU GROSS'}")
    print(f"    Zentralmasse bei ({mx:.1e}, {my:.1e}) — muss exakt (0,0) "
          f"sein  {'ok' if ruhe_ok else 'HAT SICH BEWEGT'}")
    print("    " + ("BESTANDEN" if bahn_ok and ruhe_ok
                    else "FEHLGESCHLAGEN"))
    return bahn_ok and ruhe_ok


def main() -> int:
    karten = sg.waehle_karten(kraft_f32=True)
    print(f"Karten (gemessen, f32-Kraft): {karten}")
    sim = NBodySelfGrav(karten[:1], softening_au=EPS)
    print(f"Kernel auf: {sim.name()}")
    del sim

    ergebnisse = [a_kreisbahn(karten[0]), b_energie(karten[0]),
                  c_determinismus(karten), d_kalibrierung(karten),
                  e_segmente(), f_tracer(karten[0])]
    print()
    if all(ergebnisse):
        print("ALLE TESTS BESTANDEN")
        return 0
    print(f"FEHLGESCHLAGEN ({ergebnisse.count(False)} von "
          f"{len(ergebnisse)})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
