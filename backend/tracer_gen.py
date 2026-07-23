"""Server-seitige Erzeugung der masselosen Tracer.

Tracer sind Deko ohne Identitaet: gleichmaessig in einer Kreisscheibe
verteilte Testteilchen, die mit einem Hubble-Fluss nach aussen starten und
danach dem Feld der Massen folgen. Frueher wuerfelte der BROWSER sie und lud
sie als Koerper hoch (52 B/Stueck, eine WS-Nachricht) — bei Millionen Tracern
die Wand. Jetzt reicht ein kompakter AUFTRAG (sechs Zahlen), aus dem der
Server sie selbst wuerfelt; der Upload und die Body-Objekte im Browser fallen
weg.

Repliziert die Verteilung aus `buildStrukturbildung` in `index.html` (die
Tracer-Schleife). NICHT bitgleich — Tracer haben keine Identitaet, nur die
VERTEILUNG muss stimmen; der Server wuerfelt mit eigenem Seed.
"""
from __future__ import annotations

import numpy as np


def wuerfle_tracer(auftrag: dict, seed: int = 0):
    """Aus dem Auftrag (Browser) die Startzustaende (x, y, vx, vy) als f64.

    auftrag-Felder:
      n       Anzahl der Tracer
      radius  Radius der Kreisscheibe (AE)
      cx, cy  Mittelpunkt der Scheibe (AE) — der Versatz gegen den
              Massenschwerpunkt, den der Browser mitrechnet
      hubble  Hubble-Rate (1/Jahr): Radialgeschwindigkeit = hubble * r
      streu   Anteil zufaelliger Querstreuung (0..1), z. B. 0,08
    """
    n = int(auftrag["n"])
    radius = float(auftrag["radius"])
    cx = float(auftrag["cx"])
    cy = float(auftrag["cy"])
    hubble = float(auftrag["hubble"])
    streu = float(auftrag.get("streu", 0.08))
    rng = np.random.default_rng(seed)

    a = rng.uniform(0.0, 2.0 * np.pi, n)
    # sqrt(U) → in der Flaeche gleichverteilt (nicht zum Zentrum gehaeuft)
    r = radius * np.sqrt(rng.uniform(0.0, 1.0, n))
    x = r * np.cos(a) + cx
    y = r * np.sin(a) + cy
    rr = np.hypot(x, y)
    rr[rr == 0.0] = 1.0
    vr = hubble * rr
    # Streuung in x und y unabhaengig, wie im Browser
    sx = (rng.uniform(-1.0, 1.0, n)) * streu * vr
    sy = (rng.uniform(-1.0, 1.0, n)) * streu * vr
    vx = x / rr * vr + sx
    vy = y / rr * vr + sy
    return (x, y, vx, vy)


if __name__ == "__main__":
    # Selbsttest: Verteilung (gleichmaessig in der Scheibe) und Hubble-Fluss
    # (mittlere Radialgeschwindigkeit steigt linear mit r).
    auf = {"n": 200_000, "radius": 60000.0, "cx": 0.0, "cy": 0.0,
           "hubble": 1e-4, "streu": 0.08}
    x, y, vx, vy = wuerfle_tracer(auf, seed=1)
    r = np.hypot(x, y)
    print(f"n={len(x)}  r_max={r.max():.0f} (soll ~{auf['radius']:.0f})")
    # Flaechen-Gleichverteilung: Anteil innerhalb r/2 soll ~1/4 sein
    innen = np.mean(r < auf["radius"] / 2)
    print(f"Anteil r<radius/2: {innen:.3f} (soll ~0,25 bei Flaechen-Gleichvert.)")
    # Radialgeschwindigkeit vs. hubble*r
    vrad = (x * vx + y * vy) / np.maximum(r, 1e-9)
    erwartet = auf["hubble"] * r
    fehler = np.median(np.abs(vrad - erwartet) / np.maximum(erwartet, 1e-9))
    print(f"Median |v_rad - hubble*r| / (hubble*r): {fehler:.3f} "
          f"(soll ~0,05, das ist die halbe Streuung)")
    imp_x = float(np.sum(vx))
    print(f"Rohsumme vx (mit Streuung, ~0 im Mittel): {imp_x:.3e}")
    print("OK" if abs(r.max() - auf["radius"]) < auf["radius"] * 0.05
          and abs(innen - 0.25) < 0.03 else "PRUEFEN")
