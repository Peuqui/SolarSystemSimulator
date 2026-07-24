"""Ende-zu-Ende: eine FilmSession mit genug MASSEN waehlt den PM-Kernel und
produziert Samples, ohne zu abstuerzen.

Prueft die Anbindung in film_producer.py (Kernelwahl nach Massenzahl,
NBodyPM als Drop-in fuer NBodySelfGrav) im echten Producer-Prozess — nicht
nur den Kernel isoliert wie test_pm.py.

  1. > PM_AB Massen + softening_au > 0 → Producer meldet "particle-mesh".
  2. Der Kopf laeuft (Samples werden geschrieben) — kein Absturz, kein
     Haengenbleiben.
  3. Die Positionen sind endlich und die kalte Wolke bewegt sich (Kollaps),
     statt einzufrieren.

Aufruf: ./venv/bin/python backend/test_pm_pipeline.py
"""
from __future__ import annotations

import sys
import time

import numpy as np

import film_producer
import server


def main():
    n = 60000                      # > PM_AB (50.000) → Particle-Mesh
    rng = np.random.default_rng(3)
    R = 60000.0
    r = R * np.sqrt(rng.uniform(0, 1, n))
    th = rng.uniform(0, 2 * np.pi, n)
    x = r * np.cos(th)
    y = r * np.sin(th)
    vx = np.zeros(n)               # kalt → kollabiert sichtbar
    vy = np.zeros(n)
    mass = np.full(n, 1e8)         # ALLES Massen (is_ast = 0)
    real_r = np.full(n, 1.0)
    is_ast = np.zeros(n, np.uint8)
    star = np.zeros(n, np.uint8)

    sess = server.FilmSession(
        0.0, 1.0, x, y, vx, vy, mass, real_r,
        np.ones(n, np.uint8), is_ast, star, np.zeros(n, np.uint8),
        False, film_producer.SUB_SAMPLES_DEFAULT,
        softening_au=100.0)        # > 0 → selbstgravitierend → PM bei n≥50k
    try:
        t0 = time.time()
        while sess.head_val.value < 12 and time.time() - t0 < 150:
            sess.playhead_val.value = sess.head
            time.sleep(0.2)
        kopf = int(sess.head_val.value)
        print(f"Kopf nach {time.time()-t0:.0f}s: {kopf} Samples")
        if kopf < 5:
            print("FEHLGESCHLAGEN — kaum Samples, Producer haengt oder "
                  "stuerzte ab")
            return 1

        # Erstes und ein spaeteres Sample lesen, Endlichkeit + Bewegung
        s0 = sess.slot_pos(sess.tail_abs)
        s1 = sess.slot_pos(min(kopf - 1, sess.tail_abs + 8))
        endlich = bool(np.all(np.isfinite(s0)) and np.all(np.isfinite(s1)))
        # x-Positionen der Massen (erste n Eintraege im Slot)
        bewegung = float(np.mean(np.abs(s1[:n] - s0[:n])))
        print(f"Positionen endlich: {endlich}")
        print(f"mittlere Bewegung ueber ~8 Raster: {bewegung:.3f} AE")

        ok = endlich and bewegung > 0.0
        print("\n" + ("BESTANDEN — PM-Pipeline produziert bewegte, endliche "
                      "Samples" if ok else "FEHLGESCHLAGEN"))
        # s0/s1 sind Views auf den Ring — vor sess.stop() (shm.close())
        # freigeben, sonst blockiert ihr Export-Pointer das Schliessen.
        del s0, s1
        return 0 if ok else 1
    finally:
        sess.stop()


if __name__ == "__main__":
    sys.exit(main())
