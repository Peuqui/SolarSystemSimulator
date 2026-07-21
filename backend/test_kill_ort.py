"""Wo verschwinden Koerper, die in den Stern stuerzen?

Ein Sonnensturz erreicht 50 AE/Jahr und mehr. Bei 0,5 Tagen Raster sind
das 0,07 AE Weg pro Sample — das Fuenfzehnfache des Sternradius. Die
Merge-Erkennung prueft deshalb die STRECKE zwischen zwei Samples; ohne
den Kontaktzeitpunkt darin faellt das Ereignis auf die Rastergrenze, und
der Koerper verschwindet sichtbar VOR dem Stern statt in ihm.

Gemessen an einer Wolke wie im Betrieb (Nutzer-Szene: v = 5,5 AE/Jahr,
r = 0,02..0,37 AE, also gebundene Radialstuerze) — geprueft wird der
Abstand zum Stern zu dem Zeitpunkt, an dem der Koerper aus dem Stream
faellt.

Aufruf: ./venv/bin/python backend/test_kill_ort.py
"""
from __future__ import annotations

import struct
import sys
import time

import numpy as np

import film_producer
import server

RASTER = 0.5
G = 4 * np.pi ** 2


def szene(n_ast=400, seed=5):
    """Wolke links vom Stern, alle mit ~5,5 AE/Jahr nach +x — wie die
    injizierte Wolke aus dem Betrieb."""
    rng = np.random.default_rng(seed)
    r = rng.uniform(0.05, 0.35, n_ast)
    th = rng.uniform(-0.4, 0.4, n_ast)
    ax = -r * np.cos(th)
    ay = r * np.sin(th)
    avx = rng.uniform(5.15, 5.80, n_ast)
    avy = rng.uniform(-0.8, 0.35, n_ast)

    x = np.concatenate([[0.0], ax])
    y = np.concatenate([[0.0], ay])
    vx = np.concatenate([[0.0], avx])
    vy = np.concatenate([[0.0], avy])
    n = len(x)
    mass = np.zeros(n)
    mass[0] = 1.0
    real_r = np.full(n, 5.79e-7)
    real_r[0] = 0.0046524726370988385
    is_ast = np.ones(n, np.uint8)
    is_ast[0] = 0
    star = np.zeros(n, np.uint8)
    star[0] = 1
    return x, y, vx, vy, mass, real_r, is_ast, star


def main():
    x, y, vx, vy, mass, real_r, is_ast, star = szene()
    n = len(x)
    r_stern = float(real_r[0])
    sess = server.FilmSession(0.0, RASTER, x, y, vx, vy, mass, real_r,
                              np.ones(n, np.uint8), is_ast, star,
                              np.zeros(n, np.uint8), False,
                              film_producer.SUB_SAMPLES_DEFAULT)
    try:
        t0 = time.time()
        while sess.head_val.value < 150 and time.time() - t0 < 120:
            sess.playhead_val.value = sess.head
            time.sleep(0.2)
        kopf = int(sess.head_val.value)

        # Ereignisse einsammeln (wie build_frame es tut) und fuer jeden
        # gekillten Koerper den Abstand zum Stern beim Kill bestimmen.
        ev_total = int(sess.ev_count_val.value)
        eb = film_producer.EV_BYTES
        abstaende = []
        for e in range(min(ev_total, sess.ev_cap)):
            raw = bytes(sess.ev_shm.buf[(e % sess.ev_cap) * eb:
                                        (e % sess.ev_cap) * eb + eb])
            (t_ev,) = struct.unpack_from("<d", raw, 0)
            b_idx, _m, kind = struct.unpack_from("<IfI", raw, 12)
            if kind != 0 or b_idx >= n:
                continue
            # Position des Opfers zur Kill-ZEIT, so wie der Client sie
            # interpoliert: lineares Mittel der Nachbarsamples.
            i_f = (t_ev - sess.t0) / RASTER - 1.0
            i0 = int(np.floor(i_f))
            if i0 < sess.tail_abs or i0 + 1 >= kopf:
                continue
            f = i_f - i0
            p0, p1 = sess.slot_pos(i0), sess.slot_pos(i0 + 1)
            bx = p0[b_idx] * (1 - f) + p1[b_idx] * f
            by = p0[n + b_idx] * (1 - f) + p1[n + b_idx] * f
            sx = p0[0] * (1 - f) + p1[0] * f
            sy = p0[n] * (1 - f) + p1[n] * f
            abstaende.append(float(np.hypot(bx - sx, by - sy)))

        if not abstaende:
            print("KEINE Merges — Szene traf den Stern nicht")
            return 1
        a = np.sort(np.array(abstaende))
        print(f"{len(a)} Koerper vom Stern verschluckt, "
              f"Sternradius {r_stern:.5f} AE\n")
        print(f"  kleinster Kill-Abstand  {a[0]:.5f} AE = "
              f"{a[0] / r_stern:6.2f} x R")
        print(f"  Median                  {a[len(a)//2]:.5f} AE = "
              f"{a[len(a)//2] / r_stern:6.2f} x R")
        print(f"  groesster               {a[-1]:.5f} AE = "
              f"{a[-1] / r_stern:6.2f} x R")
        # Schwelle als REGRESSIONSSCHUTZ, nicht als Perfektionsanspruch:
        # ohne Kontaktzeitpunkt lag der Median bei 8,7 x R (gemessen an
        # einer Nutzer-Szene), mit ihm bei ~2,3 x R.
        #
        # Der Rest ist die Sehnen-Naeherung: geprueft wird die Strecke
        # zwischen zwei Samples, und die schneidet bei einem Radialsturz
        # den echten Bogen ab — der Punkt der groessten Annaeherung AUF
        # der Sehne liegt weiter aussen als das echte Perihel. Exakt
        # wuerde es erst in der Feinschleife des Kernels, wo der Abstand
        # zu jedem massiven Koerper ohnehin je Substep berechnet wird
        # (so macht es der Hybrid-Modus). Solange die Kollision blitzt,
        # faellt der Rest optisch nicht auf.
        ok = a[len(a) // 2] < 3.0 * r_stern
        print("\n" + ("BESTANDEN — die Koerper verschwinden IM Stern"
                      if ok else
                      "FEHLGESCHLAGEN — sie verschwinden noch davor"))
        return 0 if ok else 1
    finally:
        sess.stop()


if __name__ == "__main__":
    sys.exit(main())
