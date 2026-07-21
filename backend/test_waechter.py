"""Der Numerik-Waechter meldet sich als eigenes Ereignis — nicht als Kollision.

Asteroiden, die schneller als das Dreifache der Fluchtgeschwindigkeit
sind, nimmt der Producer aus der Rechnung. Frueher lief dieses Aufraeumen
als `kind 0` (merge/kill) mit `a = 0xFFFFFFFF` durch. Der Client sah eine
Kollision ohne Partner, zeichnete eine Explosion an einen Ort, an dem
nichts passiert war, und zaehlte sie mit — im Galaxienhaufen blitzte es
dadurch mitten im Leeren.

Geprueft wird an einer Szene, die NUR Runaways erzeugt (alles fliegt vom
Stern weg, niemand trifft ihn):

  a) es entstehen ueberhaupt Waechter-Ereignisse,
  b) sie tragen `kind 2` und `a = 0xFFFFFFFF`,
  c) kein einziges laeuft als `kind 0` durch,
  d) der Kollisionszaehler des Producers bleibt bei 0 — Aufraeumen ist
     keine Kollision.

Aufruf: ./venv/bin/python backend/test_waechter.py
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
KEIN_PARTNER = 0xFFFFFFFF


def szene(n_ast=200, seed=7):
    """Stern in der Mitte, Asteroiden radial nach aussen mit weit mehr
    als der Fluchtgeschwindigkeit.

    v_esc bei r = 1 AE und 1 M☉ sind 8,9 AE/Jahr; der Waechter greift ab
    dem Dreifachen (26,6). Mit 60 AE/Jahr liegt die Szene sicher darueber,
    ohne dass ein Koerper je in die Naehe des Sterns kommt."""
    rng = np.random.default_rng(seed)
    r = rng.uniform(1.0, 3.0, n_ast)
    th = rng.uniform(0.0, 2 * np.pi, n_ast)
    ax, ay = r * np.cos(th), r * np.sin(th)
    tempo = rng.uniform(55.0, 65.0, n_ast)
    avx, avy = tempo * np.cos(th), tempo * np.sin(th)   # strikt radial weg

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
    sess = server.FilmSession(0.0, RASTER, x, y, vx, vy, mass, real_r,
                              np.ones(n, np.uint8), is_ast, star,
                              np.zeros(n, np.uint8), False,
                              film_producer.SUB_SAMPLES_DEFAULT)
    try:
        t0 = time.time()
        while sess.head_val.value < 60 and time.time() - t0 < 120:
            sess.playhead_val.value = sess.head
            time.sleep(0.2)

        eb = film_producer.EV_BYTES
        ev_total = int(sess.ev_count_val.value)
        nach_kind: dict[int, int] = {}
        fremde_partner = 0
        for e in range(min(ev_total, sess.ev_cap)):
            raw = bytes(sess.ev_shm.buf[(e % sess.ev_cap) * eb:
                                        (e % sess.ev_cap) * eb + eb])
            (a_idx,) = struct.unpack_from("<I", raw, 8)
            (kind,) = struct.unpack_from("<I", raw, 20)
            nach_kind[kind] = nach_kind.get(kind, 0) + 1
            if kind == 2 and a_idx != KEIN_PARTNER:
                fremde_partner += 1

        waechter = nach_kind.get(2, 0)
        merges = nach_kind.get(0, 0)
        koll = int(sess.coll_val.value)
        print(f"Ereignisse gesamt: {ev_total}")
        print(f"  kind 2 (Waechter): {waechter}")
        print(f"  kind 0 (Merge)   : {merges}")
        print(f"  kind 1 (Bounce)  : {nach_kind.get(1, 0)}")
        print(f"  Kollisionszaehler des Producers: {koll}")

        fehler = []
        if waechter == 0:
            fehler.append("keine Waechter-Ereignisse — Szene lief nicht "
                          "ueber die Schwelle")
        if fremde_partner:
            fehler.append(f"{fremde_partner} Waechter-Ereignisse mit "
                          f"echtem Partner statt 0xFFFFFFFF")
        if merges:
            fehler.append(f"{merges} Ereignisse als kind 0 — hier stuerzt "
                          f"niemand in den Stern, das koennen nur "
                          f"Waechter-Kills im alten Gewand sein")
        if koll:
            fehler.append(f"Kollisionszaehler steht auf {koll}, obwohl "
                          f"nur aufgeraeumt wurde")

        if fehler:
            print("\nFEHLGESCHLAGEN:")
            for f in fehler:
                print("  - " + f)
            return 1
        print("\nBESTANDEN — der Waechter meldet sich als kind 2 ohne "
              "Partner und zaehlt nicht als Kollision")
        return 0
    finally:
        sess.stop()


if __name__ == "__main__":
    sys.exit(main())
