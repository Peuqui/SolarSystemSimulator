"""Etappe A des Tracer-Splits: der SERVER erzeugt die Tracer, nicht der Browser.

Frueher lud der Browser die Tracer als Koerper hoch; jetzt schickt er nur
einen Auftrag (sechs Zahlen), und der Producer wuerfelt sie selbst
(tracer_gen) und haengt sie an die Massen an. Dieser Test prueft die
Server-Haelfte OHNE Browser:

  1. FilmSession bekommt NUR Massen + einen tracer_auftrag.
  2. self.n deckt Massen + Tracer (n_mass + auftrag['n']).
  3. Der Producer laeuft, schreibt Samples, und im Slot bewegen sich BEIDE:
     die Massen (Index 0..n_mass-1) und die server-erzeugten Tracer
     (n_mass..n-1) — Letztere folgen dem Feld (nicht eingefroren).

Aufruf: ./venv/bin/python backend/test_tracer_split.py
"""
from __future__ import annotations

import sys
import time

import numpy as np

import film_producer
import server


def main():
    n_mass = 60_000
    n_tracer = 100_000
    rng = np.random.default_rng(5)
    R = 60000.0
    # Massen: kalte Scheibe (kollabiert, erzeugt ein Feld)
    r = R * np.sqrt(rng.uniform(0, 1, n_mass))
    th = rng.uniform(0, 2 * np.pi, n_mass)
    x = r * np.cos(th)
    y = r * np.sin(th)
    vx = np.zeros(n_mass)
    vy = np.zeros(n_mass)
    mass = np.full(n_mass, 1e8)
    real_r = np.full(n_mass, 1.0)
    is_ast = np.zeros(n_mass, np.uint8)      # NUR Massen, keine Upload-Tracer
    star = np.zeros(n_mass, np.uint8)

    auftrag = {"n": n_tracer, "radius": R * 1.2, "cx": 0.0, "cy": 0.0,
               "hubble": 1e-4, "streu": 0.08}

    sess = server.FilmSession(
        0.0, 1.0, x, y, vx, vy, mass, real_r,
        np.ones(n_mass, np.uint8), is_ast, star, np.zeros(n_mass, np.uint8),
        False, film_producer.SUB_SAMPLES_DEFAULT,
        softening_au=100.0, tracer_auftrag=auftrag)

    n_soll = n_mass + n_tracer
    print(f"self.n = {sess.n} (soll {n_soll})")
    if sess.n != n_soll:
        print("FEHLGESCHLAGEN — self.n deckt Massen+Tracer nicht")
        sess.stop()
        return 1

    try:
        t0 = time.time()
        while sess.head_val.value < 12 and time.time() - t0 < 150:
            sess.playhead_val.value = sess.head
            time.sleep(0.2)
        kopf = int(sess.head_val.value)
        print(f"Kopf nach {time.time()-t0:.0f}s: {kopf} Samples")
        if kopf < 5:
            print("FEHLGESCHLAGEN — kaum Samples, Producer haengt/stuerzte ab")
            return 1

        # .copy(): Views vom Ring loesen, sonst BufferError beim shm.close()
        s0 = sess.slot_pos(sess.tail_abs).copy()
        s1 = sess.slot_pos(min(kopf - 1, sess.tail_abs + 8)).copy()
        # Slot ist [x(0..n-1) | y(0..n-1)] als f32 -> Laenge 2n
        if len(s0) != 2 * n_soll:
            print(f"FEHLGESCHLAGEN — Slot {len(s0)}, erwartet {2*n_soll}")
            return 1
        endlich = bool(np.all(np.isfinite(s0)) and np.all(np.isfinite(s1)))
        # Massen: Index 0..n_mass-1 ; Tracer: n_mass..n-1 (jeweils im x-Teil)
        bew_mass = float(np.mean(np.abs(s1[:n_mass] - s0[:n_mass])))
        bew_trac = float(np.mean(
            np.abs(s1[n_mass:n_soll] - s0[n_mass:n_soll])))
        print(f"Positionen endlich: {endlich}")
        print(f"Bewegung Massen ~8 Raster: {bew_mass:.3f} AE")
        print(f"Bewegung Tracer ~8 Raster: {bew_trac:.3f} AE")

        # Handover: state_at_playhead darf NUR die Massen dumpen (n_mass),
        # sonst schlaegt die Engine-Uebergabe im Client fehl (filmRefs=Massen,
        # "Zustandsuebergabe fehlgeschlagen"). Header-n und Byte-Groesse pruefen.
        import struct
        t_mid = sess.t0 + (sess.tail_abs + 3) * sess.raster_days
        dump = sess.state_at_playhead(t_mid)
        if dump is not None:
            _st, dn, _dt = struct.unpack_from("<IId", dump, 0)
            soll = 16 + 4 * 8 * n_mass
            print(f"Handover-Dump: n={dn} (soll {n_mass}), "
                  f"bytes={len(dump)} (soll {soll})")
            handover_ok = (dn == n_mass and len(dump) == soll)
        else:
            print("Handover-Dump: None (Ring zu kurz) — uebersprungen")
            handover_ok = True

        ok = (endlich and bew_mass > 0.0 and bew_trac > 0.0 and handover_ok)
        print("\n" + ("BESTANDEN — Server erzeugt Tracer, beide bewegen sich"
                      if ok else "FEHLGESCHLAGEN"))
        return 0 if ok else 1
    finally:
        sess.stop()


if __name__ == "__main__":
    sys.exit(main())
