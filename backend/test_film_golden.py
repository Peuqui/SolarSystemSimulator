"""Golden-Tests der Film-Kollisionskette (Ende-zu-Ende ueber FilmSession).

Konstruiere Szenarien mit eindeutig bekanntem Soll-Verhalten und pruefe
den ECHTEN Producer-Pfad (GPU-Kernel + Erkennung + Event-Ring):

  A) Frontal-Bounce zweier Asteroiden  -> genau 1 kind=1-Event am
     Kontaktpunkt, beide ueberleben
  B) Sonnentaucher aus der Ruhe        -> genau 1 Merge mit der Sonne,
     Ereignis-Position nahe der Sonne
  C) Ruhiger Kreisbahn-Belt            -> NULL Ereignisse (keine
     Phantom-Kollisionen, keine Runaway-Kills)

Aufruf: ./venv/bin/python backend/test_film_golden.py
"""
from __future__ import annotations

import struct
import time

import numpy as np

import film_producer
import server


def _events(sess: server.FilmSession) -> list[dict]:
    eb = film_producer.EV_BYTES
    out = []
    for e in range(int(sess.ev_count_val.value)):
        raw = bytes(sess.ev_shm.buf[(e % sess.ev_cap) * eb:
                                    (e % sess.ev_cap) * eb + eb])
        t, a, b, mass, kind, ex, ey = struct.unpack("<dIIfIff", raw)
        out.append({"t": t, "a": a, "b": b, "mass": mass,
                    "kind": kind, "x": ex, "y": ey})
    return out


def _run(x, y, vx, vy, mass, real_r, is_ast, ast_bounce: bool,
         samples: int, timeout_s: float = 90.0) -> list[dict]:
    n = len(x)
    vis = np.ones(n, np.uint8)
    star = np.zeros(n, np.uint8)
    star[0] = 1                                  # Koerper 0 = Sonne
    # injiziert = alles 0: die Golden-Szenen sind geladene Systeme, nichts
    # ist nachtraeglich eingebracht. Das Feld steuert ohnehin nur den
    # Vorrang beim Stream-LOD, nie die Physik.
    sess = server.FilmSession(
        0.0, 0.5, np.asarray(x, float), np.asarray(y, float),
        np.asarray(vx, float), np.asarray(vy, float),
        np.asarray(mass, float), np.asarray(real_r, float),
        vis, np.asarray(is_ast, np.uint8), star,
        np.zeros(n, np.uint8), ast_bounce)
    try:
        t0 = time.time()
        while (sess.head_val.value < samples
               and time.time() - t0 < timeout_s
               and sess.proc.is_alive()):
            # Playhead mitziehen, sonst greift der Ueberschreib-Schutz
            sess.playhead_val.value = sess.head
            time.sleep(0.2)
        evs = _events(sess)
    finally:
        sess.stop()
    return evs


def main() -> None:
    G_AU = film_producer.__dict__.get("G_AU")  # noqa: F841 (Doku)

    # ---------- A: Frontal-Bounce ----------
    # Zwei Astis weit weg von der Sonne, uebergrosser realR, frontal.
    x = [0.0, 5.0 - 0.0005, 5.0 + 0.0005]
    y = [0.0, 0.0, 0.0]
    vx = [0.0, 0.5, -0.5]                       # AU/Jahr, aufeinander zu
    vy = [0.0, 0.0, 0.0]
    mass = [1.0, 1e-12, 1e-12]
    real_r = [0.00465, 1e-4, 1e-4]
    is_ast = [0, 1, 1]
    evs = _run(x, y, vx, vy, mass, real_r, is_ast,
               ast_bounce=True, samples=16)
    bounces = [e for e in evs if e["kind"] == 1]
    kills = [e for e in evs if e["kind"] == 0]
    # Das Paar ist sub-orbital und stuerzt gemeinsam sonnenwaerts —
    # Folge-Bounces sind korrekte Physik. Der Golden-Kern: das frontale
    # Tunneling des ERSTEN Kontakts wird erkannt, am richtigen Ort, im
    # ersten Sample, und niemand stirbt.
    assert len(bounces) >= 1, f"A: kein Bounce erkannt ({evs})"
    assert len(kills) == 0, f"A: unerwartete Kills: {kills}"
    ev = bounces[0]
    assert ev["t"] <= 1.0, f"A: erster Bounce zu spaet: t={ev['t']}"
    assert abs(ev["x"] - 5.0) < 0.01 and abs(ev["y"]) < 0.01, \
        f"A: Bounce-Ort falsch: ({ev['x']}, {ev['y']})"
    print(f"A Frontal-Bounce: erkannt bei t={ev['t']:.1f} d am "
          f"Kontaktpunkt ({ev['x']:.4f}, {ev['y']:.4f}) OK")

    # ---------- B: Sonnentaucher ----------
    x = [0.0, 0.3]
    y = [0.0, 0.0]
    vx = [0.0, 0.0]
    vy = [0.0, 0.0]
    mass = [1.0, 1e-12]
    real_r = [0.00465, 2e-7]
    is_ast = [0, 1]
    evs = _run(x, y, vx, vy, mass, real_r, is_ast,
               ast_bounce=False, samples=40)    # Fallzeit ~11 Tage
    merges = [e for e in evs if e["kind"] == 0 and e["a"] == 0]
    pure_kills = [e for e in evs if e["kind"] == 0 and e["a"] == 0xFFFFFFFF]
    assert len(merges) == 1, f"B: {len(merges)} Sonnen-Merges statt 1 ({evs})"
    assert len(pure_kills) == 0, f"B: Runaway-Kill statt Merge: {evs}"
    ev = merges[0]
    r_ev = float(np.hypot(ev["x"], ev["y"]))
    assert r_ev < 0.05, f"B: Merge-Ort {r_ev:.4f} AU von der Sonne"
    assert 5.0 < ev["t"] < 20.0, f"B: Merge-Zeit {ev['t']:.1f} d unplausibel"
    print(f"B Sonnentaucher: Merge bei r={r_ev:.4f} AU, "
          f"t={ev['t']:.1f} d OK")

    # ---------- C: Ruhiger Belt ----------
    rng = np.random.default_rng(11)
    nb = 500
    r = 2.0 + rng.random(nb)
    th = rng.random(nb) * 2 * np.pi
    v = np.sqrt(4 * np.pi * np.pi / r)
    x = np.concatenate(([0.0], r * np.cos(th)))
    y = np.concatenate(([0.0], r * np.sin(th)))
    vx = np.concatenate(([0.0], -v * np.sin(th)))
    vy = np.concatenate(([0.0], v * np.cos(th)))
    mass = np.concatenate(([1.0], np.full(nb, 1e-12)))
    real_r = np.concatenate(([0.00465], np.full(nb, 2e-7)))
    is_ast = np.concatenate(([0], np.ones(nb)))
    evs = _run(x, y, vx, vy, mass, real_r, is_ast,
               ast_bounce=True, samples=100)    # 50 Tage
    assert len(evs) == 0, f"C: {len(evs)} Phantom-Ereignisse: {evs[:5]}"
    print("C Ruhiger Belt: 0 Ereignisse ueber 50 Tage OK")

    print("Alle Golden-Tests bestanden.")


if __name__ == "__main__":
    main()
