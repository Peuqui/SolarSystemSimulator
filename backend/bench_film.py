"""Durchsatz-Benchmark des Film-Producers (Sim-Tage pro Sekunde).

Misst die Kennzahl, an der die Optimierung haengt, ueber den ECHTEN Pfad
(FilmSession -> Producer-Prozess -> GPUs). Der Playhead wird mitgezogen,
damit der Ueberschreib-Schutz nie greift — gemessen wird also reine
Produktion.

Mit --det-gpus lassen sich Konfigurationen direkt gegeneinander stellen,
z. B. eine gegen zwei Erkennungskarten.

WICHTIG (aus dem Uebergabeprotokoll): Zwei Messungen sind nur
vergleichbar, wenn die raeumliche Verteilung aehnlich ist — dieselbe
Objektzahl genuegt NICHT. Deshalb wird die Szene aus festem Seed erzeugt
und jede Konfiguration startet bei t=0 mit derselben Szene.

Aufruf:
    cd backend && ../venv/bin/python bench_film.py -n 120000 --det-gpus 1
    cd backend && ../venv/bin/python bench_film.py -n 120000 --det-gpus 2
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import server


def _sonne_davor(xa, ya, vxa, vya, n_ast: int) -> dict:
    return {
        "x": np.concatenate(([0.0], xa)),
        "y": np.concatenate(([0.0], ya)),
        "vx": np.concatenate(([0.0], vxa)),
        "vy": np.concatenate(([0.0], vya)),
        "mass": np.concatenate(([1.0], np.full(n_ast, 1e-14))),
        "real_r": np.concatenate(([0.00465], np.full(n_ast, 2e-5))),
        "visible": np.ones(n_ast + 1, np.uint8),
        "is_ast": np.concatenate(([0], np.ones(n_ast, np.uint8))),
        "is_star_bh": np.concatenate(([1], np.zeros(n_ast, np.uint8))),
    }


def szene_guertel(n_ast: int, seed: int = 20260720) -> dict:
    """Sonne + ausgebildete Asteroidenscheibe auf Kreisbahnen. Der
    entspannte Zustand: wenige hunderttausend Kandidatenpaare pro Sample,
    die Erkennung ist launch- statt rechengebunden."""
    rng = np.random.default_rng(seed)
    r = 1.5 + 2.5 * np.sqrt(rng.random(n_ast))
    th = rng.random(n_ast) * 2 * np.pi
    v = np.sqrt(4 * np.pi * np.pi / r) * (1.0 + rng.normal(0, 0.03, n_ast))
    return _sonne_davor(r * np.cos(th), r * np.sin(th),
                        -v * np.sin(th), v * np.cos(th), n_ast)


def szene_knoedel(n_ast: int, klumpen: int = 5,
                  seed: int = 20260720) -> dict:
    """Sonne + mehrere eng uebereinander injizierte Wolken.

    DAS ist der Lastfall, fuer den die Aufteilung gebaut ist: die Klumpen
    durchdringen sich, pro Gitterzelle sitzen tausende Koerper und die
    Kandidatenpaare gehen in die zweistelligen Millionen — die Erkennung
    ist dann echt rechengebunden, waehrend die Physik-Karten sich
    langweilen. Bildet sich daraus mit der Zeit ein Guertel, nivelliert
    sich die Last wieder (siehe szene_guertel)."""
    rng = np.random.default_rng(seed)
    je = n_ast // klumpen
    xs, ys, vxs, vys = [], [], [], []
    for i in range(klumpen):
        m = je if i < klumpen - 1 else n_ast - je * (klumpen - 1)
        # Zentren dicht beieinander, Wolken deutlich groesser als ihr
        # Abstand -> sie durchdringen sich von Anfang an.
        r0 = 2.0 + 0.05 * i
        th0 = 0.35 * i
        cx, cy = r0 * np.cos(th0), r0 * np.sin(th0)
        v0 = np.sqrt(4 * np.pi * np.pi / r0)
        xs.append(rng.normal(cx, 0.12, m))
        ys.append(rng.normal(cy, 0.12, m))
        vxs.append(-v0 * np.sin(th0) + rng.normal(0, 0.4, m))
        vys.append(v0 * np.cos(th0) + rng.normal(0, 0.4, m))
    return _sonne_davor(np.concatenate(xs), np.concatenate(ys),
                        np.concatenate(vxs), np.concatenate(vys), n_ast)


SZENEN = {"guertel": szene_guertel, "knoedel": szene_knoedel}


def messen(n_ast: int, det_gpus: int, raster: float,
           aufwaermen_s: float, mess_s: float, szene: str) -> float:
    server.FilmSession.DET_GPUS = det_gpus
    s = SZENEN[szene](n_ast)
    sess = server.FilmSession(0.0, raster, s["x"], s["y"], s["vx"], s["vy"],
                              s["mass"], s["real_r"], s["visible"],
                              s["is_ast"], s["is_star_bh"],
                              np.zeros(n_ast + 1, np.uint8), ast_bounce=True)
    try:
        # Aufwaermen: Kernel-Kompilierung, CuPy-Pool, erste Batches.
        ende = time.monotonic() + aufwaermen_s
        while time.monotonic() < ende and sess.proc.is_alive():
            sess.playhead_val.value = sess.head
            time.sleep(0.1)
        k0, t0 = sess.head_val.value, time.monotonic()
        ende = t0 + mess_s
        while time.monotonic() < ende and sess.proc.is_alive():
            sess.playhead_val.value = sess.head
            time.sleep(0.1)
        dk = sess.head_val.value - k0
        dt = time.monotonic() - t0
    finally:
        sess.stop()
    return dk * raster / dt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", type=int, default=120_000, help="Asteroiden")
    ap.add_argument("--det-gpus", type=int, nargs="+", default=[1, 2],
                    help="zu vergleichende Erkennungskarten-Zahlen")
    ap.add_argument("--raster", type=float, default=0.5, metavar="TAGE")
    ap.add_argument("--warmup", type=float, default=15.0, metavar="SEK")
    ap.add_argument("--dauer", type=float, default=30.0, metavar="SEK")
    ap.add_argument("--szene", choices=sorted(SZENEN), default="guertel")
    args = ap.parse_args()

    print(f"Szene: {args.szene}, {args.n} Asteroiden, Raster "
          f"{args.raster} d, {args.warmup:.0f}s aufwaermen + "
          f"{args.dauer:.0f}s messen")
    basis = None
    for g in args.det_gpus:
        rate = messen(args.n, g, args.raster, args.warmup, args.dauer,
                      args.szene)
        if basis is None:
            basis = rate
        print(f"  {g} erkennungskarte(n): {rate:6.1f} sim-tage/s "
              f"({rate / basis:.2f}x)")


if __name__ == "__main__":
    main()
