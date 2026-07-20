"""Mikrobenchmark der Bounce-Erkennung: eine gegen mehrere Karten.

Bestimmt, ab welcher Kandidatenzahl pro Sample sich die raeumliche
Aufteilung lohnt — die Schwelle, nach der film_producer.karten_wahl
umschaltet. Gemessen wird ueber bounce_suche, also denselben Codepfad wie
im Producer (samt Thread-Pool und GIL-Konkurrenz), nur ohne Physik und
Ring drumherum.

Die Kandidatenzahl wird ueber die Wolkenbreite eingestellt: enge Wolke =
viele Koerper je Gitterzelle = quadratisch mehr Paare.

Aufruf: cd backend && ../venv/bin/python bench_erkennung.py
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import cupy as cp
import numpy as np

from film_producer import Erkennungskarte, bounce_suche
from nbody_kernel import pick_detect_devices, pick_devices

DT_Y = 0.5 / 365.25


def wolke(n: int, breite: float, seed: int = 20260720):
    """Kugelige Wolke mit Geschwindigkeitsstreuung. `breite` steuert die
    Dichte und damit die Kandidatenzahl."""
    rng = np.random.default_rng(seed)
    return [a.astype(np.float32) for a in (
        rng.normal(2.0, breite, n), rng.normal(0.0, breite, n),
        rng.normal(0.0, 1.0, n), rng.normal(6.0, 1.0, n))]


def messen(det, pool, felder_je_karte, rr_max: float,
           runden: int) -> tuple[float, int]:
    rows = [tuple(f) for f in felder_je_karte]
    # Erste Runde waermt Kernel und Speicherpool auf.
    bounce_suche(det, pool, rows, DT_Y, rr_max)
    cp.cuda.Device(det[0].dev).synchronize()
    t0 = time.monotonic()
    kand = 0
    for _ in range(runden):
        _hits, kand, _h = bounce_suche(det, pool, rows, DT_Y, rr_max)
    cp.cuda.Device(det[0].dev).synchronize()
    return (time.monotonic() - t0) / runden, kand


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", type=int, default=200_000, help="Asteroiden")
    ap.add_argument("--runden", type=int, default=8)
    ap.add_argument("--breiten", type=float, nargs="+",
                    default=[1.0, 0.5, 0.25, 0.12, 0.06, 0.03, 0.015])
    args = ap.parse_args()

    # Dieselben Karten, die auch der Producer bekaeme: alles ausser den
    # Physik-GPUs. Sonst misst der Benchmark eine Hardware-Kombination,
    # die im Betrieb nie vorkommt.
    devs = pick_detect_devices(pick_devices(), 0, 2)
    if len(devs) < 2:
        raise SystemExit("braucht mindestens zwei freie GPUs neben der "
                         "Physik")
    print(f"Erkennungskarten: {devs}, {args.n} Asteroiden, "
          f"{args.runden} Runden je Punkt")

    is_ast = np.ones(args.n, np.uint8) != 0
    vis = np.ones(args.n, np.uint8)
    rr = np.full(args.n, 2e-5, np.float32)
    rr_max = float(rr.max())
    det = [Erkennungskarte(d, is_ast, -np.inf, np.inf) for d in devs]
    for karte in det:
        karte.stammdaten(vis, rr)
    pool = ThreadPoolExecutor(max_workers=len(det) - 1)

    print(f"{'kandidaten/sample':>20} {'1 karte':>10} {'2 karten':>10} "
          f"{'gewinn':>8}")
    try:
        for breite in args.breiten:
            felder = wolke(args.n, breite)
            # Jede Karte braucht die Daten in IHREM Speicher.
            je_karte = []
            for karte in det:
                with cp.cuda.Device(karte.dev):
                    je_karte.append([cp.asarray(a) for a in felder])
            t1, kand = messen(det, pool, je_karte[:1], rr_max, args.runden)
            t2, _ = messen(det, pool, je_karte, rr_max, args.runden)
            print(f"{kand:>20,} {t1 * 1e3:>9.1f}ms {t2 * 1e3:>9.1f}ms "
                  f"{t1 / t2:>7.2f}x")
    finally:
        pool.shutdown()


if __name__ == "__main__":
    main()
