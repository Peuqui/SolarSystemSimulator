"""Aequivalenz-Test der raeumlich aufgeteilten Bounce-Erkennung.

Die Bounce-Suche laeuft auf mehreren GPUs, jede fuer einen x-Streifen der
Szene (siehe Erkennungskarte). Der Test prueft die Invariante, an der ein
frueherer Anlauf gescheitert ist:

    Die Vereinigung der Streifen-Treffer ist EXAKT die Trefferliste einer
    einzelnen Karte — kein Paar geht im Halo verloren, keines wird von
    zwei Karten doppelt gemeldet.

Geprueft ueber Szenen mit steigender Bosheit: gleichverteilte Wolke,
dichter Klumpen (viele Paare in wenigen Zellen), zwei getrennte Wolken
(die Streifengrenze faellt in die Luecke) und ein Paar direkt AUF der
Grenze (der Fall, der frueher Golden-Test A fallen liess).

Aufruf: cd backend && ../venv/bin/python test_erkennung_streifen.py
"""
from __future__ import annotations

import cupy as cp
import numpy as np

from film_producer import Erkennungskarte, _streifengrenzen
from nbody_kernel import pick_detect_devices

DT_Y = 0.5 / 365.25       # ein Raster von 0,5 Tagen


def _hits(karte: Erkennungskarte, felder, h: float, halo: float):
    treffer, _kand = karte.bounce_hits(*felder, DT_Y, h, halo)
    if treffer is None:
        return set()
    return set(zip(treffer[0].tolist(), treffer[1].tolist()))


def _pruefe(name: str, x, y, vx, vy, real_r, dev: int) -> None:
    n = len(x)
    is_ast = np.ones(n, np.uint8) != 0
    vis = np.ones(n, np.uint8)
    rr32 = real_r.astype(np.float32)

    with cp.cuda.Device(dev):
        felder = tuple(cp.asarray(a.astype(np.float32))
                       for a in (x, y, vx, vy))
        v95 = float(cp.percentile(cp.hypot(felder[2], felder[3]), 95))
    h = max(1e-4, 2.0 * v95 * DT_Y)
    halo = 2.0 * h + 2.0 * float(real_r.max())

    karte = Erkennungskarte(dev, is_ast, -np.inf, np.inf)
    karte.stammdaten(vis, rr32)
    soll = _hits(karte, felder, h, halo)

    # Dieselbe Karte, aber Streifen fuer Streifen abgearbeitet: isoliert
    # die Aufteilungs-Logik von jeder Nebenwirkung mehrerer GPUs.
    for anzahl in (2, 3, 5):
        with cp.cuda.Device(dev):
            grenzen = _streifengrenzen(felder[0], anzahl)
        ist: set = set()
        doppelt = []
        for lo, hi in grenzen:
            karte.streifen(lo, hi)
            teil = _hits(karte, felder, h, halo)
            doppelt += list(ist & teil)
            ist |= teil
        assert not doppelt, \
            f"{name}/{anzahl}: {len(doppelt)} Paare doppelt: {doppelt[:5]}"
        fehlt = soll - ist
        zuviel = ist - soll
        assert not fehlt, f"{name}/{anzahl}: {len(fehlt)} Paare verloren: " \
                          f"{sorted(fehlt)[:5]} (grenzen={grenzen})"
        assert not zuviel, f"{name}/{anzahl}: {len(zuviel)} Paare zuviel: " \
                           f"{sorted(zuviel)[:5]}"
    print(f"{name}: {len(soll)} Treffer, 2/3/5 Streifen deckungsgleich OK")


def main() -> None:
    dev_list = pick_detect_devices([], 0, 1)
    dev = dev_list[0] if dev_list else 0
    rng = np.random.default_rng(4711)

    # 1) Gleichverteilte Wolke mit grosszuegigen Radien -> viele Treffer.
    nb = 20_000
    _pruefe("wolke",
            rng.uniform(-3, 3, nb), rng.uniform(-3, 3, nb),
            rng.normal(0, 4, nb), rng.normal(0, 4, nb),
            np.full(nb, 3e-3), dev)

    # 2) Dichter Klumpen: sehr viele Paare pro Zelle.
    nb = 8_000
    _pruefe("klumpen",
            rng.normal(2.0, 0.02, nb), rng.normal(0.0, 0.02, nb),
            rng.normal(0, 6, nb), rng.normal(0, 6, nb),
            np.full(nb, 2e-4), dev)

    # 3) Zwei getrennte Wolken — der Median faellt in die Luecke, jeder
    #    Streifen sieht im Halo nur leeren Raum.
    nb = 5_000
    x = np.concatenate([rng.normal(-4.0, 0.1, nb),
                        rng.normal(4.0, 0.1, nb)])
    y = rng.normal(0.0, 0.1, 2 * nb)
    _pruefe("zwei-wolken", x, y,
            rng.normal(0, 3, 2 * nb), rng.normal(0, 3, 2 * nb),
            np.full(2 * nb, 1e-3), dev)

    # 4) Paar exakt AUF der Streifengrenze: bei zwei Koerpern liegt der
    #    Median zwischen ihnen. Genau hier verlor der frueher
    #    zurueckgerollte Anlauf den Treffer ("A: kein Bounce erkannt").
    _pruefe("auf-der-grenze",
            np.array([5.0 - 5e-4, 5.0 + 5e-4]), np.zeros(2),
            np.array([0.5, -0.5]), np.zeros(2),
            np.full(2, 1e-4), dev)

    print("Alle Streifen-Aequivalenztests bestanden.")


if __name__ == "__main__":
    main()
