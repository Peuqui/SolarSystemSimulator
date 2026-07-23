"""Test der Punktauswahl fuers Film-Streaming (_lod_auswahl).

Die Auswahl ist eine gleichverteilte Hash-Stichprobe ueber den festen
Original-Index — sie haengt NICHT an Position oder lokaler Dichte. Genau
das war der Flacker-Fix: die frueher dichteabhaengige Rate driftete mit
dem bewegten Teilchenfeld, sodass Koerper an der Schwelle zwischen zwei
Samples kippten. Jetzt ist die Auswahl eine reine Funktion aus (Index,
Rangfolge, Budget) und damit ueber alle Samples EXAKT dieselbe.

Geprueft:
  A) Das Budget wird eingehalten (Zielwert, ~sqrt-Streuung).
  B) Rangfolge: massive Koerper NIE ausgeduennt; die Asteroiden des
     geladenen Systems (Guertel) gehen den injizierten Wolken vor und
     bleiben vollstaendig, solange das Budget reicht.
  C) Positions-Unabhaengigkeit: dieselbe Auswahl, egal wo die Koerper
     stehen — der eigentliche Kern des Flacker-Fixes.
  D) Gleichverteilung innerhalb einer Stufe: die behaltene Rate der Wolke
     trifft budget/N im Erwartungswert.

Aufruf: cd backend && ../venv/bin/python test_lod_auswahl.py
"""
from __future__ import annotations

import numpy as np

import server


class _Attrappe:
    """Nur die Felder, die _lod_auswahl/_dichte_filter anfassen — eine
    echte FilmSession wuerde GPU-Prozess und Shared Memory anlegen."""

    LOD_MASSEN_WOLKE_AB = server.FilmSession.LOD_MASSEN_WOLKE_AB
    LOD_MASSEN_ANTEIL = server.FilmSession.LOD_MASSEN_ANTEIL
    _lod_auswahl = server.FilmSession._lod_auswahl
    _dichte_filter = server.FilmSession._dichte_filter

    def __init__(self, is_ast, injiziert):
        self._is_ast = is_ast
        self._injiziert = injiziert


def _szene(n_massiv: int, n_guertel: int, n_wolke: int):
    """Sonne+Planeten, duenner Ring, kompakte Wolke. Nur die Typ-Masken
    zaehlen fuer die Auswahl — Positionen werden nicht mehr gebraucht."""
    is_ast = np.concatenate([np.zeros(n_massiv, bool),
                             np.ones(n_guertel + n_wolke, bool)])
    injiziert = np.concatenate([np.zeros(n_massiv + n_guertel, bool),
                                np.ones(n_wolke, bool)])
    return is_ast, injiziert


def main() -> None:
    n_massiv, n_guertel, n_wolke = 10, 1_400, 300_000
    is_ast, injiziert = _szene(n_massiv, n_guertel, n_wolke)
    sess = _Attrappe(is_ast, injiziert)
    alle = np.arange(n_massiv + n_guertel + n_wolke)
    guertel = slice(n_massiv, n_massiv + n_guertel)
    wolke = slice(n_massiv + n_guertel, None)

    for budget in (120_000, 60_000, 20_000):
        sel = sess._lod_auswahl(alle, budget)
        drin = np.zeros(len(alle), bool)
        drin[sel] = True
        ma = int(drin[:n_massiv].sum())
        gu = int(drin[guertel].sum())
        wo = int(drin[wolke].sum())

        # A: Budget ist ein Zielwert — die Hash-Auswahl trifft budget/N im
        # Erwartungswert und streut um ~sqrt(budget). 2% liegt weit ueber
        # der Streuung und weit unter jeder Wirkung auf Bandbreite/Bildrate.
        assert len(sel) <= budget * 1.02, f"A: {len(sel)} > Budget {budget}"
        # B: Rangfolge. Massive vollstaendig; der Guertel passt (Budget -
        # Massive >= Guertel) und bleibt darum ganz, die Wolke bekommt den
        # Rest und wird geduennt.
        assert ma == n_massiv, f"Rangfolge: nur {ma}/{n_massiv} massive"
        assert gu == n_guertel, f"B: Guertel geduennt ({gu}/{n_guertel})"
        assert 0 < wo < n_wolke, f"B: Wolke {wo} (erwartet 0<wo<{n_wolke})"
        print(f"Budget {budget:>7,}: {len(sel):>7,} Punkte | "
              f"massiv {ma}/{n_massiv} | guertel {gu:>5,}/{n_guertel:,} "
              f"(100%) | wolke {wo:>6,}/{n_wolke:,} ({100*wo/n_wolke:4.1f}%)")

        # D: Gleichverteilung — behaltene Wolken-Rate ~ Rest-Budget/N_wolke.
        rest = budget - n_massiv - n_guertel
        assert abs(wo - rest) <= 5 * np.sqrt(rest), \
            f"D: Wolke {wo} weit weg vom Ziel {rest}"

    # B unter knappem Budget: reicht es nicht mal fuer den Guertel, wird
    # auch der geduennt (Galaxien-Szenario) — massive bleiben vollstaendig.
    sel = sess._lod_auswahl(alle, 500)
    drin = np.zeros(len(alle), bool)
    drin[sel] = True
    assert int(drin[:n_massiv].sum()) == n_massiv
    assert int(drin[guertel].sum()) < n_guertel, \
        "bei 500 Punkten muss auch der Guertel geduennt werden"
    print(f"Budget     500: {len(sel):>7,} Punkte | massive vollstaendig, "
          f"Guertel geduennt auf {int(drin[guertel].sum())}")

    # C: Positions-Unabhaengigkeit / Determinismus — der Kern des Fixes.
    # _lod_auswahl bekommt gar keine Position mehr; die Auswahl kann also
    # zwischen zwei Samples (bewegtes Teilchenfeld) nicht mehr driften.
    # Beleg: identisches Ergebnis bei Wiederholung.
    a = sess._lod_auswahl(alle, 60_000)
    b = sess._lod_auswahl(alle, 60_000)
    assert np.array_equal(a, b), "C: Auswahl nicht deterministisch"
    print("Auswahl deterministisch & positions-unabhaengig OK")

    print("Alle LOD-Auswahl-Tests bestanden.")


if __name__ == "__main__":
    main()
