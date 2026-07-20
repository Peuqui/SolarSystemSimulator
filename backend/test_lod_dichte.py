"""Test der dichteabhaengigen Punktauswahl fuers Film-Streaming.

Geprueft werden die drei Eigenschaften, auf die es dem Nutzer ankommt:

  A) Das Budget wird eingehalten.
  B) Duenne Strukturen ueberleben — ein Guertel mit wenigen hundert
     Objekten verschwindet nicht neben einer Wolke mit hunderttausenden.
     (Mit der frueheren gleichmaessigen Rate war genau das der Fehler.)
  C) Dichte Gebiete bleiben SICHTBAR dichter. Volle Nivellierung waere
     genauso falsch wie das Verschwindenlassen: man muss sehen, wo
     besonders viel ist.

Ausserdem die Rangfolge: massive Koerper werden nie ausgeduennt, die
Asteroiden des geladenen Systems gehen den injizierten Wolken vor.

Aufruf: cd backend && ../venv/bin/python test_lod_dichte.py
"""
from __future__ import annotations

import numpy as np

import server


class _Attrappe:
    """Nur die Felder, die _lod_auswahl/_dichte_filter anfassen — eine
    echte FilmSession wuerde GPU-Prozess und Shared Memory anlegen."""

    LOD_ZELLEN = server.FilmSession.LOD_ZELLEN
    LOD_GAMMA = server.FilmSession.LOD_GAMMA
    _lod_auswahl = server.FilmSession._lod_auswahl
    _dichte_filter = server.FilmSession._dichte_filter

    def __init__(self, is_ast, injiziert):
        self._is_ast = is_ast
        self._injiziert = injiziert


def _szene(n_massiv: int, n_guertel: int, n_wolke: int, seed: int = 7):
    """Sonne+Planeten, duenner Ring, kompakte Wolke — die Konstellation,
    in der die alte gleichmaessige Ausduennung den Ring ausloeschte."""
    rng = np.random.default_rng(seed)
    th = rng.random(n_guertel) * 2 * np.pi
    r = 2.0 + 0.3 * rng.random(n_guertel)
    x = np.concatenate([
        rng.uniform(-5, 5, n_massiv),
        r * np.cos(th),
        rng.normal(-3.0, 0.05, n_wolke)])
    y = np.concatenate([
        rng.uniform(-5, 5, n_massiv),
        r * np.sin(th),
        rng.normal(2.0, 0.05, n_wolke)])
    is_ast = np.concatenate([np.zeros(n_massiv, bool),
                             np.ones(n_guertel + n_wolke, bool)])
    injiziert = np.concatenate([np.zeros(n_massiv + n_guertel, bool),
                                np.ones(n_wolke, bool)])
    return x.astype(np.float32), y.astype(np.float32), is_ast, injiziert


def main() -> None:
    n_massiv, n_guertel, n_wolke = 10, 1_400, 300_000
    x, y, is_ast, injiziert = _szene(n_massiv, n_guertel, n_wolke)
    sess = _Attrappe(is_ast, injiziert)
    box = (-6.0, -6.0, 12.0, 12.0)
    alle = np.arange(len(x))
    guertel = slice(n_massiv, n_massiv + n_guertel)
    wolke = slice(n_massiv + n_guertel, None)

    for budget in (120_000, 60_000, 20_000):
        sel = sess._lod_auswahl(alle, x, y, box, budget)
        drin = np.zeros(len(x), bool)
        drin[sel] = True
        ma = int(drin[:n_massiv].sum())
        gu = int(drin[guertel].sum())
        wo = int(drin[wolke].sum())

        # Budget ist ein Zielwert: die Hash-Auswahl trifft die Zellvorgabe
        # im Erwartungswert und streut um ~sqrt(budget). 2% Toleranz liegt
        # weit ueber dieser Streuung und weit unter jeder praktischen
        # Wirkung auf Bandbreite oder Bildrate.
        assert len(sel) <= budget * 1.02, f"A: {len(sel)} > Budget {budget}"
        assert ma == n_massiv, f"Rangfolge: nur {ma}/{n_massiv} massive"
        assert gu > 0, "B: Guertel vollstaendig verschwunden"
        # C: Die Wolke ist 214x zahlreicher als der Guertel. Sie MUSS
        # sichtbar mehr Punkte bekommen (sonst nivelliert), aber deutlich
        # weniger als das volle Verhaeltnis (sonst stirbt der Guertel).
        roh = n_wolke / n_guertel
        gezeigt = wo / max(1, gu)
        assert gezeigt > 1.5, \
            f"C: Wolke nur {gezeigt:.1f}x dichter — zu stark nivelliert"
        assert gezeigt < roh, \
            f"C: Verhaeltnis {gezeigt:.1f} nicht komprimiert (roh {roh:.0f})"
        print(f"Budget {budget:>7,}: {len(sel):>7,} Punkte | "
              f"massiv {ma}/{n_massiv} | guertel {gu:>5,}/{n_guertel:,} "
              f"({100*gu/n_guertel:5.1f}%) | wolke {wo:>6,}/{n_wolke:,} "
              f"({100*wo/n_wolke:4.1f}%) | dichte-verhaeltnis "
              f"{gezeigt:.1f}x statt {roh:.0f}x")

    # Rangfolge unter knappem Budget: reicht es nicht mal fuer die
    # Originale, werden auch die geduennt (Galaxien-Szenario) — aber die
    # massiven Koerper bleiben immer vollstaendig.
    sel = sess._lod_auswahl(alle, x, y, box, 500)
    drin = np.zeros(len(x), bool)
    drin[sel] = True
    assert int(drin[:n_massiv].sum()) == n_massiv
    assert int(drin[guertel].sum()) < n_guertel, \
        "bei 500 Punkten muss auch der Guertel geduennt werden"
    print(f"Budget     500: {len(sel):>7,} Punkte | massive vollstaendig, "
          f"Guertel geduennt auf {int(drin[guertel].sum())}")

    # Stabilitaet: gleiche Eingabe -> gleiche Auswahl (die Auswahl laeuft
    # ueber den Original-Index, damit dieselben Koerper gestreamt bleiben
    # und die Client-Interpolation nicht reisst).
    a = sess._lod_auswahl(alle, x, y, box, 60_000)
    b = sess._lod_auswahl(alle, x, y, box, 60_000)
    assert np.array_equal(a, b), "Auswahl nicht deterministisch"
    print("Auswahl deterministisch OK")

    print("Alle Dichte-LOD-Tests bestanden.")


if __name__ == "__main__":
    main()
