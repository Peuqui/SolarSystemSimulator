"""Regelt sich die Streifenaufteilung der Erkennung wirklich ein?

Die Erkennung teilt die Szene in x-Streifen auf die freien Karten auf.
Frueher bekam jeder Streifen gleich viele Asteroiden — das ist aber nicht
gleich viel ARBEIT, weil die Paarzahl mit der Dichte quadratisch waechst.
Gemessen an einer klumpigen Szene bekam eine von zwei Karten 81,8 Mio
Kandidatenpaare, die andere 42,2 Mio; die schnellere wartete 44 von 96 ms.

`_lastanteile` schreibt die Anteile aus der gemessenen Kandidatenzahl
fort. Dieser Test braucht dafuer keine GPU: Er modelliert die Szene als
Dichteprofil und rechnet die Last eines Streifens aus dem Integral der
quadrierten Dichte — genau die Abhaengigkeit, um die es geht.

Geprueft wird, was schiefgehen kann:
  A) Konvergenz — gleicht sich die Last aus?
  B) Stabilitaet — schwingt der Regler nicht?
  C) Kein Weglaufen — bei bereits gleicher Last darf er nicht wandern.
     Genau das passierte mit der Zeit als Regelgroesse: ein schmaler
     Streifen behaelt seinen vollen Halo, wird dadurch scheinbar nie
     schneller und lief bis an die Untergrenze (0,93 / 0,07 bei einer
     gleichmaessigen Szene, in der beide Karten gleich viel zu tun
     hatten).
  D) Grenzen — verhungert kein Streifen, bleibt die Summe 1?
  E) Ruhe — bei bereits gleicher Last bleibt alles, wie es ist.

Aufruf: ./venv/bin/python backend/test_lastregler.py
"""
from __future__ import annotations

import sys

import numpy as np

from film_producer import LAST_MIN_ANTEIL, _lastanteile

SAMPLES = 60


def last_modell(anteile, dichte, tempo=None):
    """Kandidatenpaare je Karte fuer eine gegebene Aufteilung.

    `dichte` ist die Teilchendichte ueber x (gleichmaessiges Raster).
    Ein Streifen bekommt einen ANTEIL der Teilchen — seine Grenzen
    ergeben sich also aus den Quantilen der Dichte, nicht aus gleichen
    x-Abstaenden. Seine Arbeit ist das Integral der quadrierten Dichte
    darueber (Paarzahl ~ Dichte^2). `tempo` skaliert sie optional, um
    ungleich schnelle Karten zu modellieren."""
    kum = np.cumsum(dichte) / dichte.sum()
    zeiten, unten = [], 0.0
    for i, a in enumerate(anteile):
        oben = min(1.0, unten + a)
        # Indexbereich, der diesen Teilchenanteil abdeckt
        i0 = int(np.searchsorted(kum, unten))
        i1 = max(i0 + 1, int(np.searchsorted(kum, oben)))
        arbeit = float((dichte[i0:i1] ** 2).sum())
        zeiten.append(arbeit / (tempo[i] if tempo else 1.0))
        unten = oben
    return zeiten


def profil(art: str, m=40_000):
    """Dichteprofil ueber x, fein aufgeloest.

    Die Aufloesung ist nicht beliebig: `last_modell` schneidet die
    Streifen ueber Indexgrenzen (`searchsorted`), und ein grobes Raster
    laesst die modellierte Last in Stufen springen. Bei m=4000 erzeugte
    allein das eine Restschwankung von 2,7 %, die faelschlich als
    Schwingen des Reglers gelesen wurde — sie blieb bei jeder
    Regelverstaerkung exakt gleich (0,15 bis 0,50), was ein Regler nie
    tut. Mit m=40.000 sinkt sie auf 0,1 %."""
    x = np.linspace(-5, 5, m)
    if art == "gleichmaessig":
        return np.ones(m)
    if art == "knoedel":                 # dichter Klumpen + duenner Rest
        return np.exp(-0.5 * (x / 0.15) ** 2) * 50 + 1.0
    if art == "doppelklumpen":           # zwei Klumpen, asymmetrisch
        return (np.exp(-0.5 * ((x + 2) / 0.2) ** 2) * 40
                + np.exp(-0.5 * ((x - 1.5) / 0.4) ** 2) * 15 + 1.0)
    raise ValueError(art)


def einregeln(dichte, n_karten, tempo=None, samples=SAMPLES):
    anteile = [1.0 / n_karten] * n_karten
    verlauf = []
    for _ in range(samples):
        z = last_modell(anteile, dichte, tempo)
        verlauf.append(max(z) / max(1e-12, min(z)))
        anteile = _lastanteile(anteile, z)
    return anteile, verlauf


def schieflage(anteile, dichte, tempo=None):
    z = last_modell(anteile, dichte, tempo)
    return max(z) / max(1e-12, min(z))


def a_konvergenz() -> bool:
    print("\n--- A) Konvergenz")
    ok = True
    for art in ("gleichmaessig", "knoedel", "doppelklumpen"):
        for n in (2, 3):
            d = profil(art)
            start = schieflage([1.0 / n] * n, d)
            anteile, verlauf = einregeln(d, n)
            ende = verlauf[-1]
            gut = ende < 1.15 or ende < start * 0.5
            ok &= gut
            print(f"    {art:>14} {n} Karten: Schieflage "
                  f"{start:6.2f}x -> {ende:5.2f}x"
                  f"   {'ok' if gut else 'ZU HOCH'}")
    print("    " + ("BESTANDEN" if ok else "FEHLGESCHLAGEN"))
    return ok


def b_stabilitaet() -> bool:
    """Nach dem Einschwingen darf nichts mehr wandern."""
    print("\n--- B) Stabilitaet (kein Schwingen)")
    d = profil("knoedel")
    anteile, verlauf = einregeln(d, 3, samples=200)
    spaet = verlauf[-40:]
    schwankung = (max(spaet) - min(spaet)) / max(1e-12, min(spaet))
    ok = schwankung < 0.02
    print(f"    Schieflage der letzten 40 Samples: "
          f"{min(spaet):.4f} .. {max(spaet):.4f} "
          f"(Schwankung {schwankung * 100:.2f} %)")
    print("    " + ("BESTANDEN" if ok else
                    "FEHLGESCHLAGEN — der Regler schwingt"))
    return ok


def c_kein_weglaufen() -> bool:
    """Bei gleichmaessiger Szene muessen die Anteile gleich BLEIBEN.

    Mit der Zeit als Regelgroesse tat der Regler das nicht: Ein schmaler
    Streifen behaelt seinen vollen Halo, wird also nie schneller, und
    der Regler schob immer weiter — bis 0,93 / 0,07 bei nachweislich
    gleicher Last. Auf der GPU kostete das 10 % (Guertel wurde
    LANGSAMER als ohne Regelung)."""
    print("\n--- C) Kein Weglaufen bei ausgeglichener Szene")
    ok = True
    for n in (2, 3):
        d = profil("gleichmaessig")
        anteile, _ = einregeln(d, n, samples=200)
        abweichung = max(abs(a - 1.0 / n) for a in anteile)
        gut = abweichung < 0.02
        ok &= gut
        print(f"    {n} Karten nach 200 Samples: "
              f"{[round(a, 4) for a in anteile]}  "
              f"max. Abweichung {abweichung:.4f}"
              f"  {'ok' if gut else 'WEGGELAUFEN'}")
    print("    " + ("BESTANDEN" if ok else "FEHLGESCHLAGEN"))
    return ok


def d_grenzen() -> bool:
    """Summe bleibt 1, kein Streifen verhungert — auch im Extremfall."""
    print("\n--- D) Grenzen")
    ok = True
    faelle = [
        ("extreme Schieflage", [0.5, 0.5], [1e9, 1.0]),
        ("eine Karte leer", [0.5, 0.5], [1.0, 0.0]),
        ("alles null", [0.5, 0.5], [0.0, 0.0]),
        ("negativ (Messfehler)", [0.5, 0.5], [-1.0, 2.0]),
        ("eine Karte", [1.0], [42.0]),
        ("vier Karten", [0.25] * 4, [10.0, 1.0, 1.0, 1.0]),
    ]
    for name, anteile, zeiten in faelle:
        a = _lastanteile(list(anteile), list(zeiten))
        summe = sum(a)
        gut = (abs(summe - 1.0) < 1e-9 and len(a) == len(anteile)
               and all(np.isfinite(a))
               and (len(a) == 1 or min(a) >= LAST_MIN_ANTEIL - 1e-9))
        ok &= gut
        print(f"    {name:>22}: {[round(v, 4) for v in a]}"
              f"  Summe {summe:.6f} {'' if gut else ' FEHLER'}")
    # Wiederholte Extremschlaege duerfen nicht zum Verhungern fuehren.
    a = [0.5, 0.5]
    for _ in range(50):
        a = _lastanteile(a, [1e9, 1.0])
    gut = min(a) >= LAST_MIN_ANTEIL - 1e-9
    ok &= gut
    print(f"    {'50x Extremschlag':>22}: {[round(v, 4) for v in a]}"
          f"  {'' if gut else 'VERHUNGERT'}")
    print("    " + ("BESTANDEN" if ok else "FEHLGESCHLAGEN"))
    return ok


def e_ruhe() -> bool:
    """Ist die Last schon gleich, darf sich nichts bewegen."""
    print("\n--- E) Ruhe bei ausgeglichener Last")
    a = _lastanteile([0.4, 0.6], [1.0, 1.0])
    ok = abs(a[0] - 0.4) < 1e-9 and abs(a[1] - 0.6) < 1e-9
    print(f"    [0.4, 0.6] bei gleicher Zeit -> "
          f"{[round(v, 6) for v in a]}")
    print("    " + ("BESTANDEN" if ok else
                    "FEHLGESCHLAGEN — regelt ohne Anlass"))
    return ok


def main() -> int:
    ergebnisse = [a_konvergenz(), b_stabilitaet(), c_kein_weglaufen(),
                  d_grenzen(), e_ruhe()]
    print()
    if all(ergebnisse):
        print("ALLE TESTS BESTANDEN")
        return 0
    print(f"FEHLGESCHLAGEN ({ergebnisse.count(False)} von "
          f"{len(ergebnisse)})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
