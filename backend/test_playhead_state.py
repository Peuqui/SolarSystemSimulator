"""Zustand am Playhead aus dem Ring (FilmSession.state_at_playhead).

Geprueft wird die Stelle, an der sich Fehler still auswirken wuerden: die
Zuordnung Slot <-> Sim-Zeit, der Ringumlauf und die Klemmung auf den
tatsaechlich vorhandenen Bereich. Eine Verwechslung um einen Slot faellt
im Betrieb nicht auf — die Szene laeuft danach nur ein halbes Raster
falsch weiter.

Laeuft ohne GPU: die Session wird nicht gestartet, nur die vom Verfahren
benutzten Felder werden gesetzt.
"""
import struct
import sys
from types import SimpleNamespace

import numpy as np

import server


# capacity/head so gewaehlt, dass der Ring einmal umlaeuft UND die
# 16-Slot-Sicherheitsmarge von tail_abs einen echten Bereich uebrig
# laesst (tail_abs = head - capacity + 16 = 52, gueltig sind 52..99).
def baue_session(n=5, capacity=64, raster=0.5, t0=100.0, head=100):
    """Ring mit erkennbarem Inhalt: Slot i traegt x = i, y = 1000+i,
    vx = 2000+i, vy = 3000+i (je Koerper um dessen Index versetzt)."""
    s = server.FilmSession.__new__(server.FilmSession)
    s.n = n
    s.capacity = capacity
    s.raster_days = raster
    s.t0 = t0
    s.sample_bytes = 16 * n
    s.head_val = SimpleNamespace(value=head)
    buf = bytearray(capacity * s.sample_bytes)
    for i in range(head):
        slot = i % capacity
        koerper = np.arange(n, dtype=np.float32)
        block = np.concatenate([
            i + koerper, 1000.0 + i + koerper,
            2000.0 + i + koerper, 3000.0 + i + koerper]).astype("<f4")
        buf[slot * s.sample_bytes:(slot + 1) * s.sample_bytes] = block.tobytes()
    s.shm = SimpleNamespace(buf=buf)
    return s


def entpacke(paket, n):
    typ, m, t = struct.unpack_from("<IId", paket, 0)
    werte = np.frombuffer(paket, "<f8", 4 * n, 16)
    return typ, m, t, werte


def pruefe(bedingung, text):
    print(("  ok   " if bedingung else "  FEHL ") + text)
    return bedingung


def main():
    ok = True
    n, capacity, raster, t0, head = 5, 64, 0.5, 100.0, 100
    s = baue_session(n, capacity, raster, t0, head)

    # Slot i traegt die Zeit t0 + (i+1)*raster — dieselbe Konvention wie
    # batch() und build_frame(). Der Playhead auf genau dieser Zeit muss
    # exakt diesen Slot liefern.
    for i in (60, 75, 99):
        t_soll = t0 + (i + 1) * raster
        typ, m, t, werte = entpacke(s.state_at_playhead(t_soll), n)
        ok &= pruefe(typ == 5 and m == n, f"Kopf korrekt (Slot {i})")
        ok &= pruefe(abs(t - t_soll) < 1e-12,
                     f"Zeit trifft Slot {i} ({t} == {t_soll})")
        ok &= pruefe(np.allclose(werte[0:n], i + np.arange(n)),
                     f"x aus Slot {i} (Ringumlauf {i % capacity})")
        ok &= pruefe(np.allclose(werte[n:2 * n], 1000.0 + i + np.arange(n)),
                     f"y aus Slot {i}")
        ok &= pruefe(np.allclose(werte[2 * n:3 * n], 2000.0 + i + np.arange(n)),
                     f"vx aus Slot {i}")
        ok &= pruefe(np.allclose(werte[3 * n:], 3000.0 + i + np.arange(n)),
                     f"vy aus Slot {i}")

    # Playhead zwischen zwei Samples: der naechstgelegene Slot gewinnt.
    _, _, t, werte = entpacke(
        s.state_at_playhead(t0 + 76.4 * raster), n)
    ok &= pruefe(np.allclose(werte[0:n], 75 + np.arange(n)),
                 "Playhead zwischen Samples rundet auf den naechsten Slot")

    # Vor dem Tail und hinter dem Kopf: klemmen statt danebengreifen.
    _, _, _, werte = entpacke(s.state_at_playhead(t0 - 500.0), n)
    ok &= pruefe(np.allclose(werte[0:n], s.tail_abs + np.arange(n)),
                 "Playhead vor dem Tail klemmt auf tail_abs")
    _, _, _, werte = entpacke(s.state_at_playhead(t0 + 9999.0), n)
    ok &= pruefe(np.allclose(werte[0:n], (head - 1) + np.arange(n)),
                 "Playhead hinter dem Kopf klemmt auf head-1")

    # Ring noch leer -> None, damit der Aufrufer auf den f64-Dump faellt.
    leer = baue_session(head=0)
    ok &= pruefe(leer.state_at_playhead(t0) is None,
                 "leerer Ring liefert None (Rueckfall auf f64-Dump)")

    print("\n" + ("ALLE TESTS BESTANDEN" if ok else "FEHLGESCHLAGEN"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
