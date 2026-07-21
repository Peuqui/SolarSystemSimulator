"""Ring-Layout des Film-Modus: Slot schreiben (Producer) und lesen (Server).

Geprueft werden die Stellen, an denen sich Fehler still auswirken wuerden:

  a) Roundtrip des Slots — Positionen und Sub-Stuetzpunkte kommen genau so
     zurueck, wie der Producer sie abgelegt hat. Ein vertauschtes
     Feld-Layout faellt sonst erst als "Bahn sitzt neben ihrem Punkt" auf.
  b) Ueberlauf des Sub-Blocks — mehr heisse Koerper als Plaetze: die
     NIEDRIGEN Indizes bleiben (Koerper des geladenen Systems vor
     injizierten Wolken), und der Rest wird sauber abgeschnitten.
  c) Zustand am Playhead — Slot <-> Sim-Zeit, Ringumlauf, Klemmung auf den
     vorhandenen Bereich und die Geschwindigkeit aus dem zentralen
     Differenzenquotienten. Eine Verwechslung um einen Slot faellt im
     Betrieb nicht auf: die Szene laeuft danach nur ein halbes Raster
     falsch weiter.

Laeuft ohne GPU: die Session wird nicht gestartet, nur die vom Verfahren
benutzten Felder werden gesetzt.

Aufruf: ./venv/bin/python backend/test_playhead_state.py
"""
import struct
import sys
from types import SimpleNamespace

import numpy as np

import film_producer
import server

RASTER = 0.5
T0 = 100.0


def baue_session(n=5, capacity=64, head=100, m_sub=4, sub_max=3,
                 bahn=None):
    """Ring mit erkennbarem Inhalt. Slot i traegt x = i + koerperIndex,
    y = 1000 + i + koerperIndex. `bahn(i)` liefert optional (idx, pos) im
    Format von NBodyCuda._collect_sub.

    capacity/head sind so gewaehlt, dass der Ring einmal umlaeuft UND die
    16-Slot-Sicherheitsmarge von tail_abs einen echten Bereich uebrig
    laesst (tail_abs = head - capacity + 16 = 52, gueltig sind 52..99).
    """
    s = server.FilmSession.__new__(server.FilmSession)
    s.n = n
    s.capacity = capacity
    s.raster_days = RASTER
    s.t0 = T0
    s.m_sub = m_sub
    s.sub_max = sub_max
    s.sample_bytes = film_producer.slot_bytes(n, m_sub, sub_max)
    s.head_val = SimpleNamespace(value=head)
    buf = bytearray(capacity * s.sample_bytes)
    koerper = np.arange(n, dtype=np.float32)
    for i in range(head):
        # Kernel-Ausgabe ist [x|y|vx|vy]; nur x|y darf im Ring landen.
        out = np.concatenate([
            i + koerper, 1000.0 + i + koerper,
            -np.ones(n), -np.ones(n)]).astype(np.float32)
        film_producer.schreibe_slot(
            buf, (i % capacity) * s.sample_bytes, n, sub_max, out,
            bahn(i) if bahn else None)
    s.shm = SimpleNamespace(buf=buf)
    return s


def entpacke(paket, n):
    typ, m, t = struct.unpack_from("<IId", paket, 0)
    werte = np.frombuffer(paket, "<f8", 4 * n, 16)
    return typ, m, t, werte


def pruefe(bedingung, text):
    print(("  ok   " if bedingung else "  FEHL ") + text)
    return bool(bedingung)


def test_roundtrip():
    """a) Positionen und Stuetzpunkte kommen unveraendert zurueck."""
    ok = True
    n, m_sub = 5, 4

    def bahn(i):
        # Zwei heisse Koerper (Index 3 und 1 — UNSORTIERT uebergeben),
        # je mSub Stuetzpunkte mit eindeutigen Werten.
        idx = np.array([3, 1], dtype=np.int64)
        pos = np.zeros((m_sub, 2, 2), np.float32)
        for j in range(m_sub):
            pos[j, 0, 0] = 300 + i + j      # Koerper 3, x
            pos[j, 1, 0] = 400 + i + j      # Koerper 3, y
            pos[j, 0, 1] = 100 + i + j      # Koerper 1, x
            pos[j, 1, 1] = 200 + i + j      # Koerper 1, y
        return idx, pos

    s = baue_session(n=n, m_sub=m_sub, sub_max=3, bahn=bahn)
    i = 75
    pos = s.slot_pos(i)
    ok &= pruefe(np.allclose(pos[0:n], i + np.arange(n)),
                 "x kommt zurueck (Ringumlauf)")
    ok &= pruefe(np.allclose(pos[n:2 * n], 1000.0 + i + np.arange(n)),
                 "y kommt zurueck")
    ok &= pruefe(len(pos) == 2 * n,
                 "kein v im Ring (Slot traegt nur x|y)")

    sidx, bahnen = s.slot_sub(i)
    ok &= pruefe(list(sidx) == [1, 3],
                 f"Sub-Indizes aufsteigend sortiert ({list(sidx)})")
    ok &= pruefe(bahnen.shape == (2, m_sub, 2),
                 f"Bahn-Form (nh, mSub, 2) — ist {bahnen.shape}")
    erwartet1 = [[100 + i + j, 200 + i + j] for j in range(m_sub)]
    erwartet3 = [[300 + i + j, 400 + i + j] for j in range(m_sub)]
    ok &= pruefe(np.allclose(bahnen[0], erwartet1),
                 "Stuetzpunkte von Koerper 1 (x,y je Punkt korrekt)")
    ok &= pruefe(np.allclose(bahnen[1], erwartet3),
                 "Stuetzpunkte von Koerper 3 folgen ihrem Index")

    leer = baue_session(n=n, m_sub=m_sub, sub_max=3)
    lidx, lbahn = leer.slot_sub(60)
    ok &= pruefe(len(lidx) == 0 and lbahn.shape[0] == 0,
                 "Sample ohne heisse Koerper liefert leeren Sub-Block")
    return ok


def test_ueberlauf():
    """b) Mehr heisse Koerper als Plaetze: niedrige Indizes gewinnen."""
    ok = True
    n, m_sub, sub_max = 10, 2, 3

    def bahn(i):
        idx = np.array([7, 2, 9, 0, 5], dtype=np.int64)   # 5 > sub_max
        pos = np.zeros((m_sub, 2, 5), np.float32)
        for sp, ki in enumerate(idx):
            for j in range(m_sub):
                pos[j, 0, sp] = 10 * ki + j
                pos[j, 1, sp] = 10 * ki + j + 0.5
        return idx, pos

    s = baue_session(n=n, m_sub=m_sub, sub_max=sub_max, bahn=bahn)
    sidx, bahnen = s.slot_sub(80)
    ok &= pruefe(list(sidx) == [0, 2, 5],
                 f"die drei niedrigsten Indizes bleiben ({list(sidx)})")
    ok &= pruefe(len(bahnen) == sub_max,
                 "nicht mehr Bahnen als Plaetze")
    ok &= pruefe(np.allclose(bahnen[1, :, 0], [20, 21]),
                 "Bahn gehoert zum richtigen Koerper (Index 2)")

    voll = film_producer.schreibe_slot(
        bytearray(film_producer.slot_bytes(n, m_sub, sub_max)), 0, n,
        sub_max, np.zeros(4 * n, np.float32), bahn(0))
    ok &= pruefe(voll, "Ueberlauf wird gemeldet")
    return ok


def test_playhead():
    """c) Slot <-> Zeit, Klemmung, Geschwindigkeit aus dem Ring."""
    ok = True
    n, head = 5, 100
    s = baue_session(n=n, head=head)

    for i in (60, 75, 99 - 1):
        t_soll = T0 + (i + 1) * RASTER
        typ, m, t, werte = entpacke(s.state_at_playhead(t_soll), n)
        ok &= pruefe(typ == 5 and m == n, f"Kopf korrekt (Slot {i})")
        ok &= pruefe(abs(t - t_soll) < 1e-12,
                     f"Zeit trifft Slot {i} ({t} == {t_soll})")
        ok &= pruefe(np.allclose(werte[0:n], i + np.arange(n)),
                     f"x aus Slot {i} (Ringumlauf {i % s.capacity})")
        ok &= pruefe(np.allclose(werte[n:2 * n], 1000.0 + i + np.arange(n)),
                     f"y aus Slot {i}")
        # x waechst je Slot um 1, der zentrale Differenzenquotient ueber
        # zwei Raster ergibt also exakt 1 / raster_jahre.
        v_soll = 1.0 / (RASTER / server.TAGE_PRO_JAHR)
        ok &= pruefe(np.allclose(werte[2 * n:3 * n], v_soll),
                     f"vx aus zentralem Differenzenquotienten (Slot {i})")
        ok &= pruefe(np.allclose(werte[3 * n:], v_soll),
                     f"vy aus zentralem Differenzenquotienten (Slot {i})")

    _, _, _, werte = entpacke(s.state_at_playhead(T0 + 76.4 * RASTER), n)
    ok &= pruefe(np.allclose(werte[0:n], 75 + np.arange(n)),
                 "Playhead zwischen Samples rundet auf den naechsten Slot")

    # Klemmung: der Differenzenquotient braucht beide Nachbarn, der
    # nutzbare Bereich ist daher [tail_abs+1, head-2].
    _, _, _, werte = entpacke(s.state_at_playhead(T0 - 500.0), n)
    ok &= pruefe(np.allclose(werte[0:n], s.tail_abs + 1 + np.arange(n)),
                 "Playhead vor dem Tail klemmt auf tail_abs+1")
    _, _, _, werte = entpacke(s.state_at_playhead(T0 + 9999.0), n)
    ok &= pruefe(np.allclose(werte[0:n], (head - 2) + np.arange(n)),
                 "Playhead hinter dem Kopf klemmt auf head-2")

    knapp = baue_session(n=n, head=2)
    ok &= pruefe(knapp.state_at_playhead(T0) is None,
                 "zu wenige Samples liefern None (Rueckfall auf f64-Dump)")
    return ok


def main():
    ok = True
    for name, fn in (("a) Slot-Roundtrip", test_roundtrip),
                     ("b) Sub-Ueberlauf", test_ueberlauf),
                     ("c) Zustand am Playhead", test_playhead)):
        print(f"\n{name}")
        ok &= fn()
    print("\n" + ("ALLE TESTS BESTANDEN" if ok else "FEHLGESCHLAGEN"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
