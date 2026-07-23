"""Protokoll-Vertrag: build_frame (Server) gegen zerlegeFilm (Client).

Der Frame ist die Schnittstelle zwischen zwei Sprachen — hier laufen
Server und Client am ehesten auseinander, und zwar STILL: ein um vier
Bytes verschobener Offset liefert keine Fehlermeldung, sondern Positionen
irgendwo im Nichts.

Dieser Test dekodiert das Frame Feld fuer Feld nach demselben Schema wie
`zerlegeFilm` in index.html (Kommentar dort haelt das Layout fest) und
prueft die Werte gegen den bekannten Ring-Inhalt:

  a) Meta-Block (Box, logFlag, mSub, verworfene Ereignisse) und Zeiten
     liegen an den Offsets, die der Worker liest.
  b) Positionen kommen quantisiert, aber richtig zurueck.
  c) Der Sub-Block enthaelt genau die heissen Koerper, die AUCH gestreamt
     werden — und ihre Stuetzpunkte in derselben Box-Quantisierung wie
     die Positionen (sonst sitzt die Bahn neben ihrem eigenen Punkt).
  d) Bei grobem Streaming spannen die Stuetzpunkte trotzdem das GANZE
     Intervall (aus mehreren Rastern zusammengesetzt). Faellt das weg,
     ueberbrueckt der Client den Rest mit der Sehne — die schneidet den
     Bogen ab, und die Koerper rutschen sichtbar sonnenwaerts.
  e) Das erste Sample eines Frames traegt nie einen Block: sein Vorgaenger
     liegt im vorigen Frame und koennte nach einem Sprung ganz woanders
     sitzen.

Laeuft ohne GPU. Aufruf: ./venv/bin/python backend/test_film_protokoll.py
"""
import pathlib
import re
import struct
import sys
from types import SimpleNamespace

import numpy as np

import film_producer
import server

RASTER = 0.5
T0 = 100.0
N = 6
M_SUB = 4
SUB_MAX = 4
# Heisse Koerper mit Stuetzpunkten; 5 liegt ausserhalb der Kamera-Box und
# darf daher NICHT im Sub-Block landen.
HEISS = [1, 3, 5]


def welt(i, k):
    """Position von Koerper k im Sample i — auseinanderliegend, damit
    Verwechslungen auffallen. Koerper 5 sitzt weit ausserhalb."""
    if k == 5:
        return 500.0 + i, 500.0 + i
    return k * 0.5 + i * 0.01, k * 0.25 - i * 0.01


def sub_bahn(i):
    idx = np.array(HEISS, dtype=np.int64)
    pos = np.zeros((M_SUB, 2, len(HEISS)), np.float32)
    for sp, k in enumerate(idx):
        x, y = welt(i, k)
        for j in range(M_SUB):
            # Stuetzpunkte laufen vom Vorgaenger-Sample zu diesem
            pos[j, 0, sp] = x - 0.01 * (M_SUB - 1 - j) / M_SUB
            pos[j, 1, sp] = y + 0.01 * (M_SUB - 1 - j) / M_SUB
    return idx, pos


def baue_session(head=40, capacity=64):
    s = server.FilmSession.__new__(server.FilmSession)
    s.n = N
    # Kein server-seitiger Tracer-Auftrag: alle Koerper sind Massen.
    s.n_mass = N
    s.capacity = capacity
    s.raster_days = RASTER
    s.t0 = T0
    s.m_sub = M_SUB
    s.sub_max = SUB_MAX
    s.sample_bytes = film_producer.slot_bytes(N, M_SUB, SUB_MAX)
    s.head_val = SimpleNamespace(value=head)
    s.ev_count_val = SimpleNamespace(value=0)
    s.ev_cap = 16
    s.ev_shm = SimpleNamespace(buf=bytearray(16 * film_producer.EV_BYTES))
    s.sent_ev = 0
    s.ev_verworfen = 0
    s._kill_t = np.full(N, np.inf)
    s._is_ast = np.ones(N, bool)
    s._injiziert = np.zeros(N, bool)
    s.lod_budget = 0
    # Kamera-Box um den Ursprung: gross genug fuer 0..5 AE, aber weit
    # weg von Koerper 5 bei 500 AE.
    s.view = (1.0, 0.0, 4.0, 4.0)
    s.log_zoom = False
    buf = bytearray(capacity * s.sample_bytes)
    for i in range(head):
        out = np.zeros(4 * N, np.float32)
        for k in range(N):
            out[k], out[N + k] = welt(i, k)
        film_producer.schreibe_slot(
            buf, (i % capacity) * s.sample_bytes, N, SUB_MAX, out,
            sub_bahn(i))
    s.shm = SimpleNamespace(buf=buf)
    return s


def zerlege_film(frame):
    """Frame nach dem Schema von zerlegeFilm (index.html) zerlegen.

    Traegt wie der Worker die letzte Indexliste ueber Frame-Grenzen
    (Attribut `letzte_idx`) — v6 laesst sie weg, solange sie gleich
    bleibt."""
    status, n, count, ev_count = struct.unpack_from("<IIII", frame, 0)
    box = struct.unpack_from("<dddd", frame, 32)          # x0,y0,sx,sy
    log = struct.unpack_from("<d", frame, 64)[0] != 0
    m_sub = int(struct.unpack_from("<d", frame, 72)[0])
    verworfen = struct.unpack_from("<d", frame, 80)[0]
    times = np.frombuffer(frame, "<f8", count, 88)
    samples = []
    off = 88 + 8 * count
    for _ in range(count):
        (roh,) = struct.unpack_from("<I", frame, off)
        # v6: Bit 31 = Indexliste wie im vorigen Sample, sie fehlt dann.
        sel_gleich = bool(roh & 0x8000_0000)
        vis_count = roh & 0x7FFF_FFFF
        if sel_gleich:
            idx = zerlege_film.letzte_idx
            d_off = off + 4
        else:
            idx = np.frombuffer(frame, "<u4", vis_count, off + 4)
            zerlege_film.letzte_idx = idx
            d_off = off + 4 + 4 * vis_count
        qx = np.frombuffer(frame, "<u2", vis_count, d_off)
        qy = np.frombuffer(frame, "<u2", vis_count, d_off + 2 * vis_count)
        # Anonymer Tracer-Block (v7): anzahl | qx | qy, ohne Index.
        toff = d_off + 4 * vis_count
        (t_count,) = struct.unpack_from("<I", frame, toff)
        tqx = np.frombuffer(frame, "<u2", t_count, toff + 4)
        tqy = np.frombuffer(frame, "<u2", t_count, toff + 4 + 2 * t_count)
        soff = toff + 4 + 4 * t_count
        (sub_count,) = struct.unpack_from("<I", frame, soff)
        p = sub_count * m_sub
        sub_idx = np.frombuffer(frame, "<u4", sub_count, soff + 4)
        sqx = np.frombuffer(frame, "<u2", p, soff + 4 + 4 * sub_count)
        sqy = np.frombuffer(frame, "<u2", p,
                            soff + 4 + 4 * sub_count + 2 * p)
        samples.append(SimpleNamespace(idx=idx, qx=qx, qy=qy,
                                       tqx=tqx, tqy=tqy,
                                       sub_idx=sub_idx, sqx=sqx, sqy=sqy))
        block = (4 + (0 if sel_gleich else 4 * vis_count) +
                 4 * vis_count + 4 + 4 * t_count + 4 + 4 * sub_count + 4 * p)
        off += block + (-block) % 4
    rest = len(frame) - off - ev_count * film_producer.EV_BYTES
    return SimpleNamespace(status=status, n=n, count=count, box=box,
                           log=log, m_sub=m_sub, times=times,
                           samples=samples, rest=rest, verworfen=verworfen)


zerlege_film.letzte_idx = None


def entpacke(q, box, achse):
    """u16 zurueck in Weltkoordinaten (Client: _filmEntpacke, kein Log)."""
    x0, y0, sx, sy = box
    return (x0 + q * (sx / 65535)) if achse == 0 else (y0 + q * (sy / 65535))


def pruefe(bedingung, text):
    print(("  ok   " if bedingung else "  FEHL ") + text)
    return bool(bedingung)


def main():
    ok = True
    s = baue_session()

    print("\na) Kopf, Meta und Zeiten")
    frame = s.build_frame([20, 21, 22])
    f = zerlege_film(frame)
    # Toleranz = anderthalb Quantisierungsschritte der TATSAECHLICHEN Box
    # (die ist 2x Viewport, nicht 1x). astype("<u2") schneidet ab statt zu
    # runden, ein voller Schritt Abweichung ist also normal.
    tol = max(f.box[2], f.box[3]) / 65535 * 1.5
    ok &= pruefe(f.status == 4 and f.n == N and f.count == 3,
                 "Kopf: status/N/count")
    ok &= pruefe(not f.log, "logFlag 0 bei gesetzter Kamera-Box")
    ok &= pruefe(f.m_sub == M_SUB, f"mSub im Meta ({f.m_sub})")
    ok &= pruefe(f.verworfen == 0.0,
                 "verworfene Ereignisse im Meta (hier 0)")
    ok &= pruefe(np.allclose(f.times,
                             [T0 + (i + 1) * RASTER for i in (20, 21, 22)]),
                 "Zeiten je Sample")
    ok &= pruefe(f.rest == 0,
                 f"Frame geht genau auf ({f.rest} Bytes Rest)")

    print("\nb) Positionen")
    sm = f.samples[1]
    sichtbar = sorted(k for k in range(N) if k != 5)
    ok &= pruefe(list(sm.idx) == sichtbar,
                 f"nur Koerper in der Box gestreamt ({list(sm.idx)})")
    fehler = 0.0
    for m, k in enumerate(sm.idx):
        wx, wy = welt(21, int(k))
        fehler = max(fehler, abs(entpacke(sm.qx[m], f.box, 0) - wx),
                     abs(entpacke(sm.qy[m], f.box, 1) - wy))
    ok &= pruefe(fehler < tol,
                 f"Positionen dekodieren zurueck (max {fehler:.2e} AE)")

    print("\nc) Sub-Block")
    ok &= pruefe(list(sm.sub_idx) == [1, 3],
                 f"nur GESTREAMTE heisse Koerper ({list(sm.sub_idx)}) — "
                 "Koerper 5 ist heiss, aber ausserhalb der Box")
    ok &= pruefe(len(sm.sqx) == len(sm.sub_idx) * M_SUB,
                 "mSub Stuetzpunkte je Koerper")
    _, pos = sub_bahn(21)
    fehler = 0.0
    for reihe, k in enumerate(sm.sub_idx):
        sp = HEISS.index(int(k))
        for j in range(M_SUB):
            gx = entpacke(sm.sqx[reihe * M_SUB + j], f.box, 0)
            gy = entpacke(sm.sqy[reihe * M_SUB + j], f.box, 1)
            fehler = max(fehler, abs(gx - pos[j, 0, sp]),
                         abs(gy - pos[j, 1, sp]))
    ok &= pruefe(fehler < tol,
                 f"Stuetzpunkte in derselben Box-Quantisierung "
                 f"(max {fehler:.2e} AE)")
    # Der letzte Stuetzpunkt IST die Sample-Position — sonst springt der
    # Koerper am Intervallende auf seinen eigenen Punkt zurueck.
    reihe = list(sm.sub_idx).index(1)
    m = list(sm.idx).index(1)
    ok &= pruefe(
        abs(entpacke(sm.sqx[reihe * M_SUB + M_SUB - 1], f.box, 0) -
            entpacke(sm.qx[m], f.box, 0)) < tol,
        "letzter Stuetzpunkt trifft die Sample-Position")

    print("\nd) Grobes Streaming: Kette spannt trotzdem das Intervall")
    grob = zerlege_film(s.build_frame([20, 24, 28]))
    gm = grob.samples[1]           # Sample 24, Vorgaenger 20 (step 4)
    ok &= pruefe(grob.m_sub == M_SUB,
                 f"mSub unveraendert gemeldet ({grob.m_sub})")
    ok &= pruefe(list(gm.sub_idx) == [1, 3],
                 f"Stuetzpunkte auch bei step 4 ({list(gm.sub_idx)})")
    # Bei step 4 und mSub 4 stammt Punkt j aus Slot 21+j, jeweils dessen
    # LETZTER Stuetzpunkt — die Kette bleibt aequidistant ueber 4 Raster.
    reihe = list(gm.sub_idx).index(1)
    fehler = 0.0
    for j in range(M_SUB):
        _, quelle = sub_bahn(21 + j)
        sp = HEISS.index(1)
        gx = entpacke(gm.sqx[reihe * M_SUB + j], grob.box, 0)
        gy = entpacke(gm.sqy[reihe * M_SUB + j], grob.box, 1)
        fehler = max(fehler, abs(gx - quelle[M_SUB - 1, 0, sp]),
                     abs(gy - quelle[M_SUB - 1, 1, sp]))
    ok &= pruefe(fehler < tol,
                 f"Punkt j stammt aus Raster j des Intervalls "
                 f"(max {fehler:.2e} AE)")
    ok &= pruefe(grob.rest == 0, "Frame geht auch dann genau auf")

    print("\ne) Erstes Sample eines Frames ohne Block")
    ok &= pruefe(len(f.samples[0].sub_idx) == 0,
                 "kein Sub-Block im ersten Sample (Vorgaenger unbekannt)")
    ok &= pruefe(len(f.samples[2].sub_idx) > 0,
                 "aber in den folgenden")

    print("\nf) Aufloesung haengt am SICHTFENSTER, nicht an der Szene")
    # Der Kern des Fehlers, der die Wolke vor der Sonne zusammenschnappen
    # liess: die u16-Schrittweite ist Box-Breite / 65535. Solange die Box
    # ueber ALLE Koerper ging, verdarb ein einziger weit ausgeworfener
    # Asteroid die Aufloesung im Zentrum.
    eng = baue_session()
    eng.view = (0.0, 0.0, 0.01, 0.01)      # sehr nah gezoomt
    eng.log_zoom = False
    nah = zerlege_film(eng.build_frame([20, 21]))
    schritt = nah.box[2] / 65535
    ok &= pruefe(schritt < 1e-5,
                 f"Schrittweite folgt dem Zoom ({schritt:.2e} AE)")
    # Dieselbe Szene, dieselbe Kamera — aber ein Koerper weit draussen.
    # Frueher haette das die Box aufgespannt; jetzt darf es nichts aendern.
    eng2 = baue_session()
    eng2.view = (0.0, 0.0, 0.01, 0.01)
    eng2.log_zoom = False
    weit = zerlege_film(eng2.build_frame([20, 21]))
    ok &= pruefe(abs(weit.box[2] - nah.box[2]) < 1e-12,
                 "ein Koerper auf 500 AE veraendert die Box NICHT")
    # Und im Log-Zoom ebenso: Box aus dem Sichtfenster, nicht aus min/max
    eng3 = baue_session()
    eng3.view = (0.0, 0.0, 0.01, 0.01)
    eng3.log_zoom = True
    lz = zerlege_film(eng3.build_frame([20, 21]))
    ok &= pruefe(lz.log, "logFlag gesetzt")
    ok &= pruefe(abs(lz.box[2] - 4 * 0.01) < 1e-12,
                 f"auch im Log-Zoom = 2x Viewport ({lz.box[2]:.4f})")

    print("\ng) v6: die Indexliste faellt weg, solange sie gleich bleibt")
    # Der teuerste Posten im Strom (4 von 8 Byte je Punkt). Bei
    # stehender Kamera und ohne LOD-Ausduennung ist die Auswahl
    # konstant — dann darf sie NUR im ersten Sample stehen.
    sv = baue_session()
    roh_flags = []
    frame_v6 = sv.build_frame([20, 21, 22, 23])
    off = 88 + 8 * 4
    fv = zerlege_film(frame_v6)          # fuellt letzte_idx korrekt
    # Flags direkt aus dem Frame lesen, unabhaengig vom Dekoder
    for smp in fv.samples:
        (roh,) = struct.unpack_from("<I", frame_v6, off)
        roh_flags.append(bool(roh & 0x8000_0000))
        vis = roh & 0x7FFF_FFFF
        t_off = off + 4 + (0 if roh_flags[-1] else 4 * vis) + 4 * vis
        (t_count,) = struct.unpack_from("<I", frame_v6, t_off)
        sub_off = t_off + 4 + 4 * t_count
        (sub_count,) = struct.unpack_from("<I", frame_v6, sub_off)
        blk = (4 + (0 if roh_flags[-1] else 4 * vis) + 4 * vis +
               4 + 4 * t_count + 4 + 4 * sub_count + 4 * sub_count * fv.m_sub)
        off += blk + (-blk) % 4
    ok &= pruefe(roh_flags[0] is False,
                 "erstes Sample traegt die volle Liste")
    ok &= pruefe(all(roh_flags[1:]),
                 f"die folgenden nicht mehr ({roh_flags})")
    ok &= pruefe(all(np.array_equal(fv.samples[0].idx, s2.idx)
                     for s2 in fv.samples[1:]),
                 "dekodiert ergeben alle dieselbe Auswahl")
    # Der eigentliche Beweis, dass die Bytes wirklich FEHLEN: Der
    # Dekoder laeuft ueber den Frame und landet exakt am Ende. Haette
    # der Server die Listen doch mitgeschickt (oder der Dekoder sie
    # faelschlich uebersprungen), liefe er aus dem Tritt.
    vis0 = len(fv.samples[0].idx)
    ok &= pruefe(fv.rest == 0,
                 f"Frame geht genau auf ({3 * 4 * vis0} B gespart)")

    print("\ni) v6 haelt auch, wenn die Auswahl MITTENDRIN wechselt")
    # Der Fall aus dem Betrieb: Der Nutzer zoomt oder schwenkt, der
    # Server cullt daraufhin anders — mitten in einer Folge von Samples,
    # die ihre Indexliste weglassen. Das Flag darf dann NICHT stehen,
    # sonst legt der Client Positionen auf die falschen Koerper.
    sw = baue_session()
    zerlege_film.letzte_idx = None
    vorige = None
    for hw in (4.0, 4.0, 0.6, 600.0, 0.6):
        sw.view = (1.0, 0.0, hw, hw)
        fr = sw.build_frame([20])
        (roh,) = struct.unpack_from("<I", fr, 88 + 8)
        flag = bool(roh & 0x8000_0000)
        d = zerlege_film(fr)
        idx = np.asarray(d.samples[0].idx)
        gewechselt = vorige is None or not np.array_equal(idx, vorige)
        ok &= pruefe(not (flag and gewechselt),
                     f"hw={hw:6.1f}: {len(idx)} sichtbar, Flag={flag}, "
                     f"gewechselt={gewechselt}")
        ok &= pruefe(d.rest == 0 and len(d.samples[0].qx) == len(idx),
                     f"hw={hw:6.1f}: Frame geht auf, Laengen passen")
        vorige = idx

    print("\nh) Client und Server nennen dieselbe Protokollversion")
    # Die Zahl steht zwangslaeufig doppelt: server.py kennt sie als
    # Konstante, index.html schreibt sie als Literal in den FILM_START.
    # Laufen sie auseinander, lehnt der Server JEDEN Filmstart ab —
    # sichtbar nur als "Film-Protokollversion veraltet" im Browser,
    # waehrend beide Seiten fuer sich genommen fehlerfrei aussehen.
    # (Genau so passiert, als v6 eingefuehrt wurde.)
    html = (pathlib.Path(__file__).resolve().parent.parent /
            "index.html").read_text(encoding="utf-8")
    treffer = re.findall(r"\((\d+) << 4\) \| \(astAstCollisions", html)
    ok &= pruefe(len(treffer) == 1,
                 f"genau eine Versionsstelle im Client ({len(treffer)})")
    if treffer:
        ok &= pruefe(int(treffer[0]) == server.FILM_PROTO_VERSION,
                     f"Client {treffer[0]} == Server "
                     f"{server.FILM_PROTO_VERSION}")

    print("\n" + ("ALLE TESTS BESTANDEN" if ok else "FEHLGESCHLAGEN"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
