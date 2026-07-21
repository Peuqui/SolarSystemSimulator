"""Producer-Prozess des Film-Modus.

Laeuft als EIGENER Python-Prozess (spawn) und besitzt die GPU exklusiv
fuer seine Session: kein GIL-Sharing mit dem WebSocket-Server. Der Server
beantwortet Batch-Anfragen in Mikrosekunden direkt aus dem Shared-Memory-
Ringpuffer, waehrend die GPU hier mit vollem Durchsatz rechnet.

Der Producer stoppt NUR, wenn der Ring voll ist und das aelteste Sample
noch vor dem Player-Playhead liegt (Ueberschreib-Schutz) — genau die
Semantik "GPU rennt frei und pausiert erst, wenn sie zu weit vorlaeuft".

Ring-Layout: capacity Slots fester Groesse sample_bytes.
Sample k (absoluter Zaehler) liegt in Slot k % capacity und traegt die
Sim-Zeit t0 + (k+1) * raster. Head/Playhead/Kollisionen laufen ueber
multiprocessing-Values.

Slot-Format: [x|y] als f32 in Originalreihenfolge, danach der Sub-Block
mit den Zwischenbildern der heissen Asteroiden (siehe slot_bytes).
Masse und Sichtbarkeit laufen als Ereignisse im Event-Ring.
"""
from __future__ import annotations

import os
import time

import cupy as cp
import numpy as np


# Diagnose: Produktions-/Verbrauchsbilanz. Beantwortet, ob der Player an
# der Kante klebt, weil die GPU nicht nachkommt (Produktionsrate <
# Abspielrate), oder weil der Ueberschreib-Schutz den Producer anhaelt
# (Drossel-Anteil > 0). Sekunden zwischen zwei Log-Zeilen.
DIAG_INTERVAL_S = 2.0

EV_BYTES = 32    # Ereignis: f64 tTage | u32 a (Ueberlebender/0xFFFFFFFF) |
#                  u32 b (Verlierer) | f32 neueMasse | u32 kind |
#                  f32 x | f32 y (exakter Ereignis-Ort — der Client kann
#                  ihn NICHT aus dem Stream rekonstruieren: das Opfer
#                  fehlt im Folge-Sample, seine interpolierte Position
#                  stammt je nach Stream-Dichte von Tagen davor)
#                  kind 0 = merge/kill (b verschwindet), 1 = bounce (nur
#                  Zaehler + Visual, niemand stirbt), 2 = Numerik-Waechter
#                  (b verschwindet wie bei 0, ist aber KEINE Kollision:
#                  kein Partner, kein Blitz, eigener Zaehler)

BOUNCE_E = 0.6      # Restitution (wie BOUNCE_RESTITUTION im JS)

# ---- Ring-Slot: Positionen + Zwischenbilder der heissen Asteroiden ----
#
# Ein Slot traegt x|y als f32 plus einen Sub-Block mit den Stuetzpunkten
# der heissen Asteroiden (Nahbegegnung, stark gekruemmte Bahn). Die
# Geschwindigkeit ist NICHT mehr dabei: sie diente allein der
# Client-Interpolation, und dafuer sind die Stuetzpunkte selbst da — als
# Messwert statt als Schaetzung. Das halbiert den Positionsteil und
# bezahlt den Sub-Block fast von allein.
#
# Der Slot muss feste Groesse haben (Ring!), die Zahl der heissen Koerper
# schwankt aber stark. Deshalb eine Obergrenze: hoechstens jeder
# SUB_MAX_ANTEIL-te Koerper bekommt einen Platz. Ueberzaehlige fallen im
# Client auf Catmull-Rom zurueck — dieselbe Qualitaet wie vor den
# Sub-Samples, also kein Rueckschritt, nur kein Gewinn.
SUB_MAX_ANTEIL = 32
SUB_MAX_MIN = 2048      # Untergrenze fuer kleine Szenen
# Stuetzpunkte je Raster fuer Aufrufer OHNE Client (Tests, Benchmarks).
# Im Betrieb bestimmt der Regler "Bahnstuetzpunkte" den Wert; er kommt
# im FILM_START-Paket und wird hier nie gelesen.
SUB_SAMPLES_DEFAULT = 8
# Obergrenze fuer die Stuetzpunkte je Raster. Sie deckelt den
# VRAM-Puffer im Kernel und muss zu nbody_kernel.SUB_SAMPLES_MAX passen;
# hier steht sie noch einmal, weil der Serverprozess nbody_kernel nicht
# importieren darf (der Import legt einen CUDA-Kontext an — die GPU
# gehoert dem Producer-Kindprozess).
M_SUB_MAX = 32


def sub_max_fuer(n: int) -> int:
    return min(n, max(SUB_MAX_MIN, n // SUB_MAX_ANTEIL))


def slot_bytes(n: int, m_sub: int, sub_max: int) -> int:
    """Groesse eines Ring-Slots. SSOT fuer Producer und Server —
    x|y f32 | u32 Anzahl | u32 Indizes | f32 Stuetzpunkte [nh][mSub][2]."""
    return 8 * n + 4 + sub_max * (4 + 8 * m_sub)


def schreibe_slot(buf, basis: int, n: int, sub_max: int,
                  out_i, sub_i) -> bool:
    """Ein Ring-Sample ablegen: Positionen + Sub-Block. Gibt True zurueck,
    wenn mehr heisse Koerper anfielen als Plaetze da sind.

    `out_i` ist die Kernel-Ausgabe [x|y|vx|vy] (4n) — nur der Positions-
    teil geht in den Ring. `sub_i` ist (idx, pos) aus
    NBodyCuda._collect_sub: idx sind Originalindizes, pos ist
    (mSub, 2, nh). Im Ring liegen die Stuetzpunkte KOERPERWEISE
    (nh, mSub, 2), damit der Server die Bahn eines gestreamten Koerpers
    am Stueck herausschneiden kann.

    Die Indizes werden sortiert abgelegt: der Server schneidet sie per
    searchsorted gegen seine LOD-Auswahl. Beim Ueberlauf bleiben dadurch
    die NIEDRIGEN Indizes — das sind die Koerper des geladenen Systems,
    nachtraeglich injizierte Wolken fallen zuerst heraus. Dieselbe
    Rangfolge wie beim Dichte-LOD des Streams.
    """
    import struct
    buf[basis:basis + 8 * n] = out_i[0:2 * n].tobytes()
    off = basis + 8 * n
    if sub_i is None or not sub_max:
        buf[off:off + 4] = struct.pack("<I", 0)
        return False
    idx, pos = sub_i
    ueberlauf = len(idx) > sub_max
    ordnung = np.argsort(idx, kind="stable")[:sub_max]
    nh = len(ordnung)
    buf[off:off + 4] = struct.pack("<I", nh)
    off += 4
    buf[off:off + 4 * nh] = idx[ordnung].astype("<u4").tobytes()
    off += 4 * sub_max                     # Indexfeld hat feste Laenge
    if nh:
        bahnen = np.ascontiguousarray(
            pos[:, :, ordnung].transpose(2, 0, 1)).astype("<f4")
        buf[off:off + bahnen.nbytes] = bahnen.tobytes()
    return ueberlauf

# Erkennungskarten pro Session. Die Bounce-Suche ist der Engpass (75-93%
# der Batchzeit); sie skaliert raeumlich, weil Kollisionen lokal sind.
DET_MAX = 2

# Umschaltschwellen der zweiten Erkennungskarte, in KANDIDATENPAAREN pro
# Sample — nicht in Koerpern! In einem dichten Klumpen sitzen tausende
# Koerper in derselben Gitterzelle, und Paare wachsen quadratisch: 250k
# Asteroiden ergeben dann ueber 10^9 zu pruefende Paare, waehrend
# derselbe Bestand als ausgebildeter Guertel unter 10^6 bleibt.
#
# Gemessen mit bench_erkennung.py (250k Asteroiden, zwei RTX 8000):
#     3,4 Mio    0,65x     22,9 Mio   1,55x
#     6,2 Mio    0,83x     91 Mio     1,72x
#    11,4 Mio    1,18x    178 Mio     1,65x
# Der Umschlagpunkt liegt bei rund 8-9 Mio; darunter kosten der doppelte
# Batch-Upload und die doppelten Kernel-Starts mehr, als die halbierte
# GPU-Arbeit einbringt. Die Schwellen liegen bewusst beidseitig daneben.
#
# Jenseits von ~10^10 Paaren faellt der Gewinn wieder auf 1,0x: dort wird
# die Zellgroesse h so gross, dass der Halo (2h) einen erheblichen Teil
# des Nachbarstreifens mit abdeckt und beide Karten dieselbe Arbeit
# doppelt tun. Ein solcher Zustand haelt nur waehrend der ersten
# Durchdringung frisch injizierter Wolken an — nicht optimiert.
DET_ZU_AB = 15_000_000
DET_AB_UNTER = 6_000_000

# Restitution und Suchgitter gehoeren zusammen: die Zellgroesse h richtet
# sich nach der Strecke, die ein schneller Koerper in einem Raster
# zuruecklegt. Nachbarzellen werden bis Versatz 1 geprueft, ein Paar kann
# also hoechstens 2h auseinanderliegen (plus Beruehrungsradien).
NACHBARN = ((0, 0), (1, 0), (0, 1), (1, 1), (1, -1))

# Kompakte Wolken (Injektion) haetten beim Voll-Materialisieren aller
# Kandidaten zweistellige Millionen Paare x mehrere Arrays (GBs VRAM-Peak,
# den der CuPy-Pool dauerhaft behielte). Daher Chunk-Verarbeitung:
# expandieren + sweepen in Stuecken, nur Treffer ueberleben. Physik
# identisch, Reihenfolge der Treffer identisch (Offsets + Chunks
# aufsteigend).
CHUNK = 4_000_000


class Erkennungskarte:
    """Eine Erkennungs-GPU samt ihrer residenten Hilfsarrays.

    Zustaendig ist sie fuer den x-Streifen [lo, hi) der Szene: ein
    Asteroidenpaar gehoert ihr, wenn min(x_i, x_j) darin liegt. Damit sie
    jedes eigene Paar auch sehen kann, prueft sie Koerper bis hi + halo —
    der Partner liegt hoechstens eine Suchreichweite weiter rechts, und
    weiter links als lo kann er nicht liegen (sonst waere ER das Minimum).

    Die Streifen sind dadurch lueckenlos UND ueberschneidungsfrei: kein
    Paar geht verloren, keines wird doppelt gezaehlt. Zwischen den Karten
    ist deshalb nichts abzustimmen — sie brauchen nur denselben Skalar
    `grenze` pro Sample, den der Aufrufer einmal bestimmt.

    Bei nur einer Karte ist der Streifen (-inf, +inf) und alle
    Streifen-Tests sind wirkungslos: EIN Codepfad fuer beide Faelle.
    """

    def __init__(self, dev: int, is_ast: np.ndarray, lo: float, hi: float):
        self.dev = dev
        self.lo = cp.float32(lo)
        self.hi = cp.float32(hi)
        with cp.cuda.Device(dev):
            self.g_ast = cp.asarray(is_ast)
        self.g_vis = None
        self.g_rr = None

    def stammdaten(self, vis: np.ndarray, real_r: np.ndarray) -> None:
        """Sichtbarkeit und Beruehrungsradien neu hochladen. Beide aendern
        sich nur bei Merges/Kills — sonst waeren es 2 H2D pro Sample."""
        with cp.cuda.Device(self.dev):
            self.g_vis = cp.asarray(vis)
            self.g_rr = cp.asarray(real_r.astype(np.float32))

    def hochladen(self, outs: np.ndarray):
        """EIN Host->GPU-Transfer fuer den GESAMTEN Batch statt vier pro
        Sample. Vorher waren das bei K=8 und n=267k 32 Einzeltransfers mit
        je eigenem Launch- und Sync-Overhead — die Erkennungs-GPU bekam
        ihre Arbeit in Haeppchen und lief bei ~38% Auslastung, waehrend
        der Hauptloop auf sie wartete."""
        with cp.cuda.Device(self.dev):
            return cp.asarray(outs)

    def streifen(self, lo: float, hi: float) -> None:
        """Zustaendigkeitsbereich fuer das naechste Sample setzen."""
        self.lo = cp.float32(lo)
        self.hi = cp.float32(hi)

    def bounce_hits(self, gx, gy, gvx, gvy, dt_y: float,
                    h: float, halo: float):
        """Asteroid-x-Asteroid-Stoesse im eigenen Streifen suchen.

        Rueckgabe: ((i, j) als Host-Arrays der Treffer-Paare oder None,
        Zahl der geprueften Kandidatenpaare). Mutiert nichts."""
        with cp.cuda.Device(self.dev):
            return self._hits(gx, gy, gvx, gvy, dt_y, h, halo)

    def _hits(self, gx, gy, gvx, gvy, dt_y, h, halo):
        # VORFILTER komplett in f32 (auf der f64-schwachen
        # Erkennungskarte ~30x schneller). Der Beruehrungsradius wird um
        # mehr als den f32-Rundungsfehler (~3e-7 AU bei 5 AU) aufgeweitet
        # — kein echter Treffer kann verloren gehen. Die wenigen
        # Kandidaten werden danach exakt in f64 nachgeprueft.
        F32_TOL = cp.float32(1e-6)
        g_alive = self.g_ast & (self.g_vis != 0) & \
            (gx >= self.lo) & (gx < self.hi + cp.float32(halo))
        ai = cp.flatnonzero(g_alive)
        if int(ai.size) < 2:
            return None, 0
        ix = cp.floor(gx[ai].astype(cp.float64) / h).astype(cp.int64)
        iy = cp.floor(gy[ai].astype(cp.float64) / h).astype(cp.int64)
        key = _zellschluessel(ix, iy)
        order = cp.argsort(key)
        ks = key[order]

        def sweep_hits(pi, pj):
            # f32-Vorfilter eines Kandidaten-Chunks: Beruehrung mit
            # aufgeweitetem Radius; Reihenfolge bleibt erhalten.
            dpx = gx[pj] - gx[pi]
            dpy = gy[pj] - gy[pi]
            dvx_ = gvx[pj] - gvx[pi]
            dvy_ = gvy[pj] - gvy[pi]
            dv2 = dvx_ * dvx_ + dvy_ * dvy_
            # Fenster BEIDSEITIG: [-dt, +dt] — der Live-Algo prueft per
            # CCD rueckwaerts ueber den Frame; nur vorwaerts verpasste
            # frontales Tunneling im letzten Kernel-Step.
            tmin = cp.clip(-(dpx * dvx_ + dpy * dvy_) /
                           cp.where(dv2 > 0, dv2, cp.float32(1.0)),
                           cp.float32(-dt_y), cp.float32(dt_y))
            cxm = dpx + dvx_ * tmin
            cym = dpy + dvy_ * tmin
            rsum = self.g_rr[pi] + self.g_rr[pj] + F32_TOL
            hidx = cp.flatnonzero(cxm * cxm + cym * cym <= rsum * rsum)
            return pi[hidx], pj[hidx]

        hit_i_parts = []
        hit_j_parts = []
        kandidaten = 0
        for ox, oy in NACHBARN:
            k2 = _zellschluessel(ix + ox, iy + oy)
            lo_r = cp.searchsorted(ks, k2, side="left")
            hi_r = cp.searchsorted(ks, k2, side="right")
            lens = hi_r - lo_r
            tot = int(lens.sum())
            kandidaten += tot
            if tot == 0:
                continue
            cum = cp.cumsum(lens)
            starts = cum - lens
            for c0 in range(0, tot, CHUNK):
                c1 = min(c0 + CHUNK, tot)
                r = cp.arange(c0, c1, dtype=cp.int64)
                rows = cp.searchsorted(cum, r, side="right")
                cols = r - starts[rows] + lo_r[rows]
                a = ai[rows]
                b = ai[order[cols]]
                keep = (a < b) if (ox == 0 and oy == 0) else (a != b)
                hi_c, hj_c = sweep_hits(a[keep], b[keep])
                if int(hi_c.size):
                    hit_i_parts.append(hi_c)
                    hit_j_parts.append(hj_c)
        if not hit_i_parts:
            return None, kandidaten
        ci = cp.concatenate(hit_i_parts)
        cj = cp.concatenate(hit_j_parts)
        # BESITZ-REGEL: nur Paare behalten, deren linkerer Partner im
        # eigenen Streifen liegt. Ein Paar im Halo findet die
        # Nachbarkarte ebenfalls — behalten darf es genau eine.
        # Auf der Trefferliste (wenige tausend) statt auf den Kandidaten
        # (Millionen) gefiltert.
        eigen = cp.flatnonzero(cp.minimum(gx[ci], gx[cj]) < self.hi)
        if int(eigen.size) == 0:
            return None, kandidaten
        ci = ci[eigen]
        cj = cj[eigen]
        # Exakte f64-Nachpruefung NUR der Vorfilter-Kandidaten (wenige
        # tausend statt Millionen Paare).
        dpx = gx[cj].astype(cp.float64) - gx[ci].astype(cp.float64)
        dpy = gy[cj].astype(cp.float64) - gy[ci].astype(cp.float64)
        dvx_ = gvx[cj].astype(cp.float64) - gvx[ci].astype(cp.float64)
        dvy_ = gvy[cj].astype(cp.float64) - gvy[ci].astype(cp.float64)
        dv2 = dvx_ * dvx_ + dvy_ * dvy_
        tmin = cp.clip(-(dpx * dvx_ + dpy * dvy_) /
                       cp.where(dv2 > 0, dv2, 1.0), -dt_y, dt_y)
        cxm = dpx + dvx_ * tmin
        cym = dpy + dvy_ * tmin
        rsum = self.g_rr[ci].astype(cp.float64) + \
            self.g_rr[cj].astype(cp.float64)
        hidx = cp.flatnonzero(cxm * cxm + cym * cym <= rsum * rsum)
        if int(hidx.size) == 0:
            return None, kandidaten
        return (cp.asnumpy(ci[hidx]), cp.asnumpy(cj[hidx])), kandidaten


def _zellschluessel(cx, cy):
    return cx * cp.int64(73856093) ^ cy * cp.int64(19349663)


def bounce_suche(det, det_pool, rows, dt_y: float, rr_max: float):
    """Asteroid-x-Asteroid-Stoesse ueber alle aktiven Erkennungskarten.

    Raeumlich aufgeteilt: Kollisionen sind lokal — zwei Koerper auf
    gegenueberliegenden Seiten der Sonne koennen sich in einem Raster
    nicht treffen. Jede Karte prueft ihren x-Streifen plus einen Halo in
    Suchreichweite und behaelt nur die Paare, deren linkerer Partner ihr
    gehoert (siehe Erkennungskarte). Zwischen den Karten wird NICHTS
    ausgetauscht — sie brauchen nur dieselben Skalare.

    rows[i] sind die vier Feld-Views des Samples auf Karte i; die LAENGE
    von rows bestimmt, ueber wie viele Karten aufgeteilt wird. Streifen 0
    rechnet der aufrufende Thread selbst, die uebrigen laufen im Pool.

    Rueckgabe: (treffer|None, geprueifte Kandidatenpaare, Zellgroesse)."""
    gx, gy, gvx, gvy = rows[0]
    # Suchreichweite und Streifengrenzen EINMAL global bestimmen und an
    # alle Karten geben (SSOT): rechnete jede Karte ihr eigenes h aus
    # ihrer Teilmenge, waeren die Halo-Breiten verschieden und die
    # Besitz-Regel nicht mehr lueckenlos.
    with cp.cuda.Device(det[0].dev):
        lebt = cp.flatnonzero(det[0].g_ast & (det[0].g_vis != 0))
        if int(lebt.size) < 2:
            return None, 0, 0.0
        v95 = float(cp.percentile(cp.hypot(gvx[lebt], gvy[lebt]), 95))
        grenzen = _streifengrenzen(gx[lebt], len(rows))
    h = max(1e-4, 2.0 * v95 * dt_y)
    # Halo: NACHBARN prueft bis Zellversatz 1, ein Paar liegt also
    # hoechstens 2h auseinander — plus die beiden Beruehrungsradien.
    halo = 2.0 * h + 2.0 * rr_max
    for karte, (lo, hi) in zip(det, grenzen):
        karte.streifen(lo, hi)

    rest = [det_pool.submit(karte.bounce_hits, *row, dt_y, h, halo)
            for karte, row in zip(det[1:len(rows)], rows[1:])]
    teile = [det[0].bounce_hits(gx, gy, gvx, gvy, dt_y, h, halo)]
    teile += [f.result() for f in rest]

    hits = [t for t, _ in teile if t is not None]
    kandidaten = sum(kand for _, kand in teile)
    if not hits:
        return None, kandidaten, h
    return ((np.concatenate([t[0] for t in hits]),
             np.concatenate([t[1] for t in hits])), kandidaten, h)


def _streifengrenzen(gx_lebt, anzahl: int) -> list[tuple[float, float]]:
    """Die Szene in `anzahl` x-Streifen mit gleich vielen lebenden
    Asteroiden schneiden. Gleiche Koerperzahl ist nicht dasselbe wie
    gleiche Arbeit (die Paarzahl waechst mit der Dichte quadratisch), aber
    ein Quantil ist ein Kernel-Aufruf — eine Lastmessung waere ein
    Rueckkanal von den Karten zum Host in einem Pfad, der ohnehin am GIL
    haengt. Die aeusseren Grenzen sind unendlich, damit die Streifen die
    Szene lueckenlos ueberdecken."""
    if anzahl < 2:
        return [(-np.inf, np.inf)]
    q = [100.0 * i / anzahl for i in range(1, anzahl)]
    schnitte = [float(v) for v in cp.asnumpy(cp.percentile(gx_lebt, q))]
    kanten = [-np.inf, *schnitte, np.inf]
    return list(zip(kanten[:-1], kanten[1:]))


def producer_main(shm_name: str, sample_bytes: int, capacity: int,
                  ev_name: str, ev_cap: int, ev_count_val,
                  dump_name: str, dump_req_val,
                  head_val, playhead_val, coll_val, running_val,
                  state: dict, raster_days: float, t0_days: float,
                  ast_bounce: bool = False,
                  shatter_flag=None, shatter_a=None, shatter_b=None,
                  shatter_t=None, det_rank: int = 0,
                  det_gpus: int = DET_MAX, diag: bool = False,
                  m_sub: int = 0, sub_max: int = 0) -> None:
    # CUDA-Kontexte erst IM Kindprozess anlegen (spawn-Kontext!)
    from multiprocessing import shared_memory

    from concurrent.futures import ThreadPoolExecutor

    from nbody_kernel import (G_AU, NBodyCuda, pick_detect_devices,
                              pick_devices)

    # Multi-GPU lohnt erst, wenn die Rechenlast den Barrier-Overhead
    # (~PCIe-Roundtrips pro Substep) klar uebersteigt — gemessen ab
    # ~30k Asteroiden. Darunter ist die beste Einzelkarte schneller.
    #
    # ACHTUNG, im Betrieb gegengemessen (n=317k, Film-Modus):
    #     3 GPUs Physik + RTX-Erkennung: 11,0-11,3 Tage/s
    #     1 GPU  Physik + V100-Erkennung: 1,7-2,2 Tage/s
    # Die Einzelkarte ist also 5-6x LANGSAMER, obwohl der isolierte
    # Kernel-Benchmark in test_kernel.py das Gegenteil nahelegt (dort
    # 1 GPU bis 200k vorn). Der Benchmark misst den Kernel ohne die
    # Erkennungs-Pipeline; sobald diese mitlaeuft, kehrt sich das Bild um.
    # Die Schwelle daher NICHT nach dem Benchmark anpassen.
    MULTI_GPU_AB = 30_000
    n_ast_total = int(np.count_nonzero(
        np.asarray(state["isAst"], dtype=np.uint8)))
    phys_devs = pick_devices() if n_ast_total >= MULTI_GPU_AB \
        else pick_devices()[:1]
    sim = NBodyCuda(phys_devs, m_sub=m_sub)
    # Kollisions-/Bounce-Erkennung auf die freien GPUs auslagern: sie
    # laeuft dann UEBERLAPPT mit dem naechsten Kernel-Step (Pipeline).
    # Preis: Erkennungs-Ergebnisse von Sample k werden erst vor Step k+2
    # angewandt (1 Raster Versatz) — feiner als der Live-JS-Algo, der
    # Kollisionen einmal pro Frame (oft 1-2 Tage) aufloest. Ohne freie
    # GPU laeuft die Analyse im selben Muster auf der Physik-GPU.
    # Mehr als eine Erkennungskarte lohnt nur fuer die Bounce-Suche — sie
    # allein ist raeumlich aufgeteilt. Ohne Bounces bliebe die zweite
    # Karte untaetig und bekaeme trotzdem jeden Batch geschickt.
    det_devs = pick_detect_devices(phys_devs, det_rank,
                                   max(1, det_gpus) if ast_bounce else 1)
    ausgelagert = bool(det_devs)
    if not ausgelagert:
        det_devs = [sim.device]
    ana_dev = det_devs[0]      # Merge-Erkennung (2-4%, nicht aufgeteilt)
    print(f"[film] physik auf gpus {phys_devs}, erkennung auf gpus "
          f"{det_devs}" + (" (pipelined)" if ausgelagert else
                           " (seriell, keine weitere gpu)"), flush=True)
    x = state["x"]
    y = state["y"]
    vx = state["vx"]
    vy = state["vy"]
    mass = np.array(state["mass"], dtype=np.float64, copy=True)
    real_r = np.array(state["realR"], dtype=np.float64, copy=True)
    vis = np.array(state["visible"], dtype=np.uint8, copy=True)
    is_ast = np.array(state["isAst"], dtype=np.uint8, copy=True) != 0
    is_star_bh = np.array(state.get("isStarBH",
                                    np.zeros(len(is_ast), np.uint8)),
                          dtype=np.uint8, copy=True) != 0
    # real_r mitgeben: damit erkennt die Feinschleife Beruehrungen mit
    # massiven Koerpern selbst — auf ~Radius/20 genau statt auf ein
    # Sample-Raster (bei einem Sonnensturz 0,07 AE).
    st = sim.load_state(x, y, vx, vy, mass, vis, state["isAst"], real_r)
    n = len(x)
    collisions = 0
    dt_years = raster_days / 365.25

    # Erkennungskarten: lueckenlose x-Streifen ueber die Szene. Die
    # Streifengrenze wird pro Sample neu bestimmt (Median der lebenden
    # Asteroiden) — die Szene wandert und verdichtet sich laufend, eine
    # feste Grenze liefe binnen Minuten aus dem Gleichgewicht.
    det = [Erkennungskarte(d, is_ast, -np.inf, np.inf) for d in det_devs]
    prev_det = None              # voriges Sample (fuer den Merge-Sweep)
    # Beruehrungsradius des groessten Asteroiden: geht in die Halo-Breite
    # ein. Wird zusammen mit den Stammdaten fortgeschrieben.
    ast_rr_max = [0.0]

    def stammdaten_laden() -> None:
        """vis/real_r auf ALLE Erkennungskarten spiegeln. Beide aendern
        sich nur bei Merges/Kills — sonst waeren es 2 H2D pro Sample und
        Karte."""
        rr32 = real_r.astype(np.float32)
        for karte in det:
            karte.stammdaten(vis, rr32)
        ast_rr_max[0] = float(real_r[is_ast].max()) if is_ast.any() else 0.0

    stammdaten_laden()
    det_dirty = [False]
    # Zeitanteile INNERHALB der Erkennung. Sie laeuft im Pipeline-Thread,
    # deshalb Listen statt nonlocal. Trennt GPU-Anteile (merge-det,
    # bounce-det) vom reinen Host-Anteil (bounce-host: bounce_deltas
    # rechnet komplett auf der CPU) — nur so ist entscheidbar, welcher
    # Teil sich ueberhaupt auf die GPU verlagern laesst.
    diag_t_merge_det = [0.0]
    diag_t_bounce_det = [0.0]
    diag_t_bounce_host = [0.0]
    # Kandidatenpaare im Gitter-Vorfilter: der eigentliche Kostentreiber
    # der Bounce-Erkennung. Die Zellgroesse h richtet sich nach der
    # GESCHWINDIGKEIT der schnellsten 5% und gilt global — in dichten
    # Clustern landen dadurch sehr viele Koerper in derselben Zelle und
    # die Paarzahl waechst quadratisch.
    diag_kandidaten = [0]
    diag_zellgroesse = [0.0]

    shm = shared_memory.SharedMemory(name=shm_name)
    buf = shm.buf
    ev_shm = shared_memory.SharedMemory(name=ev_name)
    ev_buf = ev_shm.buf
    dump_shm = shared_memory.SharedMemory(name=dump_name)
    k = 0

    def dump_state() -> None:
        # Exakten f64-Zustand (Originalreihenfolge) fuer die Uebergabe an
        # andere Engines exportieren — sonst verlieren alle Koerper beim
        # Verlassen des Film-Modus ihren Impuls (Samples tragen nur x,y).
        out4 = sim.export_f64(st).astype("<f8")
        dump_shm.buf[0:out4.nbytes] = out4.tobytes()

    import struct as _struct

    sub_ueberlauf = [0]
    # Beruehrungszeitpunkte aus der Feinschleife, in ABSOLUTEN Sample-
    # Einheiten (k + Bruchteil). Die Streckenpruefung der Erkennung findet
    # dieselben Treffer, kann sie zeitlich aber nur auf ein Raster genau
    # verorten; hier steht der Kontakt auf ~Radius/20. Eintraege werden
    # nach Gebrauch entfernt — sonst waechst das Dict ueber die Session.
    kernel_kontakt: dict[int, float] = {}

    def emit_event(a: int, b: int, new_mass: float, kind: int,
                   k_ev: int, ex: float, ey: float,
                   frac: float = 0.0) -> None:
        # Merge/Kill/Bounce als Ereignis in den Event-Ring — Samples selbst
        # tragen nur noch Positionen (reines Punkte-Streaming). k_ev ist
        # der Sample-Zaehler der ANALYSE (Pipeline: Anwendung 1 spaeter).
        #
        # `frac` verschiebt den Zeitpunkt INNERHALB des Rasters (negativ =
        # davor, positiv = danach) — er kommt aus der Streckenpruefung der
        # Erkennung. Ohne ihn faellt jedes Ereignis auf die Rastergrenze,
        # und ein Koerper, der mit 50 AE/Jahr in den Stern stuerzt,
        # verschwindet eine ganze Rasterweite zu frueh: bei 0,5 Tagen sind
        # das 0,07 AE, das Fuenfzehnfache des Sternradius.
        i = ev_count_val.value % ev_cap
        t_ev = t0_days + (k_ev + 1 + frac) * raster_days
        ev_buf[i * EV_BYTES:(i + 1) * EV_BYTES] = _struct.pack(
            "<dIIfIff", t_ev, a & 0xFFFFFFFF, b, new_mass, kind,
            float(ex), float(ey))
        ev_count_val.value += 1

    def analyze_merge(sx, sy, gx, gy, gvx, gvy):
        """Merge-Kandidaten + Runaways auf der ersten Erkennungs-GPU
        bestimmen. Reine Analyse — mutiert nichts; Anwendung im Hauptloop.

        NICHT raeumlich aufgeteilt: massiv-x-alles kostet nur 2-4% der
        Batchzeit, eine Aufteilung braechte hier nichts ausser Komplexitaet
        (und liefe waehrend der Bounce-Suche ohnehin ueberlappt mit)."""
        nonlocal prev_det
        g_vis_a = det[0].g_vis
        g_rr_a = det[0].g_rr
        hit_pairs = []
        runaway_np = None
        px, py = (gx, gy) if prev_det is None else prev_det
        m_idx_all = st["m_idx_h"]
        m_alive = m_idx_all[(vis[m_idx_all] != 0) & (mass[m_idx_all] > 0)]
        if len(m_alive):
            gm = cp.asarray(m_alive)
            cx = gx[gm][:, None]
            cy = gy[gm][:, None]
            rsum = g_rr_a[gm][:, None] + g_rr_a[None, :]
            rsum2 = rsum * rsum
            alive = g_vis_a != 0

            def seg_hit(p0x, p0y, p1x, p1y):
                """Trifft die STRECKE p0->p1 die Kugel? Liefert zusaetzlich
                tt in [0,1] — wo auf der Strecke der Punkt der groessten
                Annaeherung liegt. Ohne tt waere nur bekannt, dass der
                Treffer irgendwo in diesem Raster passiert; der Koerper
                verschwaende dann eine ganze Rasterweite zu frueh, bei
                einem Sonnensturz also bis zu 0,07 AE VOR dem Stern."""
                ssx = (p1x - p0x)[None, :]
                ssy = (p1y - p0y)[None, :]
                seg2 = ssx * ssx + ssy * ssy
                tt = ((cx - p0x[None, :]) * ssx +
                      (cy - p0y[None, :]) * ssy) / cp.where(
                          seg2 > 0, seg2, cp.float32(1.0))
                tt = cp.clip(tt, 0.0, 1.0)
                ddx = p0x[None, :] + tt * ssx - cx
                ddy = p0y[None, :] + tt * ssy - cy
                return ddx * ddx + ddy * ddy <= rsum2, tt

            g_dt = cp.float32(dt_years)
            # Segment 1 endet auf DIESEM Sample, Segment 2 extrapoliert ins
            # naechste. Der Bruchteil wird auf das Sample-Ende bezogen:
            # negativ = vor diesem Sample, positiv = danach.
            hit_a, tt_a = seg_hit(px, py, gx, gy)
            hit_b, tt_b = seg_hit(gx, gy, gx + gvx * g_dt, gy + gvy * g_dt)
            hit2d = (hit_a | hit_b) & alive[None, :]
            frac2d = cp.where(hit_a, tt_a - cp.float32(1.0), tt_b)
            hit2d[cp.arange(len(gm)), gm] = False
            rows, cols = cp.nonzero(hit2d)
            if rows.size:
                fr = cp.asnumpy(frac2d[rows, cols]).tolist()
                hit_pairs = list(zip(cp.asnumpy(rows).tolist(),
                                     cp.asnumpy(cols).tolist(), fr))
            mw = mass[m_alive]
            msum = float(mw.sum())
            if msum > 0:
                bx = float((sx[m_alive] * mw).sum() / msum)
                by = float((sy[m_alive] * mw).sum() / msum)
                r = cp.maximum(cp.hypot(gx - cp.float32(bx),
                                        gy - cp.float32(by)),
                               cp.float32(1e-6))
                v2 = gvx * gvx + gvy * gvy
                vesc2 = cp.float32(2.0 * G_AU * msum) / r
                runaway = (v2 > 9.0 * vesc2) & det[0].g_ast & alive
                ridx = cp.flatnonzero(runaway)
                if ridx.size:
                    runaway_np = cp.asnumpy(ridx)
        prev_det = (gx.copy(), gy.copy())
        pairs = [(int(m_alive[row]), int(j), float(fr))
                 for row, j, fr in hit_pairs]
        return pairs, runaway_np

    def apply_merges(sample, pairs, runaway_np, k_ev):
        """Merge-/Kill-Ergebnisse der Analyse auf den residenten Zustand
        (Physik-GPU) und die Host-Spiegel anwenden."""
        nonlocal collisions
        st_local = st
        sx = sample[0:n]
        sy = sample[n:2 * n]
        svx = sample[2 * n:3 * n]
        svy = sample[3 * n:4 * n]
        changed = False
        for mi, j, frac in pairs:
            # Hat die Feinschleife diesen Koerper selbst beruehren sehen?
            # Dann gilt IHR Zeitpunkt: sie prueft je Substep gegen die
            # echte Bahn, die Erkennung nur die Sehne zwischen zwei
            # Samples (bei einem Sturz 0,07 AE am Stueck).
            tk = kernel_kontakt.pop(j, None)
            if tk is not None:
                f_kern = tk - (k_ev + 1)
                if -1.0 <= f_kern <= 1.0:
                    frac = f_kern
            # Zerbersten (wie _tryCollide im JS): Koerper x Koerper ohne
            # Stern/SL bei vImp >= 1,5 vEsc. Der Producer erkennt NUR:
            # Zustand einfrieren, f64-Dump, Selbst-Stopp — die Fragment-
            # Physik macht der Client mit seinem shatter() (SSOT) und
            # startet den Film neu.
            if (shatter_flag is not None
                    and not is_ast[mi] and not is_ast[j]
                    and not is_star_bh[mi] and not is_star_bh[j]
                    and vis[mi] and vis[j]
                    and mass[mi] > 0 and mass[j] > 0):
                v_imp = float(np.hypot(sample[2 * n + j] - sample[2 * n + mi],
                                       sample[3 * n + j] - sample[3 * n + mi]))
                touch = max(1e-12, real_r[mi] + real_r[j])
                v_esc = float(np.sqrt(
                    2.0 * G_AU * (mass[mi] + mass[j]) / touch))
                if v_imp >= 1.5 * v_esc:
                    shatter_a.value = int(mi)
                    shatter_b.value = int(j)
                    shatter_t.value = t0_days + (k_ev + 1 + frac) * raster_days
                    dump_state()
                    shatter_flag.value = 1
                    return
            if not vis[mi] or not vis[j] or mass[mi] <= 0:
                continue
            a, b = (mi, j) if mass[mi] >= mass[j] else (j, mi)
            m_a, m_b = mass[a], mass[b]
            m_ges = m_a + m_b
            if m_ges <= 0:
                continue
            nx = (sx[a] * m_a + sx[b] * m_b) / m_ges
            ny = (sy[a] * m_a + sy[b] * m_b) / m_ges
            nvx = (svx[a] * m_a + svx[b] * m_b) / m_ges
            nvy = (svy[a] * m_a + svy[b] * m_b) / m_ges
            sim.apply_body_state(st_local, a, nx, ny, nvx, nvy, m_ges)
            sim.deactivate_body(st_local, b)
            mass[a] = m_ges
            mass[b] = 0.0
            vis[b] = 0
            real_r[a] = (real_r[a] ** 3 + real_r[b] ** 3) ** (1.0 / 3.0)
            sim.set_radius(st_local, a, float(real_r[a]))
            emit_event(a, b, float(m_ges), 0, k_ev, nx, ny, frac)
            collisions += 1
            changed = True
        if runaway_np is not None:
            for j in runaway_np:
                j = int(j)
                if not vis[j]:
                    continue
                sim.deactivate_body(st_local, j)
                mass[j] = 0.0
                vis[j] = 0
                # Numerik-Waechter: eigenes kind, damit der Client das
                # Aufraeumen nicht als Kollision zeigt. Frueher lief es
                # als kind 0 mit a=0xFFFFFFFF durch — der Client sah eine
                # Kollision ohne Partner und blitzte im Leeren.
                emit_event(0xFFFFFFFF, j, 0.0, 2, k_ev,
                           float(sx[j]), float(sy[j]))
                changed = True
        if changed:
            coll_val.value = collisions
            det_dirty[0] = True

    bounce_count = 0
    kand_letzte = [0]        # Kandidatenpaare des zuletzt geprueften Samples
    det_aktiv = [1]

    def karten_wahl() -> int:
        """Wie viele Erkennungskarten der naechste Batch benutzt.

        Die Aufteilung halbiert die GPU-ARBEIT, verdoppelt aber Uploads
        und Kernel-Starts und laesst zwei Threads um den GIL konkurrieren.
        Das lohnt nur, solange die Suche rechengebunden ist — gemessen
        (siehe bench_erkennung.py):

            dichte Klumpen, >10^8 Kandidaten/Sample: 2 Karten ~1,4-2,0x
            ausgebildeter Guertel, <10^6:             2 Karten ~0,7x

        Genau dieser Verlauf entsteht von selbst, wenn eng injizierte
        Wolken sich durchdringen (Erkennung am Anschlag, Physik-Karten
        langweilen sich) und mit der Zeit zu einem Guertel ausduennen.
        Deshalb wird pro Batch neu entschieden — ein Filmneustart ist
        dafuer NICHT noetig: die Streifen werden ohnehin je Sample
        zugeteilt, und eine ruhende Karte bekommt schlicht keinen Upload
        mehr. Ihre residenten Arrays bleiben liegen, das Zuschalten
        kostet daher nur den naechsten Batch-Upload.

        Getrennte Schwellen fuer Hoch- und Runterschalten (Hysterese):
        an der Kante pendelt die Kandidatenzahl sonst um einen einzigen
        Wert und die Karte wird im Sekundentakt zu- und abgeschaltet."""
        if len(det) < 2:
            return 1
        kand = kand_letzte[0]
        if det_aktiv[0] == 1 and kand > DET_ZU_AB:
            det_aktiv[0] = len(det)
        elif det_aktiv[0] > 1 and kand < DET_AB_UNTER:
            det_aktiv[0] = 1
        return det_aktiv[0]

    def analyze_bounce(rows):
        """Bounce-Suche fuer ein Sample; Anwendung als Deltas im
        Hauptloop. Die Aufteilungs-Logik steckt in bounce_suche (SSOT mit
        bench_erkennung.py)."""
        hits, kandidaten, h = bounce_suche(
            det, det_pool, rows, raster_days / 365.25, ast_rr_max[0])
        diag_kandidaten[0] += kandidaten
        diag_zellgroesse[0] = h
        kand_letzte[0] = kandidaten
        return hits

    def bounce_deltas(sample: np.ndarray, hits_host):
        """Host-Teil des Bounce-Algos: Ein-Stoss-Filter, Impuls und
        Ueberlapp-Push — wie im JS, aber als DELTAS (dx, dy, dvx, dvy),
        weil die Anwendung pipelined ein Sample spaeter erfolgt."""
        dt_y = raster_days / 365.25
        hi, hj = hits_host
        # NUR Views auf die f32-Daten — die vier Arrays werden
        # ausschliesslich ueber [hi]/[hj] indiziert, also ein paar Dutzend
        # Kollisionspartner. Ein volles .astype(float64) ueber alle n
        # Koerper kostete 4 x 2 MB Allokation PRO Sample und hielt dabei
        # den GIL; die Erkennung konnte deshalb nicht mit der Physik
        # ueberlappen (gemessen: wait ~= erkennung, also null Parallelitaet).
        x = sample[0:n]
        y = sample[n:2 * n]
        vxa = sample[2 * n:3 * n]
        vya = sample[3 * n:4 * n]
        used = np.zeros(n, dtype=bool)
        keep_idx = []
        for t_i in range(len(hi)):
            a_i = int(hi[t_i])
            b_i = int(hj[t_i])
            if used[a_i] or used[b_i]:
                continue
            used[a_i] = True
            used[b_i] = True
            keep_idx.append(t_i)
        if not keep_idx:
            return None
        keep_idx = np.asarray(keep_idx)
        hi = hi[keep_idx].astype(np.int64)
        hj = hj[keep_idx].astype(np.int64)
        # Erst indizieren, DANN nach f64 wandeln: identisches Ergebnis wie
        # die frueheren Voll-Konversionen (die Quelldaten sind f32), aber
        # ueber die Trefferliste statt ueber alle n Koerper.
        dpx = x[hj].astype(np.float64) - x[hi].astype(np.float64)
        dpy = y[hj].astype(np.float64) - y[hi].astype(np.float64)
        dvx_ = vxa[hj].astype(np.float64) - vxa[hi].astype(np.float64)
        dvy_ = vya[hj].astype(np.float64) - vya[hi].astype(np.float64)
        dv2 = dvx_ * dvx_ + dvy_ * dvy_
        # Kontaktzeitpunkt wie im JS-_tryCollide: statischer Hit ->
        # tContact = 0; sonst Bahnschnitt-Quadratik, Kontakt am
        # EINTRITT (tEnter, dort naehern sie sich nachweislich an),
        # geclippt auf das Frame-Fenster [-dt, +dt].
        touch = real_r[hi] + real_r[hj]
        c_ = dpx * dpx + dpy * dpy - touch * touch
        b_ = 2.0 * (dpx * dvx_ + dpy * dvy_)
        disc = b_ * b_ - 4.0 * dv2 * c_
        sq = np.sqrt(np.maximum(disc, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            t_enter = (-b_ - sq) / (2.0 * np.where(dv2 > 0, dv2, 1.0))
        t_c = np.where(c_ < 0.0, 0.0,
                       np.clip(np.nan_to_num(t_enter), -dt_y, dt_y))
        gueltig = (c_ < 0.0) | (disc >= -(touch * touch) * 0.01)
        ncx = dpx + dvx_ * t_c
        ncy = dpy + dvy_ * t_c
        dist = np.hypot(ncx, ncy)
        dist = np.where(dist > 1e-30, dist, 1.0)
        nx_ = ncx / dist
        ny_ = ncy / dist
        vrel = dvx_ * nx_ + dvy_ * ny_
        act = gueltig & (vrel < 0)
        if not act.any():
            return None
        hi = hi[act]
        hj = hj[act]
        nx_ = nx_[act]
        ny_ = ny_[act]
        vrel = vrel[act]
        mi_ = mass[hi]
        mj_ = mass[hj]
        with np.errstate(divide="ignore", invalid="ignore"):
            imp = -(1.0 + BOUNCE_E) * vrel / (1.0 / mi_ + 1.0 / mj_)
        imp = np.nan_to_num(imp)
        ddx = np.zeros(n)
        ddy = np.zeros(n)
        ddvx = np.zeros(n)
        ddvy = np.zeros(n)
        ddvx[hi] -= imp * nx_ / mi_
        ddvy[hi] -= imp * ny_ / mi_
        ddvx[hj] += imp * nx_ / mj_
        ddvy[hj] += imp * ny_ / mj_
        pdx = x[hj].astype(np.float64) - x[hi].astype(np.float64)
        pdy = y[hj].astype(np.float64) - y[hi].astype(np.float64)
        pdist = np.hypot(pdx, pdy)
        touch = real_r[hi] + real_r[hj]     # nach act-Filter neu gefiltert
        overlap = touch - pdist
        ol = (overlap > 0) & (pdist > 1e-30)
        if ol.any():
            oi = hi[ol]
            oj = hj[ol]
            pnx = pdx[ol] / pdist[ol]
            pny = pdy[ol] / pdist[ol]
            mges = mass[oi] + mass[oj]
            wa = mass[oj] / mges
            wb = mass[oi] / mges
            shift = overlap[ol] * 1.1
            ddx[oi] -= pnx * shift * wa
            ddy[oi] -= pny * shift * wa
            ddx[oj] += pnx * shift * wb
            ddy[oj] += pny * shift * wb
        betroffen = np.unique(np.concatenate([hi, hj]))
        deltas = np.stack([ddx[betroffen], ddy[betroffen],
                           ddvx[betroffen], ddvy[betroffen]], axis=1)
        return hi, hj, betroffen.astype(np.int64), deltas

    def apply_bounce(res_bounce, k_ev, res_sample):
        """Bounce-Deltas auf den residenten Zustand (Physik-GPU) und die
        Ereignisse/Zaehler anwenden."""
        nonlocal bounce_count, collisions
        hi, hj, betroffen, deltas = res_bounce
        sim.apply_deltas(st, betroffen, deltas)
        # Jeder Bounce zaehlt wie in den Live-Engines als Kollision und
        # geht als kind=1-Ereignis an den Client (Zaehler + Blitz), ohne
        # dass ein Koerper stirbt. a = schwererer (Flash-Track wie im JS).
        sample = res_sample
        for t_i in range(len(hi)):
            a_i = int(hi[t_i])
            b_i = int(hj[t_i])
            heavy, light = (a_i, b_i) if mass[a_i] >= mass[b_i] \
                else (b_i, a_i)
            emit_event(heavy, light, 0.0, 1, k_ev,
                       float(sample[heavy]), float(sample[n + heavy]))
        collisions += len(hi)
        coll_val.value = collisions
        bounce_count += len(hi)

    def analyze_batch_timed(outs: np.ndarray, k0: int) -> tuple:
        """Wie analyze_batch, misst aber die REINE Rechenzeit im
        Pipeline-Thread. Vergleich mit der Wartezeit des Hauptloops zeigt,
        ob die Erkennung tatsaechlich mit der Physik ueberlappt: wartet der
        Loop etwa so lange, wie die Analyse rechnet, laeuft nichts parallel
        (GIL-Konkurrenz oder Transfer-Serialisierung)."""
        _t = time.monotonic()
        res = analyze_batch(outs, k0)
        return res, time.monotonic() - _t

    def analyze_batch(outs: np.ndarray, k0: int) -> list:
        # EIN Host->GPU-Transfer fuer den GESAMTEN Batch statt vier pro
        # Sample. Vorher waren das bei K=8 und n=267k 32 Einzeltransfers
        # mit je eigenem Launch- und Sync-Overhead — die Erkennungs-GPU
        # bekam ihre Arbeit in Haeppchen und lief bei ~38% Auslastung,
        # waehrend der Hauptloop auf sie wartete (gemessen: wait ~=
        # erkennung, also keinerlei Ueberlappung mit der Physik).
        # outs ist (K, 4n) f32 und zusammenhaengend, die Zeilen-Views
        # brauchen daher keine weitere Kopie.
        #
        # JEDE Erkennungskarte bekommt den GESAMTEN Batch und filtert
        # ihren Streifen selbst auf der GPU. Die Alternative — Masken auf
        # dem Host bilden und nur die jeweilige Haelfte uebertragen —
        # spart Transfer, kostet aber CPU-Zeit im Erkennungs-Thread, und
        # genau dort klemmt der GIL.
        aktiv = karten_wahl()
        rest = [det_pool.submit(karte.hochladen, outs)
                for karte in det[1:aktiv]]
        g_all = [det[0].hochladen(outs)] + [f.result() for f in rest]
        if det_dirty[0]:
            stammdaten_laden()
            det_dirty[0] = False
        return [analyze_sample(outs[i], k0 + i,
                               [g[i] for g in g_all])
                for i in range(len(outs))]

    def analyze_sample(out_np: np.ndarray, k_ev: int, g_zeilen) -> dict:
        """Komplette Erkennung eines Samples auf den Erkennungs-GPUs —
        laeuft im Pipeline-Thread, mutiert nichts. vis/mass/real_r werden
        hier nur GELESEN; der Hauptloop mutiert sie erst nach dem
        Einsammeln des Ergebnisses (keine Gleichzeitigkeit).

        g_zeilen[i] ist die bereits auf Karte i liegende Zeile des Batches
        (siehe analyze_batch) — hier wird nur noch geschnitten, nicht mehr
        transferiert."""
        rows = [(g[0:n], g[n:2 * n], g[2 * n:3 * n], g[3 * n:4 * n])
                for g in g_zeilen]
        _t = time.monotonic()
        with cp.cuda.Device(ana_dev):
            pairs, runaway_np = analyze_merge(
                out_np[0:n], out_np[n:2 * n], *rows[0])
        diag_t_merge_det[0] += time.monotonic() - _t
        _t = time.monotonic()
        hits = analyze_bounce(rows) if ast_bounce else None
        diag_t_bounce_det[0] += time.monotonic() - _t
        _t = time.monotonic()
        bounce = bounce_deltas(out_np, hits) if hits is not None else None
        diag_t_bounce_host[0] += time.monotonic() - _t
        return {"k": k_ev, "sample": out_np, "pairs": pairs,
                "runaways": runaway_np, "bounce": bounce}

    def apply_analysis(res: dict) -> None:
        nonlocal diag_t_merges, diag_t_bounce
        _t = time.monotonic()
        apply_merges(res["sample"], res["pairs"], res["runaways"], res["k"])
        diag_t_merges += time.monotonic() - _t
        if res["bounce"] is not None:
            _t = time.monotonic()
            apply_bounce(res["bounce"], res["k"], res["sample"])
            diag_t_bounce += time.monotonic() - _t

    executor = ThreadPoolExecutor(max_workers=1)
    # Fuer die Streifen 1..N-1; Streifen 0 rechnet der Erkennungs-Thread
    # selbst. Kein Pool bei nur einer Karte.
    det_pool = ThreadPoolExecutor(max_workers=len(det) - 1) \
        if len(det) > 1 else None
    future = None
    # Batch-Groesse: K Raster pro Kernel-Launch. Erkennungs-Ergebnisse
    # werden nach dem Batch angewandt — mit K=8 (4 Tage) bleibt der
    # Versatz in der Groessenordnung grosser Live-Frames, halbiert aber
    # den Launch-/Pipeline-Overhead nochmals.
    K = 8
    diag_prev_s = time.monotonic()
    diag_prev_k = k
    diag_prev_ph = playhead_val.value
    diag_throttled_s = 0.0
    # Zeitanteile im Loop: trennt GPU-Warten von echter CPU-Arbeit. Nur so
    # ist entscheidbar, ob Auslagern etwas bringt — 100% CPU-Last kann bei
    # CUDA auch reines Spin-Wait auf die GPU sein.
    diag_t_step = 0.0        # sim.step_batch (Kernel + Sync)
    diag_t_analyse = 0.0     # Summe: Einsammeln + Anwenden der Analyse
    diag_t_wait = 0.0        # davon: Blockieren in future.result()
    diag_t_erkennung = 0.0   # reine Rechenzeit im Erkennungs-Thread
    diag_t_merges = 0.0      # davon apply_merges (GPU-Roundtrips pro Paar)
    diag_t_bounce = 0.0      # davon apply_bounce (Deltas + emit_event)
    diag_t_ring = 0.0        # tobytes + Kopie in den Shared-Memory-Ring
    # Elternwaechter: die PID des Servers beim Start merken. Aendert sie
    # sich, ist der Server weg und wir wurden von init (oder einem
    # subreaper) adoptiert — dann beenden wir uns selbst.
    #
    # `daemon=True` allein genuegt nicht: Python beendet Daemon-Kinder ueber
    # einen atexit-Handler, und der laeuft NICHT, wenn der Server ein
    # SIGTERM oder SIGKILL bekommt. Genau so entstehen die Waisen, die
    # danach stundenlang CUDA-Kontexte und damit GPU-Speicher halten,
    # ohne je wieder zu rechnen (beobachtet: 1,35 GB ueber vier Karten,
    # 14 h lang, nachdem ein Test per `timeout` abgewuergt worden war).
    #
    # Der Vergleich gegen die GEMERKTE PID statt gegen 1 ist noetig, weil
    # unter systemd nicht init adoptiert, sondern der User-Manager.
    eltern_pid = os.getppid()
    try:
        while running_val.value:
            if os.getppid() != eltern_pid:
                break        # verwaist — Server ist weg
            if shatter_flag is not None and shatter_flag.value:
                break        # Zerbersten erkannt — Client uebernimmt
            # Pipeline: Erkennungs-Ergebnisse des VORIGEN Batches
            # einsammeln und anwenden, bevor der naechste Launch startet.
            if future is not None:
                _t = time.monotonic()
                ergebnisse, dauer_erkennung = future.result()
                diag_t_wait += time.monotonic() - _t
                diag_t_erkennung += dauer_erkennung
                for res in ergebnisse:
                    apply_analysis(res)
                diag_t_analyse += time.monotonic() - _t
                future = None
                # ERST JETZT sind die Ereignisse des vorigen Batches im
                # Ring — vorher gemeldete Samples haetten der Player
                # abgespielt, bevor ihre Kollisionen ueberhaupt
                # existieren (die Explosionen kamen 1-4 Tage zu spaet).
                # head laeuft daher bewusst K Raster hinter der
                # Rohproduktion her: was gemeldet wird, ist vollstaendig.
                head_val.value = k
            if dump_req_val.value == 1:
                dump_state()
                dump_req_val.value = 2
            # Ueberschreib-Schutz: Slot (k - capacity) wird gleich
            # ueberschrieben — er muss hinter dem Player-Playhead liegen.
            ph_val = playhead_val.value
            ph_abs = int((ph_val - t0_days) / raster_days)
            diag_now_s = time.monotonic()
            diag_span = diag_now_s - diag_prev_s
            if diag_span >= DIAG_INTERVAL_S:
                # Beide Raten in Sim-Tagen/s — direkt vergleichbar: liegt
                # prod unter play, laeuft der Puffer zwangslaeufig leer.
                prod_rate = (k - diag_prev_k) * raster_days / diag_span
                play_rate = (ph_val - diag_prev_ph) / diag_span
                vorlauf_abs = k - ph_abs
                if diag:
                    print(f"[film-diag] n={n} raster={raster_days:g}d "
                          f"prod={prod_rate:.1f}d/s play={play_rate:.1f}d/s "
                          f"vorlauf={vorlauf_abs * raster_days:.0f}d "
                          f"({100.0 * vorlauf_abs / capacity:.0f}% ring) "
                          f"drossel={100.0 * diag_throttled_s / diag_span:.0f}% "
                          f"| step={100.0 * diag_t_step / diag_span:.0f}% "
                          f"apply={100.0 * diag_t_analyse / diag_span:.0f}% "
                          f"(wait={100.0 * diag_t_wait / diag_span:.0f}% "
                          f"erkennung={100.0 * diag_t_erkennung / diag_span:.0f}% "
                          f"merge={100.0 * diag_t_merges / diag_span:.0f}% "
                          f"bounce={100.0 * diag_t_bounce / diag_span:.0f}%) "
                          f"ring={100.0 * diag_t_ring / diag_span:.0f}% "
                          f"| det: mergeGPU="
                          f"{100.0 * diag_t_merge_det[0] / diag_span:.0f}% "
                          f"bounceGPU="
                          f"{100.0 * diag_t_bounce_det[0] / diag_span:.0f}% "
                          f"bounceCPU="
                          f"{100.0 * diag_t_bounce_host[0] / diag_span:.0f}% "
                          f"kandidaten/sample="
                          f"{diag_kandidaten[0] / max(1, k - diag_prev_k):,.0f} "
                          f"zelle={diag_zellgroesse[0]:.4f}AE "
                          f"detkarten={det_aktiv[0]}/{len(det)}",
                          flush=True)
                diag_t_merge_det[0] = 0.0
                diag_t_bounce_det[0] = 0.0
                diag_t_bounce_host[0] = 0.0
                diag_kandidaten[0] = 0
                diag_prev_s = diag_now_s
                diag_prev_k = k
                diag_prev_ph = ph_val
                diag_throttled_s = 0.0
                diag_t_step = 0.0
                diag_t_analyse = 0.0
                diag_t_wait = 0.0
                diag_t_erkennung = 0.0
                diag_t_merges = 0.0
                diag_t_bounce = 0.0
                diag_t_ring = 0.0
            # 70% des Rings als Vorlauf, 30% bleiben Rueckspul-Historie —
            # sonst frisst die Eviction sich bis an den Playhead heran
            # und der Player reitet stotternd auf der Abbruchkante.
            if k + K - ph_abs >= int(capacity * 0.7):
                time.sleep(0.01)
                diag_throttled_s += 0.01
                continue
            _t = time.monotonic()
            outs = sim.step_batch(st, dt_years, K)
            diag_t_step += time.monotonic() - _t
            kh = st.get("hits")
            if kh is not None and len(kh[0]):
                for a_idx, t_rel in zip(kh[0], kh[1]):
                    kernel_kontakt[int(a_idx)] = k + float(t_rel) / dt_years
            # Erkennung laeuft UEBERLAPPT auf der Erkennungs-GPU, waehrend
            # die Physik-GPUs schon den naechsten Batch rechnen.
            future = executor.submit(analyze_batch_timed, outs, k)
            # x|y je Koerper plus die Stuetzpunkte der heissen Asteroiden
            # (siehe schreibe_slot). Masse/Sichtbarkeit laufen weiter als
            # Ereignis. outs[i] ist [x|y|vx|vy] (4n) — der v-Teil geht
            # NICHT in den Ring: er diente nur der Client-Interpolation,
            # und die stuetzt sich jetzt auf gemessene Zwischenbilder.
            _t = time.monotonic()
            subs = st.get("sub")
            for i in range(K):
                voll = schreibe_slot(
                    buf, ((k + i) % capacity) * sample_bytes, n, sub_max,
                    outs[i], subs[i] if subs else None)
                if voll:
                    sub_ueberlauf[0] += 1
                    if sub_ueberlauf[0] == 1:
                        print("[film] mehr heisse asteroiden als "
                              f"sub-plaetze ({len(subs[i][0])} > {sub_max})"
                              " — ueberzaehlige interpoliert der client "
                              "wie bisher", flush=True)
            diag_t_ring += time.monotonic() - _t
            k += K
            if ast_bounce and k % 500 == 0 and bounce_count:
                print(f"[film] {bounce_count} asti-bounces nach "
                      f"{k} samples", flush=True)
    finally:
        # Letzter Zustand fuer die Engine-Uebergabe, dann sauber schliessen.
        # Nach einem Zerbersten ist der SHATTER-Dump massgeblich — dann
        # weder ausstehende Analysen anwenden noch neu dumpen.
        try:
            # det_pool erst NACH dem Einsammeln schliessen: die ausstehende
            # Analyse verteilt ihre Streifen noch darueber.
            executor.shutdown(wait=False)
            if shatter_flag is not None and shatter_flag.value:
                # Shatter-Dump liegt bereits — nur als bereit markieren,
                # damit der FILM_STOP-Pfad des Servers nicht wartet.
                dump_req_val.value = 2
            else:
                if future is not None:
                    for res in future.result()[0]:
                        apply_analysis(res)
                dump_state()
                dump_req_val.value = 2
        except Exception:
            pass
        if det_pool is not None:
            det_pool.shutdown(wait=False)
        shm.close()
        ev_shm.close()
        dump_shm.close()
