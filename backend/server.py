"""WebSocket-Backend: CUDA-f64-Physik fuer den Sonnensystem-Simulator.

Start:  ./venv/bin/python backend/server.py [--port 8765] [--device N]

Server-authoritativer Zustand: Der Browser laedt den Vollzustand nur bei
Mutationen hoch (Kollisionen, Injects, Edits, Engine-Wechsel) — normale
Frames sind reine STEP-Nachrichten (16 Bytes), der Koerperzustand bleibt
als f64 GPU-resident. Die Antwort sind kompakte f32-Renderdaten. Damit
faellt der Upload pro Frame weg und der Download halbiert sich — wichtig
fuer Remote-Nutzung (WAN) und hohe Frameraten.

Kollisionen, Trails und UI bleiben im Browser; die HTML funktioniert
ohne Backend unveraendert (hardwareagnostisch).

Binaerprotokoll (Little-Endian):
  FULL:     u32 typ=0 | u32 N | f64 dtYears |
            x[N] f64 | y[N] | vx[N] | vy[N] | mass[N] |
            visible[N] u8 | isAst[N] u8
  STEP:     u32 typ=1 | u32 pad | f64 dtYears
  DELTA:    u32 typ=2 | u32 anzahl | f64 dtYears |
            anzahl × (u32 idx | u32 pad | f64 x | y | vx | vy)
  Response: u32 status=0 | u32 N | x[N] f32 | y[N] | vx[N] | vy[N]
            bei status!=0: stattdessen UTF-8-Fehlertext
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import multiprocessing as mp
import os
import socket
import struct
import threading
import time
from multiprocessing import shared_memory

import numpy as np
import websockets

import film_producer
from nbody_kernel import M_MAX, NBodyCuda, pick_device

log = logging.getLogger("solarsim-cuda")

# Wartezeit auf den f64-Dump des Producers in 10-ms-Schritten (5 s).
# Obergrenze ist die Trennfrist des Clients (CUDA_DISCONNECT_MS, 8 s):
# wird die Verbindung vorher gekappt, kommt der Dump nie an und der
# Client rechnet mit geschaetzten Impulsen weiter.
DUMP_WAIT_STEPS = 500

# Sekunden zwischen zwei Stream-Diagnosezeilen (Bandbreite, Ist-Kosten je
# Sample, Dichte). Zeigt, ob die Drosselung mit realistischen Groessen
# rechnet — bei stride > 1 lag die alte Schaetzung um genau diesen Faktor
# daneben.
STREAM_DIAG_INTERVAL_S = 2.0

# Mindest-Sendezeit, die in ein Bandbreiten-Messfenster einfliesst, bevor
# daraus ein Schaetzwert gebildet wird. Zu kurz gewaehlt misst man nur
# Puffer-Schreibvorgaenge (dur -> 0, Schaetzung explodiert), zu lang
# reagiert die Regelung traege auf Lastwechsel.
BW_FENSTER_S = 0.2

# Film-Protokollversion (Bits 4-7 des Flag-Bytes in MSG_FILM_START).
# 2 = viertes u8-Array `injiziert` fuer den LOD-Vorrang.
# 3 = Sample = 16 B/Koerper (x|y|vx|vy) + selektiver v-Block je Frame
#     (Hermite-Interpolation schneller sonnennaher Koerper).
# 4 = Sample = 8 B/Koerper (x|y) + Sub-Block je Frame: GEMESSENE
#     Zwischenbilder der heissen Asteroiden statt geschaetzter Tangenten.
FILM_PROTO_VERSION = 6

TAGE_PRO_JAHR = 365.25

# Log-Zoom-Basis, identisch zum Client (logTransform in index.html):
# r_log = log(1+r) / log(33). Im Log-Zoom werden die Positionen in DIESEM
# Raum quantisiert — die u16-Aufloesung ist dann im gestauchten Zentrum
# ~25x feiner (dort schaut der Nutzer hin), ohne mehr Bandbreite. Sonst
# waere die lineare u16 ueber die riesige Auto-Box (bis ~100 AE) im Zentrum
# grob und zeigte ein Schachbrettraster.
FILM_LOG33 = float(np.log(33.0))

HEADER = struct.Struct("<IId")   # typ, N/pad, dtYears
MSG_FULL = 0
MSG_STEP = 1
MSG_DELTA = 2
MSG_FILM_START = 3   # u32 typ | u32 N | f64 rasterTage | f64 t0Tage | FULL-Arrays
MSG_FILM_STOP = 4    # u32 typ | u32 amPlayhead | f64 playheadTage
MSG_FILM_SUB = 7     # u32 typ | u32 pad | f64 tTage | f64 rateTageProSek |
#                      f64 cx | f64 cy | f64 halbW | f64 halbH (Welt-AE;
#                      halbW<=0 = Auto-Box ueber alle Koerper) |
#                      f64 lodBudget (Punkte/Sample, 0 = Automatik)
#   Abo: Client meldet Playhead + Tempo (Start, Scrub, Tempo-Wechsel,
#   1-Hz-Heartbeat). Der Server STREAMT daraufhin kontinuierlich kleine
#   Frames (Push) und haelt den Client-Puffer ~5 s Playback voll —
#   TCP-Backpressure statt Anfrage-Roundtrips (Diashow-Ursache remote).
# Delta-Record: u32 idx | u32 pad | f64 x | f64 y | f64 vx | f64 vy
DELTA_REC = np.dtype([("idx", "<u4"), ("pad", "<u4"), ("v", "<f8", (4,))])

def _anzeige_koords(x, y, log_zoom: bool):
    """Welt- zu Anzeige-Koordinaten. Im Log-Zoom radial gestaucht
    (r_log = log(1+r)/log(33), identisch zum Client), sonst unveraendert.

    SSOT fuer Box, Culling, LOD und Quantisierung — die Sub-Stuetzpunkte
    MUESSEN dieselbe Transform durchlaufen wie die Sample-Positionen,
    sonst sitzt die Bahn eines Koerpers neben seinem eigenen Punkt."""
    if not log_zoom:
        return x, y
    r = np.hypot(x, y)
    sk = np.where(r > 1e-9,
                  np.log1p(r) / (FILM_LOG33 * np.maximum(r, 1e-30)), 0.0)
    return x * sk, y * sk


_SESSION_SEQ = [0]   # Round-Robin fuer die Erkennungskarten-Zuteilung


def _index_hash(idx: np.ndarray) -> np.ndarray:
    """Gleichverteilte Werte in [0,1) aus dem Original-Index.

    Der Mischer (SplitMix64-Finalisierer) sorgt dafuer, dass benachbarte
    Indizes voellig verschiedene Werte bekommen — ohne das legte eine
    Auswahl "jeder n-te" in gleichmaessig erzeugten Wolken ein sichtbares
    Raster an. Rein deterministisch: derselbe Koerper bekommt in jedem
    Sample denselben Wert und bleibt dadurch stabil gestreamt."""
    h = idx.astype(np.uint64)
    h ^= h >> np.uint64(30)
    h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(27)
    h *= np.uint64(0x94D049BB133111EB)
    h ^= h >> np.uint64(31)
    return (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def _shm_budget(anteil: float = 0.4) -> int:
    """Wie viel /dev/shm ein einzelner Film-Ring belegen darf.

    Gemessen am FREIEN Platz, nicht an der Gesamtgroesse: Laeuft schon
    eine Sitzung, bekommt die naechste entsprechend weniger, statt den
    tmpfs zu sprengen und den Producer beim Start scheitern zu lassen.
    """
    st = os.statvfs("/dev/shm")
    return int(st.f_bavail * st.f_frsize * anteil)


class FilmSession:
    """Proxy auf den Producer-PROZESS (film_producer.py): eigener Python-
    Prozess besitzt die GPU und schreibt in einen Shared-Memory-Ring —
    kein GIL-Sharing. Der Server liest Batches in Mikrosekunden direkt
    aus dem Ring, waehrend die GPU mit vollem Durchsatz rechnet und nur
    pausiert, wenn der Ring voll ist (Ueberschreib-Schutz vor dem
    Player-Playhead)."""

    # Ringpuffer-Obergrenze. Bestimmt, wie viel VERGANGENHEIT navigierbar
    # ist: capacity = MAX_BYTES / (8 * n) Slots, mal raster_days ergibt die
    # Historie in Sim-Tagen. Bei grossem n wird das schnell knapp — 267k
    # Koerper ergeben mit 4 GiB nur ~1000 Sim-Tage, die zudem staendig
    # wegwandern (der Tail folgt dem Producer). Ueber --ring-gib
    # einstellbar; Obergrenze ist der freie Platz in /dev/shm, NICHT der
    # freie RAM (shared_memory liegt im tmpfs).
    MAX_BYTES = 8 << 30

    # Zweite, ZEITLICHE Grenze des Rings (--ring-jahre). Der Ring traegt
    # hoechstens so viel Sim-Zeit, unabhaengig davon, wie klein die
    # Samples sind.
    #
    # 300 Jahre sind an der WIEDERGABEZEIT bemessen, nicht an der
    # Sim-Zeit: Bei 186 Tagen/s Abspieltempo ergibt der 70-%-Vorlauf rund
    # SIEBEN MINUTEN Puffer — genug, um Netz- oder Rechenaussetzer zu
    # ueberbruecken und bequem zurueckzuspulen. Der Speicher sinkt
    # trotzdem: 11.220 Koerper bei Raster 1,78 belegen 5,2 statt 8 GiB,
    # und bei noch kleineren Szenen entsprechend weniger. Bei grosser
    # Koerperzahl bleibt ohnehin das Bytebudget die bindende Grenze.
    MAX_RING_TAGE = 300 * 365.25

    # Erkennungskarten pro Session (--det-gpus). Die Bounce-Suche ist der
    # Engpass und wird raeumlich auf sie aufgeteilt; mehr Karten helfen
    # nur, solange welche frei sind (Physik hat Vorrang).
    DET_GPUS = film_producer.DET_MAX

    # Zeitanteils-Diagnose in Producer und Stream-Loop (--diag).
    # Im Normalbetrieb aus: die Zeilen sind fuer die Engpass-Suche
    # gebaut, nicht fuer den Alltag.
    DIAG = False

    def __init__(self, t0_days: float, raster_days: float,
                 x, y, vx, vy, mass, real_r, visible, is_ast,
                 is_star_bh, injiziert, ast_bounce: bool, m_sub: int,
                 softening_au: float = 0.0):
        self.raster_days = max(0.1, raster_days)
        self.t0 = t0_days
        self.n = len(x)
        # Slot = x|y f32 + Sub-Block (Stuetzpunkte der heissen Asteroiden,
        # 0 = aus). Die Geschwindigkeit liegt NICHT mehr im Ring: sie
        # diente allein der Client-Interpolation, und dafuer sind die
        # Stuetzpunkte da. Masse/Sichtbarkeit laufen als Ereignisse im
        # Event-Ring. m_sub kommt vom Client-Regler und bestimmt die
        # Slotgroesse — ein Wechsel braucht eine neue Session.
        self.m_sub = min(film_producer.M_SUB_MAX, max(0, int(m_sub)))
        self.sub_max = film_producer.sub_max_fuer(self.n) if self.m_sub \
            else 0
        self.sample_bytes = film_producer.slot_bytes(
            self.n, self.m_sub, self.sub_max)
        # Ringgroesse: das KLEINERE aus Byte- und Zeitbudget.
        #
        # Nur nach Bytes zu gehen liefert je nach Koerperzahl voellig
        # verschiedene Vorlaeufe — bei 8 GiB und Raster 1,78 Tage:
        #     11.220 Koerper ->  88 KB/Sample ->  91,7 Jahre Vorlauf
        #    400.000 Koerper -> 3,1 MB/Sample ->   2,6 Jahre
        # Faktor 35 fuer denselben Speicher. Bei kleinem N ist der Ring
        # sinnlos gross: Der Producer fuellt ihn in Sekunden, drosselt
        # sich und steht dann still (die GPU wirkt "idle", obwohl alles in
        # Ordnung ist), waehrend 8 GiB in /dev/shm liegen. Zwei
        # gleichzeitige Sitzungen sprengen damit ein 16-GB-tmpfs — genau
        # das hat im Betrieb einen Producer am Start scheitern lassen.
        #
        # Das Zeitbudget deckelt den Vorlauf auf ein Mass, das zum
        # Zurueckspulen reicht; bei grossem N bleibt das Bytebudget die
        # bindende Grenze und nichts aendert sich.
        # Zusaetzlich an den TATSAECHLICH freien Platz koppeln. Ein
        # fester Wert (8 GiB) passt sich der Maschine nicht an: Auf
        # einem 16-GB-tmpfs belegt er die Haelfte, und die zweite
        # Sitzung — beim Neuladen ueberlappen alte und neue kurz —
        # findet keinen Platz mehr. 40 % lassen zwei nebeneinander
        # zu und behalten Luft fuer den Ereignisring.
        nach_bytes = int(min(self.MAX_BYTES, _shm_budget())
                         // self.sample_bytes)
        nach_zeit = int(self.MAX_RING_TAGE / self.raster_days)
        # nach_bytes ist die HARTE Grenze — der physisch im tmpfs verfuegbare
        # Platz. Der Mindest-Vorlauf von 2000 Slots gilt nur gegen das
        # ZEIT-Budget und darf den Platz NIE ueberschreiten: bei Millionen
        # Koerpern ist ein Slot viele MB, 2000 davon sprengen ein 16-GB-tmpfs,
        # und der Producer stirbt beim Schreiben ueber die Mapping-Grenze
        # hinaus mit SIGBUS (Signal 7). Darum min() aussen, max() nur innen.
        self.capacity = min(nach_bytes, max(2000, nach_zeit))
        self.shm = shared_memory.SharedMemory(
            create=True, size=self.capacity * self.sample_bytes)
        # Ereignisring. Muss den RUECKSTAND puffern koennen: erzeugt der
        # Producer mehr Kollisionen, als der Stream abtransportiert,
        # waechst er, und beim Ueberlauf gehen Ereignisse verloren. Bei
        # einer in den Stern stuerzenden Wolke sind sechsstellige Zahlen
        # in Sekunden erreicht — 65536 Plaetze (2 MB) waren dafuer zu
        # knapp. 262144 sind 8 MB und puffern das Vierfache.
        self.ev_cap = 262144
        self.ev_shm = shared_memory.SharedMemory(
            create=True, size=self.ev_cap * film_producer.EV_BYTES)
        # Zustands-Dump fuer die Engine-Uebergabe (x,y,vx,vy als f64)
        self.dump_shm = shared_memory.SharedMemory(
            create=True, size=4 * 8 * self.n)
        ctx = mp.get_context("spawn")
        self.head_val = ctx.Value("q", 0, lock=False)
        self.playhead_val = ctx.Value("d", t0_days, lock=False)
        self.coll_val = ctx.Value("q", 0, lock=False)
        self.ev_count_val = ctx.Value("q", 0, lock=False)
        self.dump_req_val = ctx.Value("b", 0, lock=False)
        self.running_val = ctx.Value("b", 1, lock=False)
        # Zerbersten (Koerper x Koerper, vImp >= 1,5 vEsc): der Producer
        # erkennt nur, dumpt seinen f64-Zustand und stoppt sich selbst —
        # die Shatter-PHYSIK fuehrt der Client mit seinem shatter() aus
        # (SSOT) und startet den Film mit den Fragmenten neu.
        self.shatter_flag = ctx.Value("b", 0, lock=False)
        self.shatter_a = ctx.Value("q", 0, lock=False)
        self.shatter_b = ctx.Value("q", 0, lock=False)
        self.shatter_t = ctx.Value("d", 0.0, lock=False)
        # Kill-Zeitpunkte (aus dem Event-Strom gepflegt): ein Sample
        # cullt nur Koerper, die zu SEINER Sim-Zeit schon tot sind.
        # Ein sofortiger bool-Spiegel liess Opfer aus dem Stream
        # verschwinden, lange bevor der Playhead den Kollisionszeitpunkt
        # erreichte — ihre Client-Position fror unterwegs ein und die
        # Explosion blitzte spaeter irgendwo im Leeren.
        self._kill_t = np.where(
            np.array(visible, dtype=np.uint8) != 0, np.inf, -np.inf)
        self._is_ast = np.array(is_ast, dtype=np.uint8, copy=True) != 0
        # Vorrang beim LOD-Budget: Koerper des geladenen Systems zuerst,
        # nachtraeglich injizierte Wolken bekommen den Rest.
        self._injiziert = np.array(injiziert, dtype=np.uint8, copy=True) != 0
        # Punktbudget pro Sample; 0 = automatisch aus der gemessenen
        # Bandbreite. Kommt live im Abo (MSG_FILM_SUB), wirkt daher ohne
        # Filmneustart.
        self.lod_budget = 0
        # cx, cy, halbW, halbH des Client-Sichtfensters. halbW <= 0 heisst
        # "noch nichts gemeldet" — dann Auto-Box ueber alle Koerper.
        self.view = (0.0, 0.0, -1.0, -1.0)
        # Positionen log-polar kodieren (Anzeigeart des Clients). Getrennt
        # von der Box-Wahl: der Log-Zoom bestimmt den KOORDINATENRAUM, das
        # Sichtfenster den Ausschnitt darin.
        self.log_zoom = False
        state = {k: np.array(v, copy=True) for k, v in
                 (("x", x), ("y", y), ("vx", vx), ("vy", vy),
                  ("mass", mass), ("realR", real_r),
                  ("visible", visible), ("isAst", is_ast),
                  ("isStarBH", is_star_bh))}
        self.proc = ctx.Process(
            target=film_producer.producer_main,
            args=(self.shm.name, self.sample_bytes, self.capacity,
                  self.ev_shm.name, self.ev_cap, self.ev_count_val,
                  self.dump_shm.name, self.dump_req_val,
                  self.head_val, self.playhead_val, self.coll_val,
                  self.running_val, state, self.raster_days, t0_days,
                  bool(ast_bounce),
                  self.shatter_flag, self.shatter_a, self.shatter_b,
                  self.shatter_t, _SESSION_SEQ[0], self.DET_GPUS,
                  self.DIAG, self.m_sub, self.sub_max,
                  float(softening_au)),
            daemon=True)
        _SESSION_SEQ[0] += 1
        self.proc.start()

    @property
    def head(self) -> float:
        return self.t0 + self.head_val.value * self.raster_days

    @property
    def tail_abs(self) -> int:
        # Sicherheitsmarge gegen Lese-/Schreib-Ueberlappung am Ringende
        return max(0, self.head_val.value - self.capacity + 16)

    @property
    def tail(self) -> float:
        return self.t0 + self.tail_abs * self.raster_days

    @property
    def running(self) -> bool:
        return self.proc.is_alive()

    @property
    def samples(self):
        # Kompatibilitaet zur Warte-Schleife im Handler ("Puffer leer?")
        return self.head_val.value

    def slot_pos(self, i: int) -> np.ndarray:
        """x|y des absoluten Samples i als f32-View auf den Ring (2n)."""
        return np.frombuffer(self.shm.buf, "<f4", 2 * self.n,
                             (i % self.capacity) * self.sample_bytes)

    def slot_sub(self, i: int):
        """Sub-Block des Samples i: (idx, bahnen) mit idx = Original-
        indizes (aufsteigend) und bahnen = (nh, mSub, 2) f32. Leer, wenn
        das Sample keine heissen Koerper hatte."""
        if not self.m_sub:
            return np.empty(0, np.uint32), np.empty((0, 0, 2), np.float32)
        basis = (i % self.capacity) * self.sample_bytes + 8 * self.n
        nh = int(np.frombuffer(self.shm.buf, "<u4", 1, basis)[0])
        nh = min(nh, self.sub_max)
        idx = np.frombuffer(self.shm.buf, "<u4", nh, basis + 4)
        bahnen = np.frombuffer(
            self.shm.buf, "<f4", nh * self.m_sub * 2,
            basis + 4 + 4 * self.sub_max).reshape(nh, self.m_sub, 2)
        return idx, bahnen

    # ---- Streaming (Push) ----
    sub_rate = 60.0          # Tage/s Playback-Tempo des Clients
    stream_task = None
    sent_abs = None          # aktuelle Stream-Position (absoluter Sample-Index)
    _bw = 4e6                # gemessene Leitungs-Bandbreite (Bytes/s, EWMA)
    # GEMESSENE Bytes pro gestreamtem Sample (EWMA). Die Drosselung darf
    # NICHT mit sample_bytes (der volle Ring-Slot) rechnen: ein Frame
    # enthaelt nur die gecullten und per Dichte-LOD ausgeduennten
    # Koerper. Bei grossem N liegt sample_bytes um den LOD-Faktor
    # daneben (bei einem Neuntel also 9x zu hoch) — der Server haelt seine
    # Frames dann fuer viel teurer als sie sind, liefert entsprechend
    # weniger Samples und der Client stockt, obwohl die Leitung leer ist.
    # None = noch nichts gemessen, dann konservativ sample_bytes.
    _sample_cost = None
    # Zuletzt gesendete Indexliste (v6): Solange sie gleich
    # bleibt, traegt der Frame sie nicht mit. Pro Session, weil
    # der Bezug ueber Frame-Grenzen laeuft — ein neuer FILM_START
    # erzeugt eine neue Session und damit einen sauberen Anfang.
    _letzte_sel = None
    _diag_s = 0.0            # monotonic() der letzten Stream-Diagnosezeile
    _bw_bytes = 0.0          # Bytes im laufenden Bandbreiten-Messfenster
    _bw_dur = 0.0            # aufsummierte reine Sendezeit im Fenster
    ph_ms = 0.0              # monotonic() des letzten Playhead-Heartbeats
    # Der Client meldet den Playhead nur im 1-Hz-Heartbeat. Ohne
    # Extrapolation rechnet der Stream bis zu eine Sekunde lang mit einem
    # veralteten Playhead, haelt den Client-Puffer faelschlich fuer voll
    # und liefert dadurch stossweise statt gleichmaessig. Deckel, damit
    # ein pausierter Client den geschaetzten Playhead nicht davonlaufen
    # laesst (dann lieber zu wenig als zu viel senden).
    PH_EXTRAPOLATE_MAX_S = 2.0

    sent_ev = 0              # bereits gestreamte Ereignisse
    ev_verworfen = 0         # beim Ringueberlauf uebersprungene (s. build_frame)

    def build_frame(self, idxs: list) -> bytes:
        """v4-Frame: pro Sample nur die Koerper in der Referenz-Box
        (Culling!), Koordinaten als u16 relativ zur Box (User-Design:
        Integer-Streaming). Der Client dekodiert zurueck in Welt-
        koordinaten — die Kamera bleibt frei, die Box bestimmt nur
        Culling und Praezision."""
        # Neue Ereignisse einsammeln + lokalen vis-Spiegel pflegen
        ev_total = int(self.ev_count_val.value)
        ev_from = max(self.sent_ev, ev_total - self.ev_cap + 8)
        # Uebersprungene Ereignisse: der Ring ist uebergelaufen, diese
        # Kollisionen sind unwiederbringlich weg. Sie duerfen aber nicht
        # aus dem ZAEHLER verschwinden — der Client bekommt ihre Anzahl
        # und rechnet sie hinzu, auch wenn ihr Blitz fehlt.
        verworfen = max(0, ev_from - self.sent_ev)
        self.ev_verworfen += verworfen
        # 4096/Frame: Bounce-Stroeme (kind=1) erzeugen deutlich mehr
        # Ereignisse als Merges. Der Rueckstand holt ueber Folgeframes
        # auf — je hoeher diese Schranke, desto seltener laeuft der Ring
        # ueber. 4096 x 32 B = 128 KB je Frame.
        ev_n = min(4096, ev_total - ev_from)
        eb = film_producer.EV_BYTES
        ev_parts = []
        for e in range(ev_from, ev_from + ev_n):
            raw = bytes(self.ev_shm.buf[(e % self.ev_cap) * eb:
                                        (e % self.ev_cap) * eb + eb])
            ev_parts.append(raw)
            (t_ev,) = struct.unpack_from("<d", raw, 0)
            b_idx, _m, kind = struct.unpack_from("<IfI", raw, 12)
            if kind == 0 and b_idx < self.n:
                self._kill_t[b_idx] = min(self._kill_t[b_idx], t_ev)
        self.sent_ev = ev_from + ev_n

        cx, cy, hw, hh = self.view
        m_sub = self.m_sub
        times = np.asarray(
            [self.t0 + (i + 1) * self.raster_days for i in idxs], "<f8")
        blocks = []
        box = None
        log_zoom = self.log_zoom
        # Auto-Box nur noch, solange der Client sein Sichtfenster nicht
        # gemeldet hat (vor dem ersten Abo). Sie ueber ALLE Koerper zu
        # spannen kostet Aufloesung: die u16-Schrittweite ist die
        # Box-Breite / 65535, ein weit ausgeworfener Koerper verdirbt sie
        # also fuer das Zentrum, wo der Nutzer hinschaut.
        auto_box = hw <= 0 or hh <= 0
        for nr, i in enumerate(idxs):
            pos = self.slot_pos(i)
            t_i = self.t0 + (i + 1) * self.raster_days
            # Ein Raster ueber den Tod hinaus mitstreamen. Der Kill-
            # Zeitpunkt liegt seit der Streckenpruefung MITTEN im Raster;
            # ohne das Sample dahinter fehlt dem Client das Ziel, er
            # bliebe auf der letzten Position stehen und der Koerper
            # verschwaende sichtbar VOR dem Stern statt in ihm. Der
            # Producer wendet den Merge erst ein Sample spaeter an, die
            # Position ist hier also noch die echte Bahn.
            alive_i = self._kill_t > t_i - self.raster_days
            x = pos[0:self.n]
            y = pos[self.n:2 * self.n]
            # Anzeige-Koordinaten fuer Box/Culling/LOD/Quantisierung: im
            # Log-Zoom (hw<=0, Auto-Box) der radial gestauchte Log-Raum,
            # sonst die Weltkoordinaten. Der Client dekodiert sie per
            # inverser Log-Transform zurueck.
            qxs, qys = _anzeige_koords(x, y, log_zoom)
            if box is None:
                if auto_box:
                    # Noch kein Sichtfenster gemeldet: gesamte Verteilung
                    # (+5% Rand). Nur Startwert bis zum ersten Abo.
                    bx0, bx1 = float(qxs.min()), float(qxs.max())
                    by0, by1 = float(qys.min()), float(qys.max())
                    mx = 0.05 * max(bx1 - bx0, 1e-6)
                    my = 0.05 * max(by1 - by0, 1e-6)
                    box = (bx0 - mx, by0 - my,
                           max(bx1 - bx0 + 2 * mx, 1e-6),
                           max(by1 - by0 + 2 * my, 1e-6))
                else:
                    # Referenz-Box = 2x Viewport um die Kamera. Im
                    # Log-Zoom sind cx/cy/hw/hh bereits Log-Koordinaten
                    # (der Client rechnet in diesem Raum), die Box passt
                    # also unveraendert.
                    box = (cx - 2 * hw, cy - 2 * hh,
                           max(4 * hw, 1e-6), max(4 * hh, 1e-6))
            x0, y0, spanx, spany = box
            sel = np.flatnonzero(
                alive_i & (qxs >= x0) & (qxs <= x0 + spanx)
                & (qys >= y0) & (qys <= y0 + spany))
            # Dichte-LOD: mehr sichtbare Punkte, als Bandbreite und
            # Client-Dekodierung bei 20 Samples/s verkraften.
            # Obergrenze 120k: mehr Punkte pro Sample schafft der
            # Client-Dekodier-/Interpolations-Loop nicht bei 60 FPS.
            # lod_budget != 0 = vom Nutzer gesetzt (Regler), sonst Auto.
            lod_max = self.lod_budget or min(
                120_000, max(20000, int(self._bw * 0.7 / 20.0 / 8.0)))
            # LOD + Quantisierung laufen in Anzeige-Koordinaten (qxs/qys):
            # so ist die Dichte-Ausduennung gleichmaessig ueber das ANGE-
            # ZEIGTE Bild und die u16-Aufloesung passt zur Anzeige.
            sel = self._lod_auswahl(sel, qxs, qys, box, lod_max)
            qx = np.clip((qxs[sel] - x0) / spanx * 65535.0,
                         0, 65535).astype("<u2")
            qy = np.clip((qys[sel] - y0) / spany * 65535.0,
                         0, 65535).astype("<u2")
            # Sub-Block: die GEMESSENEN Zwischenbilder der heissen
            # Asteroiden, beschraenkt auf die tatsaechlich gestreamten
            # Koerper. Fuer sie interpoliert der Client linear entlang der
            # Stuetzpunkte statt eine Tangente zu schaetzen — genau die
            # Koerper, bei denen die Sehne sonst den Bogen abschneidet.
            #
            # Das ERSTE Sample eines Frames bekommt keinen: die Kette
            # beginnt beim Vorgaenger-Sample, und der liegt im vorigen
            # Frame — nach einem Sprung womoeglich ganz woanders. So ist
            # der Block immer sicher verankert und der Client braucht
            # keine Zusatzpruefung.
            sblock = struct.pack("<I", 0)
            if m_sub and nr > 0:
                ssel, bahn = self._sub_kette(idxs[nr - 1], i, sel)
                if ssel is not None:
                    bqx, bqy = _anzeige_koords(
                        bahn[:, :, 0], bahn[:, :, 1], log_zoom)
                    sblock = struct.pack("<I", len(ssel)) + \
                        ssel.astype("<u4").tobytes() + \
                        np.clip((bqx - x0) / spanx * 65535.0, 0, 65535
                                ).astype("<u2").tobytes() + \
                        np.clip((bqy - y0) / spany * 65535.0, 0, 65535
                                ).astype("<u2").tobytes()
            # Die Indexliste ist die HAELFTE der Nutzlast (4 von 8 Byte je
            # Punkt), aendert sich zwischen zwei Samples aber meist gar
            # nicht: Ohne Kamerabewegung und ohne LOD-Ausduennung ist sie
            # schlicht konstant. Also nur bei Aenderung senden und sonst
            # Bit 31 der Laenge setzen — der Client behaelt die letzte.
            # Das halbiert den Strom (gemessen: 9,8 -> 19,5 Samples/s auf
            # einer 3,45-MB/s-Leitung, UEBERGABE 6.14).
            #
            # Der Bezug laeuft ueber FRAME-GRENZEN hinweg, anders als beim
            # Sub-Block: Remote passt oft nur EIN Sample in einen Frame
            # (Budget 512 KB gegen 354 KB Kosten), eine frameweise
            # Verankerung spraenge also nie an. Tragfaehig ist das, weil
            # der Client jedes Frame dekodiert — er verwirft keine.
            sel_u4 = sel.astype("<u4")
            if self._letzte_sel is not None and \
                    len(sel_u4) == len(self._letzte_sel) and \
                    np.array_equal(sel_u4, self._letzte_sel):
                kopf = struct.pack("<I", len(sel) | 0x8000_0000)
                sel_bytes = b""
            else:
                self._letzte_sel = sel_u4
                kopf = struct.pack("<I", len(sel))
                sel_bytes = sel_u4.tobytes()
            block = kopf + sel_bytes + qx.tobytes() + qy.tobytes() + sblock
            block += b"\x00" * ((-len(block)) % 4)
            blocks.append(block)

        head = struct.pack("<IIII", 4, self.n, len(idxs), ev_n)
        # 7. f64 = logFlag: 1 = Positionen sind log-polar kodiert (der
        # Client transformiert zurueck), 0 = Weltkoordinaten.
        # 8. f64 = mSub: Stuetzpunkte je Koerper im Sub-Block, 0 = keiner.
        # 9. f64 = insgesamt verworfene Ereignisse. Der Client zaehlt sie
        # mit, sonst faelle jede Kollision, deren Ereignis den Ringueberlauf
        # nicht ueberlebt hat, still aus dem Zaehler.
        meta = struct.pack("<ddddddddd", self.tail, self.head,
                           box[0], box[1], box[2], box[3],
                           1.0 if log_zoom else 0.0, float(m_sub),
                           float(self.ev_verworfen))
        return head + meta + times.tobytes() + \
            b"".join(blocks) + b"".join(ev_parts)

    def _sub_kette(self, vor: int, i: int, sel):
        """mSub Stuetzpunkte ueber das gestreamte Intervall (t_vor, t_i].

        Der Kernel legt seine Zwischenbilder je RASTER ab. Der Server
        streamt aber nur selten raster-dicht: bei 60 Tage/s Abspieltempo
        liegen 3 Sim-Tage zwischen zwei Samples (step 6), bei hohem Tempo
        bis zu 50. Nur das letzte Raster mit Stuetzpunkten zu belegen
        haette dem Client nichts genuetzt — er haette den Rest weiter mit
        Sehne/Catmull ueberbrueckt, und genau dort schneidet die Sehne den
        Bogen ab (die Koerper rutschen sonnenwaerts, vor der Sonne bildet
        sich eine Luecke).

        Deshalb wird die Kette ueber das GANZE Intervall gespannt: aus den
        `step * mSub` Stuetzpunkten der ueberdeckten Ring-Slots werden
        mSub gleichabstaendige gezogen, der letzte ist die Sample-Position.
        Die Bandbreite je heissem Koerper bleibt damit konstant, egal wie
        schnell abgespielt wird — nur die Stuetzpunkte werden gruober.

        Angefasst werden nur die Slots, aus denen wirklich ein Punkt
        stammt (hoechstens mSub Stueck, nicht step). Ein Koerper muss in
        genau diesen heiss gewesen sein; war er zwischendurch kuehl, ist
        seine Bahn dort ohnehin fast gerade.

        Rueckgabe (idx, bahnen) mit bahnen (nh, mSub, 2), oder
        (None, None), wenn kein Koerper durchgehend Stuetzpunkte hat.
        """
        ms = self.m_sub
        schritt = i - vor
        if schritt <= 0:
            return None, None
        # Kettenposition der mSub Ausgabepunkte, aequidistant und mit dem
        # letzten genau auf dem Sample. Bei schritt == 1 sind das exakt
        # die Stuetzpunkte des Rasters.
        p = (np.arange(1, ms + 1) * (schritt * ms)) // ms - 1
        slot_nr = vor + 1 + p // ms
        punkt_nr = p % ms
        daten = {}
        gemeinsam = sel
        for s in np.unique(slot_nr):
            sidx, bahnen = self.slot_sub(int(s))
            if len(sidx) == 0:
                return None, None
            daten[int(s)] = (sidx, bahnen)
            stelle = np.searchsorted(sidx, gemeinsam)
            stelle = np.minimum(stelle, len(sidx) - 1)
            gemeinsam = gemeinsam[sidx[stelle] == gemeinsam]
            if len(gemeinsam) == 0:
                return None, None
        aus = np.empty((len(gemeinsam), ms, 2), np.float32)
        for j in range(ms):
            sidx, bahnen = daten[int(slot_nr[j])]
            aus[:, j, :] = bahnen[np.searchsorted(sidx, gemeinsam),
                                  punkt_nr[j], :]
        return gemeinsam, aus

    # Zellen je Achse fuer die Dichteschaetzung. Die Auto-Box umspannt
    # ALLE Koerper — auch weit hinausgeschleuderte —, der sichtbare
    # Ausschnitt ist davon oft nur ein Bruchteil. Ein grobes Gitter legt
    # dort entsprechend grosse Zellen an, deren Dichtesprung man als
    # Rechteck sieht. 512 statt 128 viertelt die Kantenlaenge; die Kosten
    # bleiben klein (bincount ueber 512x512 Zellen, unabhaengig von n).
    # Ab so vielen massiven Koerpern gelten sie als PUNKTWOLKE und
    # werden bei knappem Budget mitgeduennt, statt es allein
    # aufzubrauchen. Unterhalb bleibt der harte Vorrang: Ein paar
    # benannte Koerper muessen immer sichtbar sein.
    LOD_MASSEN_WOLKE_AB = 200
    # Anteil des Budgets, den die Massen dann hoechstens bekommen. Der
    # Rest gehoert den Tracern — sie sind es, die die Struktur zeichnen.
    LOD_MASSEN_ANTEIL = 0.5

    LOD_ZELLEN = 512
    # Dichte-Kontrast. behalten ~ anzahl^GAMMA je Zelle:
    #   1,0 = wie frueher (dichte Zellen behalten proportional alles,
    #         duenne Strukturen fallen unter die Sichtbarkeitsschwelle)
    #   0,0 = alle Zellen gleich viele Punkte (Dichteunterschiede
    #         verschwinden voellig — die Szene sieht ueberall gleich aus)
    # 0,5 (Wurzel) laesst dichte Gebiete deutlich dichter erscheinen und
    # haelt duenne trotzdem sichtbar: eine Zelle mit 10.000 Koerpern zeigt
    # gegenueber einer mit 100 noch das 10-fache statt des 100-fachen.
    LOD_GAMMA = 0.5

    def _lod_auswahl(self, sel, x, y, box, budget):
        """Punkte fuers Sample auswaehlen, wenn `sel` das Budget sprengt.

        Rangfolge:
          1. Massive Koerper (Sonne, Planeten, Rogues, Sterne, SL) —
             immer vollstaendig. Es sind wenige, und ohne sie ist die
             Szene unlesbar.
          2. Asteroiden des geladenen Systems (Guertel, Szenario).
          3. Nachtraeglich injizierte Wolken — bekommen den Rest.

        Innerhalb einer Stufe wird DICHTEABHAENGIG geduennt (siehe
        LOD_GAMMA), nicht gleichmaessig: eine feste Rate ueber alle
        Koerper loescht duenne Strukturen (der Guertel hat nur ein paar
        hundert Objekte auf riesigem Ring), waehrend kompakte Wolken auch
        stark geduennt noch dicht wirken.

        Sprengt schon Stufe 2 das Budget, wird auch dort geduennt — das
        ist die normale Wirkung der Rangfolge unter knappem Budget, kein
        Sonderfall (Szenarien mit sehr vielen vorbelegten Koerpern)."""
        if len(sel) <= budget:
            return sel
        ast = self._is_ast[sel]
        massiv = sel[~ast]
        rest = budget - len(massiv)
        if rest <= 0:
            # Die massiven Koerper sprengen das Budget allein. Im
            # klassischen Fall ist das eine Handvoll benannter Koerper,
            # die man sehen MUSS — dann gilt der Vorrang ohne Wenn und
            # Aber, und das Budget wird bewusst ueberschritten.
            #
            # Im selbstgravitierenden Fall sind es zehntausende Punkte
            # einer Wolke, und dann kippt die Regel ins Gegenteil: Alle
            # Massen ungedueennt zu schicken heisst, dass fuer die Tracer
            # NICHTS bleibt — nicht wenig, sondern null. Beobachtet bei
            # 31.623 Massen und 20.000 Budget: kein einziger von 44.668
            # Tracern kam durch, das Bild wurde schlagartig duenner.
            # Hier werden die Massen darum selbst geduennt, und die
            # Tracer bekommen einen garantierten Anteil.
            if len(massiv) > self.LOD_MASSEN_WOLKE_AB and len(sel) > len(massiv):
                m_budget = max(1, int(budget * self.LOD_MASSEN_ANTEIL))
                teile = [self._dichte_filter(massiv, x, y, box, m_budget)]
                rest = budget - len(teile[0])
                a_sel = sel[ast]
                if rest > 0 and len(a_sel):
                    teile.append(
                        self._dichte_filter(a_sel, x, y, box, rest))
                return np.sort(np.concatenate(teile))
            return sel[~ast]
        a_sel = sel[ast]
        inj = self._injiziert[a_sel]
        teile = [massiv]
        for kandidaten in (a_sel[~inj], a_sel[inj]):
            if rest <= 0 or not len(kandidaten):
                continue
            behalten = self._dichte_filter(kandidaten, x, y, box, rest)
            teile.append(behalten)
            rest -= len(behalten)
        return np.sort(np.concatenate(teile))

    def _dichte_filter(self, idx, x, y, box, budget):
        """Aus `idx` hoechstens `budget` Indizes waehlen, dichte Gebiete
        staerker duennend als duenne (behalten ~ anzahl^LOD_GAMMA).

        Die Auswahl laeuft je Zelle ueber den ORIGINAL-Index
        (`idx % schritt == 0`) und ist damit ueber Samples hinweg stabil:
        dieselben Koerper bleiben gestreamt, die Client-Interpolation
        reisst nicht."""
        if len(idx) <= budget:
            return idx
        x0, y0, spanx, spany = box
        k = self.LOD_ZELLEN
        cx = np.clip(((x[idx] - x0) / spanx * k).astype(np.int32), 0, k - 1)
        cy = np.clip(((y[idx] - y0) / spany * k).astype(np.int32), 0, k - 1)
        zelle = cy * k + cx
        anzahl = np.bincount(zelle, minlength=k * k)
        belegt = anzahl > 0
        gewicht = np.zeros(k * k)
        gewicht[belegt] = anzahl[belegt] ** self.LOD_GAMMA
        # Skalierung so waehlen, dass die Summe der behaltenen Punkte das
        # Budget trifft. min(anzahl, s*gewicht) ist monoton in s, also per
        # Bisektion loesbar — geschlossen ginge es nur ohne die Deckelung
        # auf die tatsaechliche Zellbelegung.
        #
        # Das Budget ist damit ein ZIELWERT, keine harte Schranke: die
        # Hash-Auswahl trifft die Zellvorgabe im Erwartungswert, die
        # tatsaechliche Zahl streut um rund sqrt(budget) (bei 20.000 also
        # etwa 140 Punkte, 0,7%). Ein exakter Deckel braeuchte eine
        # Teilsortierung der Trefferliste — Aufwand ohne Wirkung, denn
        # das Budget selbst stammt aus einer Bandbreitenschaetzung.
        lo, hi = 0.0, float(budget)
        for _ in range(40):
            s = 0.5 * (lo + hi)
            if np.minimum(anzahl, s * gewicht).sum() > budget:
                hi = s
            else:
                lo = s
        ziel = np.minimum(anzahl, lo * gewicht)
        # Behalte-Rate je Zelle, STUFENLOS. Ein ganzzahliger Schritt je
        # Zelle (jeder n-te Index) kann nur die Raten 1, 1/2, 1/3 ...
        # treffen; zwei Nachbarzellen landen dann auf 1/3 und 1/4 und
        # unterscheiden sich sichtbar um ein Drittel — mit harter Kante
        # entlang der Zellgrenze. Genau das erzeugte rechteckige
        # Block-Artefakte im Bild.
        rate = np.zeros(k * k).reshape(k, k)   # [cy, cx]
        np.divide(ziel, anzahl, out=rate.reshape(-1), where=belegt)
        # Behalte-Rate BILINEAR zwischen den Zellzentren interpolieren
        # statt pro Zelle konstant: sonst springt die Rate an jeder
        # Zellgrenze hart, und bei nahem Zoom (wenige grosse Zellen im
        # Bild) sieht man das als Schachbrettraster. Der Punkt liegt bei
        # kontinuierlichen Zellkoordinaten; seine Rate ist der bilineare
        # Mix der vier umgebenden Zellzentren (Zentrum bei i+0.5).
        fx = np.clip((x[idx] - x0) / spanx * k - 0.5, 0, k - 1)
        fy = np.clip((y[idx] - y0) / spany * k - 0.5, 0, k - 1)
        ix = np.floor(fx).astype(np.int32)
        iy = np.floor(fy).astype(np.int32)
        ix1 = np.minimum(ix + 1, k - 1)
        iy1 = np.minimum(iy + 1, k - 1)
        tx = fx - ix
        ty = fy - iy
        r_p = (rate[iy, ix] * (1 - tx) * (1 - ty) +
               rate[iy, ix1] * tx * (1 - ty) +
               rate[iy1, ix] * (1 - tx) * ty +
               rate[iy1, ix1] * tx * ty)
        # Auswahl ueber einen HASH des Original-Index statt ueber den
        # Index selbst: liefert eine stufenlose Rate und bricht zugleich
        # die Regelmaessigkeit auf (jeder n-te Index legte in gleichmaessig
        # erzeugten Wolken sichtbare Raster an). Deterministisch, also
        # ueber Samples hinweg stabil — dieselben Koerper bleiben
        # gestreamt und die Client-Interpolation reisst nicht.
        return idx[_index_hash(idx) < r_p]

    async def stream(self, ws) -> None:
        """Kontinuierlicher Sample-Push: haelt den Client-Puffer ~5 s
        Playback voll. Kleine Frames (<=256 KB) — TCP-Backpressure via
        await send drosselt automatisch auf Leitungstempo."""
        try:
            while self.running_val.value:
                # Producer tot, ohne dass ein Zerbersten ihn gestoppt hat =
                # Absturz. Ohne diese Pruefung wartet die Schleife ewig auf
                # ein head_val, das nie mehr waechst: der Film startet nicht
                # und niemand erfaehrt warum (der Traceback des KINDES steht
                # nur im Server-Log). Genau so verhielt sich ein Zustand mit
                # mehr als M_MAX massiven Koerpern.
                if not self.proc.is_alive() and not self.shatter_flag.value:
                    log.error("Film-Producer ist gestorben (exitcode %s) — "
                              "Stream beendet", self.proc.exitcode)
                    await ws.send(build_error(
                        "Film-Producer abgestuerzt (exitcode "
                        f"{self.proc.exitcode}) — Grund im Server-Log"))
                    return
                if self.shatter_flag.value == 1:
                    self.shatter_flag.value = 2
                    # status=7: Zerberst-Meldung + f64-Dump — der Client
                    # fuehrt shatter() aus und startet den Film neu.
                    pkt = struct.pack(
                        "<IIIId", 7, self.n,
                        int(self.shatter_a.value),
                        int(self.shatter_b.value),
                        self.shatter_t.value) + \
                        bytes(self.dump_shm.buf[0:4 * 8 * self.n])
                    await ws.send(pkt)
                ph = self.playhead_val.value
                if self.ph_ms:
                    alter_s = min(time.monotonic() - self.ph_ms,
                                  self.PH_EXTRAPOLATE_MAX_S)
                    ph += self.sub_rate * max(0.0, alter_s)
                if self.sent_abs is None:
                    self.sent_abs = max(self.tail_abs,
                                        int((ph - self.t0) /
                                            self.raster_days) - 1)
                sent_abs = self.sent_abs
                head_abs = self.head_val.value
                # Puffer-Ziel: 5 s Playback voraus (mind. 8 Raster)
                target = max(8 * self.raster_days, self.sub_rate * 5.0)
                sent_t = self.t0 + sent_abs * self.raster_days
                # Diagnose VOR den continue-Zweigen: nur so ist sichtbar,
                # warum im Stillstand nichts fliesst. "puffer" = Client hat
                # laut (extrapoliertem) Playhead genug Vorrat, "head" =
                # Producer noch nicht so weit, "-" = es wird gesendet.
                now_s = time.monotonic()
                if FilmSession.DIAG and \
                        now_s - self._diag_s >= STREAM_DIAG_INTERVAL_S:
                    self._diag_s = now_s
                    if sent_t - ph > target:
                        grund = "puffer"
                    elif sent_abs >= head_abs:
                        grund = "head"
                    else:
                        grund = "-"
                    # step/sps zeigen, wie fein gestreamt wird: bei kleiner
                    # sub_rate wird spacing klein und step faellt auf 1 —
                    # dann baut build_frame pro Sim-Tag ein Vielfaches an
                    # Frames (CPU!), obwohl weniger abgespielt wird.
                    kosten_d = self._sample_cost or self.sample_bytes
                    sps_d = min(20.0, max(0.5, self._bw * 0.7 / kosten_d))
                    step_d = max(1, int(round(
                        max(self.raster_days, self.sub_rate / sps_d) /
                        self.raster_days)))
                    log.info(
                        "[stream] block=%s ph=%.1f vorrat=%.1f (ziel %.1f) "
                        "tail=%.1f head=%.1f rate=%.1f sps=%.1f step=%d "
                        "kosten=%.0fKB",
                        grund, ph, sent_t - ph, target,
                        self.tail, self.head, self.sub_rate, sps_d, step_d,
                        kosten_d / 1024)
                if sent_t - ph > target or sent_abs >= head_abs:
                    await asyncio.sleep(0.03)
                    continue
                # Adaptive Dichte (Videostreaming-Prinzip): Wunsch sind
                # 20 Samples/s Anzeige, aber die GEMESSENE Bandbreite
                # deckelt — bei 55k Koerpern (1,2 MB/Sample) auf schmaler
                # Leitung werden es z. B. 1-2 Samples/s mit grossem
                # Sim-Abstand; der Client interpoliert dazwischen.
                # (Raster-dicht ohne Ruecksicht = 1 Bild pro Transferzeit
                # = Diashow.)
                kosten = self._sample_cost or self.sample_bytes
                sps = min(20.0, max(0.5, self._bw * 0.7 / kosten))
                spacing = max(self.raster_days, self.sub_rate / sps)
                step = max(1, int(round(spacing / self.raster_days)))
                avail = head_abs - sent_abs
                if avail < step:
                    await asyncio.sleep(0.03)
                    continue
                # Frame-Budget = ~50 ms Leitungszeit (gemessene _bw),
                # geklemmt auf 512 KB..8 MB: lokal passen so auch bei
                # 230k Koerpern mehrere Samples in einen Frame (sonst
                # wird der Stream-Loop selbst zum Engpass und der
                # Vergangenheits-Playhead reitet auf der Download-Kante),
                # remote bleibt es bei kleinen Frames.
                budget = min(8 * 1024 * 1024,
                             max(512 * 1024, int(self._bw * 0.05)))
                max_count = max(1, min(24, int(budget // kosten)))
                idxs = list(range(sent_abs, head_abs, step))[:max_count]
                if not idxs:
                    await asyncio.sleep(0.03)
                    continue
                frame = self.build_frame(idxs)
                t_send = asyncio.get_event_loop().time()
                await ws.send(frame)
                dur = asyncio.get_event_loop().time() - t_send
                # Bandbreiten-EWMA. Blitzschnelle lokale Sends (dur -> 0)
                # als Untergrenzen-Schaetzung werten statt sie zu
                # verwerfen — sonst klemmt _bw ewig auf dem Startwert und
                # die Stream-Dichte bleibt bei grossen N grundlos duenn
                # (Playhead reitet auf der Download-Kante: Ruckeln).
                # Ist-Kosten pro Sample fortschreiben — die Grundlage der
                # Dichte-/Budget-Rechnung im naechsten Durchlauf.
                ist_kosten = len(frame) / len(idxs)
                self._sample_cost = ist_kosten \
                    if self._sample_cost is None \
                    else 0.7 * self._sample_cost + 0.3 * ist_kosten
                # Bandbreite ueber ein FENSTER aus mehreren Sends mitteln,
                # nicht pro Frame. Ein einzelnes `await ws.send` kehrt
                # zurueck, sobald der Frame im Socket-Puffer liegt — bei
                # freiem Puffer geht dur gegen 0 und die Schaetzung
                # explodierte (gemessen 724 MB/s bei real ~4 MB/s). Erst
                # wenn der Puffer voll ist, blockiert send wirklich; ueber
                # mehrere Frames gemittelt enthaelt die Summe daher beides
                # und ergibt die tatsaechliche Abnahmerate des Clients.
                self._bw_bytes += len(frame)
                self._bw_dur += dur
                if self._bw_dur >= BW_FENSTER_S:
                    ist = self._bw_bytes / self._bw_dur
                    self._bw_n = getattr(self, "_bw_n", 0) + 1
                    w = 0.5 if self._bw_n <= 4 else 0.3
                    self._bw = (1 - w) * self._bw + w * ist
                    self._bw_bytes = 0.0
                    self._bw_dur = 0.0
                self.sent_abs = idxs[-1] + step
        except Exception:
            pass

    def resubscribe(self, t_days: float, rate: float,
                    jump: bool = False) -> None:
        # Auf den tatsaechlich vorhandenen Ringbereich klemmen. Ungeklemmt
        # fuehrte ein Sprung unter den Tail in einen stillen Deadlock: Der
        # Stream klemmt zwar seine Leseposition auf tail_abs, vergleicht
        # aber gegen den UNgeklemmten Playhead — `sent_t - ph > target`
        # ist dann dauerhaft wahr und es wird kein einziges Byte gesendet.
        # Gleichzeitig steht der Producer, weil er sich weit vor dem
        # (vermeintlichen) Playhead waehnt. Beide warten dann auf einen
        # Playhead, den nur der Client bewegen koennte — der aber auf
        # Daten wartet, die nie kommen. Ohne Log, ohne Fehler.
        self.playhead_val.value = min(max(t_days, self.tail), self.head)
        self.sub_rate = max(0.1, rate)
        self.ph_ms = time.monotonic()
        # Sprung wird vom CLIENT deklariert (Scrub/LIVE/Start) — eine
        # Heuristik ueber Zeitfenster erkannte kleine Ruck-Scrubs nicht
        # und streamte von der alten Position weiter (Wiedergabe hing).
        # Heartbeats (jump=False) fassen den laufenden Stream nie an.
        if jump:
            self.sent_abs = None

    def state_at_playhead(self, t_days: float) -> bytes | None:
        """Zustand am PLAYHEAD aus dem Ring statt am Producer-Kopf.

        Der Producer rechnet weit voraus (gemessen Kopf 462 bei Playhead
        81). Sein f64-Dump beschreibt daher den Kopf — startet der Client
        eine MUTATION (Injektion, Edit, Loeschen) damit neu, springt die
        Darstellung um den gesamten Vorlauf in die Zukunft.

        Fuer diesen Fall ist der Ring die richtige Quelle. Er traegt nur
        Positionen (f32); die Geschwindigkeit kommt aus dem ZENTRALEN
        Differenzenquotienten der Nachbarsamples. Das ist etwas anderes
        als die frueher verworfene Client-Schaetzung: die rechnete mit
        u16-quantisierten Positionen und lag dadurch 10-20 % daneben.
        Aus f32-Positionen ist der Fehler O(dt^2) — bei 0,5 Tagen Raster
        rund 1e-6 relativ fuer einen Guertelasteroiden. Sonnennahe
        Koerper mit stark gekruemmter Bahn sind schlechter gestellt, doch
        gerade sie tragen Sub-Stuetzpunkte; genauer wird es erst noetig,
        wenn das im Betrieb auffaellt.

        Der Genauigkeitsverlust ist hier ohnehin ohne Belang — eine
        Injektion aendert die Szene. Fuer die Engine-Uebergabe bleibt es
        beim exakten f64-Dump.

        Die vorausgerechnete Zukunft wird mit der Session verworfen; der
        Client startet den Film ab diesem Zustand neu.

        `t_days` kommt aus dem STOP-Paket, nicht aus `playhead_val`: der
        Heartbeat ist bis zu eine Sekunde alt und liegt bei hohem
        Abspieltempo um Dutzende Sim-Tage daneben.
        """
        head_abs = self.head_val.value
        # Der zentrale Differenzenquotient braucht beide Nachbarn; mit
        # weniger als drei Samples im Ring gibt es keinen brauchbaren
        # Zustand — dann traegt der f64-Dump (Vorlauf ist noch klein).
        if head_abs - self.tail_abs < 3:
            return None
        # Slot i traegt die Zeit t0 + (i+1)*raster (wie in build_frame)
        i = int(round((t_days - self.t0) / self.raster_days)) - 1
        i = max(self.tail_abs + 1, min(i, head_abs - 2))
        pos = self.slot_pos(i)
        dt_jahre = 2.0 * self.raster_days / TAGE_PRO_JAHR
        v = (self.slot_pos(i + 1) - self.slot_pos(i - 1)) / dt_jahre
        self._v_aus_stuetzpunkten(i, v)
        t_i = self.t0 + (i + 1) * self.raster_days
        return struct.pack("<IId", 5, self.n, t_i) + \
            pos.astype("<f8").tobytes() + v.astype("<f8").tobytes()

    def _v_aus_stuetzpunkten(self, i: int, v) -> None:
        """Geschwindigkeit der HEISSEN Koerper in `v` nachbessern.

        Der grobe Differenzenquotient ueber zwei Raster misst die SEHNE,
        nicht den Bogen. Sonnennah legt ein Koerper pro Raster fast einen
        halben Bogen zurueck, und dann ist die Sehne deutlich kuerzer:
        gemessen bei r = 0,05 AE 9,6 % zu wenig (die Bahn faellt danach
        von a = 0,050 auf 0,042 AE), bei r = 0,1 noch 1,3 %. Beim
        Neustart nach einer Injektion sackt die Szene dadurch sonnenwaerts
        zusammen — und weil der Fehler mit 1/r waechst, wird eine Wolke
        dabei auseinandergezogen.

        Genau diese Koerper tragen aber Stuetzpunkte. Der Differenzen-
        quotient ueber den vorletzten Stuetzpunkt von Slot i und den
        ersten von Slot i+1 spannt nur 2*raster/mSub — bei mSub = 8 also
        ein Achtel der Strecke, der Fehler faellt quadratisch auf ein
        Vierundsechzigstel. Koerper ohne Stuetzpunkte sind weit genug
        draussen, dass der grobe Quotient genuegt (ab r = 0,3 AE unter
        0,05 %).
        """
        ms = self.m_sub
        if ms < 2:
            return
        sidx_i, bahn_i = self.slot_sub(i)
        sidx_n, bahn_n = self.slot_sub(i + 1)
        if len(sidx_i) == 0 or len(sidx_n) == 0:
            return
        stelle = np.minimum(np.searchsorted(sidx_n, sidx_i),
                            len(sidx_n) - 1)
        beide = sidx_n[stelle] == sidx_i
        if not beide.any():
            return
        idx = sidx_i[beide].astype(np.int64)
        # vorletzter Stuetzpunkt von i liegt bei t_i - raster/mSub,
        # erster von i+1 bei t_i + raster/mSub
        dt = 2.0 * (self.raster_days / ms) / TAGE_PRO_JAHR
        vor = bahn_i[beide, ms - 2, :]
        nach = bahn_n[stelle[beide], 0, :]
        v[idx] = (nach[:, 0] - vor[:, 0]) / dt
        v[self.n + idx] = (nach[:, 1] - vor[:, 1]) / dt

    async def dump_state(self, playhead_days: float | None = None) \
            -> bytes | None:
        """Zustand fuer den Client-Neustart: exakter f64-Dump des Producers
        (Engine-Uebergabe) oder — wenn `playhead_days` gesetzt ist — der
        Ring-Zustand zur angezeigten Zeit (Mutation, s. state_at_playhead)."""
        if self.dump_req_val.value == 2 and not self.proc.is_alive():
            # Producer hat sich selbst beendet (Zerbersten) — sein
            # letzter Dump liegt bereit, niemand wuerde noch antworten.
            return struct.pack("<IId", 5, self.n, self.head) + \
                bytes(self.dump_shm.buf[0:4 * 8 * self.n])
        if playhead_days is not None:
            zustand = self.state_at_playhead(playhead_days)
            if zustand is not None:
                return zustand
            # Ring noch leer (Mutation direkt nach dem Filmstart): auf den
            # f64-Dump zurueckfallen — der Vorlauf ist dann ohnehin klein.
        self.dump_req_val.value = 1
        # Der Producer bedient den Dump erst zwischen zwei Kernel-Batches;
        # bei grossen N dauert ein Batch entsprechend lange. Kommt der Dump
        # nicht rechtzeitig, faellt der Client auf die u16-Schaetzung
        # zurueck und die Szene zerfliegt — deshalb grosszuegig warten.
        # Muss unter der Trennfrist des Clients bleiben (CUDA_DISCONNECT_MS).
        for _ in range(DUMP_WAIT_STEPS):
            if self.dump_req_val.value == 2:
                t_head = self.head
                return struct.pack("<IId", 5, self.n, t_head) + \
                    bytes(self.dump_shm.buf[0:4 * 8 * self.n])
            await asyncio.sleep(0.01)
        return None

    def stop(self) -> None:
        if self.stream_task:
            self.stream_task.cancel()
        self.running_val.value = 0
        try:
            self.proc.join(timeout=1.5)
            if self.proc.is_alive():
                self.proc.terminate()
        except Exception:
            pass
        try:
            self.shm.close()
            self.shm.unlink()
            self.ev_shm.close()
            self.ev_shm.unlink()
            self.dump_shm.close()
            self.dump_shm.unlink()
        except Exception:
            pass


def parse_film_start(buf: bytes):
    _typ, n, raster_days = HEADER.unpack_from(buf, 0)
    (t0_days,) = struct.unpack_from("<d", buf, HEADER.size)
    off = HEADER.size + 8
    f64 = np.dtype("<f8")
    arrays = []
    for _ in range(6):                      # x, y, vx, vy, mass, realR
        arrays.append(np.frombuffer(buf, f64, n, off))
        off += 8 * n
    visible = np.frombuffer(buf, np.uint8, n, off)
    off += n
    is_ast = np.frombuffer(buf, np.uint8, n, off)
    off += n
    is_star_bh = np.frombuffer(buf, np.uint8, n, off)
    off += n
    # Nachtraeglich injiziert (Wolken): steuert nur den Vorrang beim
    # Dichte-LOD des Streams, nicht die Physik.
    injiziert = np.frombuffer(buf, np.uint8, n, off)
    off += n
    # Flag-Byte (Bits 4-7 Version, Bit 0 Asti-Bounce) + Bahnstuetzpunkte
    flags = buf[off]
    if flags >> 4 != FILM_PROTO_VERSION:
        raise ValueError(
            "Film-Protokollversion veraltet — Seite neu laden (Strg+F5)")
    ast_bounce = (flags & 1) != 0
    m_sub = buf[off + 1]
    off += 2
    # Plummer-Softening in AE. > 0 waehlt den selbstgravitierenden Kernel
    # (selfgrav_kernel.py) und schaltet die gesamte Erkennung ab. Ein
    # eigener Wert statt Flag + Wert: 0 heisst "aus", das ist SSOT.
    (softening_au,) = struct.unpack_from("<d", buf, off)
    off += 8
    if off != len(buf):
        raise ValueError(f"Protokollfehler: {len(buf)} Bytes, erwartet {off}")
    return (raster_days, t0_days, arrays, visible, is_ast, is_star_bh,
            injiziert, ast_bounce, m_sub, softening_au)


def parse_full(buf: bytes):
    _typ, n, dt_years = HEADER.unpack_from(buf, 0)
    off = HEADER.size
    f64 = np.dtype("<f8")
    arrays = []
    for _ in range(5):                      # x, y, vx, vy, mass
        arrays.append(np.frombuffer(buf, f64, n, off))
        off += 8 * n
    visible = np.frombuffer(buf, np.uint8, n, off)
    off += n
    is_ast = np.frombuffer(buf, np.uint8, n, off)
    off += n
    if off != len(buf):
        raise ValueError(f"Protokollfehler: {len(buf)} Bytes, erwartet {off}")
    return dt_years, arrays, visible, is_ast


def build_response(n: int, f32_state: np.ndarray) -> bytes:
    return struct.pack("<II", 0, n) + f32_state.tobytes()


def build_error(msg: str) -> bytes:
    return struct.pack("<II", 1, 0) + msg.encode()


# Lazy-Kernel: der Serverprozess fasst die GPU erst an, wenn ein Client
# wirklich den Live-CUDA-Pfad nutzt (MSG_FULL). Serverstart, Auto-
# Detection-Ping und Film-Modus (Producer hat seinen eigenen Prozess +
# Context) belegen dann keinerlei VRAM. Der Lock verhindert eine
# Doppel-Kompilierung, wenn zwei Clients gleichzeitig den ersten
# FULL-Frame schicken.
_sim: NBodyCuda | None = None
_sim_lock = threading.Lock()
_device: int = 0


def get_sim() -> NBodyCuda:
    global _sim
    with _sim_lock:
        if _sim is None:
            _sim = NBodyCuda(_device)
            log.info("CUDA-Kernel initialisiert: Device %d (%s)",
                     _device, _sim.name())
        return _sim


def device_name(dev: int) -> str:
    # Reine Runtime-Abfrage — erzeugt keinen CUDA-Context.
    import cupy as cp
    return cp.cuda.runtime.getDeviceProperties(dev)["name"].decode()


# Verbindungszaehler fuer den Idle-Exit (socket activation): systemd
# startet den Server beim ersten Connect, wir beenden uns selbst, wenn
# laenger niemand verbunden ist — GPU und RAM sind dann komplett frei.
_active = [0]
_last_disconnect = [time.monotonic()]


async def handle(ws):
    peer = ws.remote_address
    _active[0] += 1
    log.info("Client verbunden: %s", peer)
    # Residenter Zustand DIESER Verbindung — mehrere Clients (lokal +
    # remote) haben getrennte Zustaende und stoeren sich nicht.
    state = None
    film: FilmSession | None = None
    frames = 0
    fulls = 0
    try:
        async for message in ws:
            if isinstance(message, str):
                # Textnachricht = Ping des Frontends bei der Auto-Detection.
                # mMax geht mit, damit der Client die Grenze kennt, BEVOR
                # er einen zu grossen Zustand hochlaedt — die Konstante
                # bleibt so in nbody_kernel.py zuhause statt im Client
                # dupliziert zu werden.
                await ws.send('{"backend":"cuda","device":"%s","mMax":%d}'
                              % (device_name(_device), M_MAX))
                continue
            try:
                typ, _n, dt_years = HEADER.unpack_from(message, 0)
                if typ == MSG_FILM_START:
                    raster_days, t0_days, (x, y, vx, vy, mass, real_r), \
                        visible, is_ast, is_star_bh, injiziert, \
                        ast_bounce, m_sub, softening_au = \
                        parse_film_start(message)
                    if film:
                        film.stop()
                    film = FilmSession(t0_days, raster_days, x, y, vx, vy,
                                       mass, real_r, visible, is_ast,
                                       is_star_bh, injiziert, ast_bounce,
                                       m_sub, softening_au)
                    fulls += 1
                    log.info(
                        "Film gestartet: N=%d, Raster %.2f Tage, "
                        "mSub=%d (%d Sub-Plaetze, Slot %.1f KB)%s",
                        len(x), film.raster_days, film.m_sub, film.sub_max,
                        film.sample_bytes / 1024,
                        f", selbstgravitierend (eps={softening_au:.1f} AE)"
                        if softening_au > 0 else "")
                    continue
                if typ == MSG_FILM_STOP:
                    if film:
                        # Exakten Endzustand an den Client — sonst startet
                        # die naechste Engine mit eingefrorenen Impulsen.
                        # _n = 1: Neustart nach Mutation, dann zaehlt der
                        # Zustand am PLAYHEAD (sonst springt die Szene um
                        # den Producer-Vorlauf in die Zukunft); der Playhead
                        # steht im dt_years-Feld des Headers.
                        state_msg = await film.dump_state(
                            dt_years if _n else None)
                        if state_msg:
                            await ws.send(state_msg)
                        film.stop()
                        film = None
                    continue
                if typ == MSG_FILM_SUB:
                    if film is None:
                        raise ValueError("kein Film aktiv")
                    rate, vcx, vcy, vhw, vhh, budget = struct.unpack_from(
                        "<dddddd", message, HEADER.size)
                    film.view = (vcx, vcy, vhw, vhh)
                    # Bit 1: Positionen log-polar kodieren. FRUEHER wurde
                    # das an vhw <= 0 erkannt, was zugleich eine Auto-Box
                    # ueber alle Koerper anforderte — beides ist jetzt
                    # getrennt, die Box folgt auch im Log-Zoom dem
                    # Sichtfenster (siehe build_frame).
                    film.log_zoom = bool(_n & 2)
                    # 0 = Automatik (aus gemessener Bandbreite)
                    film.lod_budget = max(0, int(budget))
                    film.resubscribe(dt_years, rate, jump=bool(_n & 1))
                    if film.stream_task is None:
                        film.stream_task = asyncio.create_task(
                            film.stream(ws))
                    frames += 1
                    continue
                if typ == MSG_FULL:
                    dt_years, (x, y, vx, vy, mass), visible, is_ast = \
                        parse_full(message)
                    sim = await asyncio.to_thread(get_sim)
                    state = await asyncio.to_thread(
                        sim.load_state, x, y, vx, vy, mass, visible, is_ast)
                    fulls += 1
                elif typ == MSG_DELTA:
                    recs = np.frombuffer(message, DELTA_REC, _n, HEADER.size)
                    await asyncio.to_thread(
                        get_sim().apply_updates, state,
                        recs["idx"].astype(np.int64), recs["v"])
                elif typ != MSG_STEP:
                    raise ValueError(f"unbekannter Nachrichtentyp {typ}")
                out = await asyncio.to_thread(get_sim().step, state, dt_years)
                frames += 1
                await ws.send(build_response(len(out) // 4, out))
            except Exception as e:          # Fehler zum Client melden
                log.exception("Frame-Fehler")
                await ws.send(build_error(str(e)))
    finally:
        if film:
            film.stop()
        _active[0] -= 1
        _last_disconnect[0] = time.monotonic()
        log.info("Client getrennt: %s (%d Frames, davon %d FULL-Uploads)",
                 peer, frames, fulls)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--ring-gib", type=float, default=8.0, metavar="GIB",
                    help="Groesse des Film-Ringpuffers in GiB. Begrenzt "
                         "durch den freien Platz in /dev/shm (df /dev/shm), "
                         "nicht durch den freien RAM.")
    ap.add_argument("--ring-jahre", type=float, default=300.0,
                    metavar="JAHRE",
                    help="Zweite Grenze des Ringpuffers: hoechstens so "
                         "viel SIM-ZEIT. Wirkt bei kleiner Koerperzahl, "
                         "wo das Byte-Budget sonst absurd viel Vorlauf "
                         "ergibt (11k Koerper bei 8 GiB: 92 Jahre). Es "
                         "gilt immer die kleinere der beiden Grenzen.")
    ap.add_argument("--device", type=int, default=None,
                    help="CUDA-Device-Index (Default: beste f64-GPU)")
    ap.add_argument("--det-gpus", type=int, default=FilmSession.DET_GPUS,
                    metavar="N",
                    help="Erkennungskarten pro Film-Session. Die "
                         "Bounce-Suche wird raeumlich auf sie aufgeteilt; "
                         "genutzt werden nur GPUs, die die Physik frei "
                         "laesst.")
    ap.add_argument("--diag", action="store_true",
                    help="Zeitanteile von Producer und Stream-Loop "
                         "mitloggen (Engpass-Suche, kostet etwas Tempo)")
    ap.add_argument("--idle-exit", type=int, default=0, metavar="SEK",
                    help="Selbst beenden nach SEK ohne Client (fuer "
                         "systemd socket activation); 0 = nie")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    global _device
    _device = args.device if args.device is not None else pick_device()
    FilmSession.MAX_BYTES = int(args.ring_gib * (1 << 30))
    FilmSession.MAX_RING_TAGE = args.ring_jahre * 365.25
    FilmSession.DET_GPUS = max(1, args.det_gpus)
    FilmSession.DIAG = args.diag
    log.info("Film-Ringpuffer: %.1f GiB, Erkennungskarten: %d",
             args.ring_gib, FilmSession.DET_GPUS)

    # systemd socket activation: LISTEN_FDS=1 -> der lauschende Socket
    # kommt als fd 3 herein; ohne systemd binden wir selbst.
    sock = None
    if os.environ.get("LISTEN_FDS"):
        sock = socket.socket(fileno=3)
    log.info("CUDA-Backend bereit auf Device %d (%s), %s — "
             "Kernel-Init lazy beim ersten Live-CUDA-Client",
             _device, device_name(_device),
             "socket-aktiviert" if sock else f"Port {args.port}")

    stop = asyncio.Event()

    async def idle_watch() -> None:
        while True:
            await asyncio.sleep(10)
            if _active[0] == 0 and \
                    time.monotonic() - _last_disconnect[0] > args.idle_exit:
                log.info("%d s ohne Client — beende mich (socket "
                         "activation startet bei Bedarf neu)",
                         args.idle_exit)
                stop.set()
                return

    # permessage-deflate AUS: die websockets-Default-Kompression jagt
    # jeden Frame durch zlib (~20-50 MB/s single-thread) — DAS war der
    # Stream-Durchsatz-Deckel. u16-Punktwolken komprimieren ohnehin
    # kaum; Bandbreite spart stattdessen das Dichte-LOD.
    # max_size deckt den FILM_START-Upload: der Client schickt den kompletten
    # Anfangszustand in EINER Nachricht, 52 B/Koerper (6×f64 + 4×u8). Die
    # Millionen Koerper der Particle-Mesh-Szenarien (1M Massen + 1M Tracer ≈
    # 104 MB) sprengen die alten 64 MiB. 256 MiB deckt ~5M Koerper; der Server
    # ist localhost-only — ein enges DoS-Limit braucht er nicht.
    serve_kwargs = dict(max_size=256 * 1024 * 1024, ping_interval=None,
                        compression=None)
    if sock is not None:
        server_ctx = websockets.serve(handle, sock=sock, **serve_kwargs)
    else:
        server_ctx = websockets.serve(handle, "127.0.0.1", args.port,
                                      **serve_kwargs)
    async with server_ctx:
        if args.idle_exit > 0:
            asyncio.create_task(idle_watch())
            await stop.wait()
        else:
            await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
