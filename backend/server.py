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
from nbody_kernel import NBodyCuda, pick_device

log = logging.getLogger("solarsim-cuda")

HEADER = struct.Struct("<IId")   # typ, N/pad, dtYears
MSG_FULL = 0
MSG_STEP = 1
MSG_DELTA = 2
MSG_FILM_START = 3   # u32 typ | u32 N | f64 rasterTage | f64 t0Tage | FULL-Arrays
MSG_FILM_STOP = 4    # nur Header
MSG_FILM_SUB = 7     # u32 typ | u32 pad | f64 tTage | f64 rateTageProSek |
#                      f64 cx | f64 cy | f64 halbW | f64 halbH (Welt-AE;
#                      halbW<=0 = Auto-Box ueber alle Koerper)
#   Abo: Client meldet Playhead + Tempo (Start, Scrub, Tempo-Wechsel,
#   1-Hz-Heartbeat). Der Server STREAMT daraufhin kontinuierlich kleine
#   Frames (Push) und haelt den Client-Puffer ~5 s Playback voll —
#   TCP-Backpressure statt Anfrage-Roundtrips (Diashow-Ursache remote).
# Delta-Record: u32 idx | u32 pad | f64 x | f64 y | f64 vx | f64 vy
DELTA_REC = np.dtype([("idx", "<u4"), ("pad", "<u4"), ("v", "<f8", (4,))])

# Film-Antwort (Batch): u32 status=2 | u32 N | u32 count | u32 pad |
#   f64 tail | f64 head | count x f64 zeiten | count x sample (4N f32)


_SESSION_SEQ = [0]   # Round-Robin fuer die Erkennungskarten-Zuteilung


class FilmSession:
    """Proxy auf den Producer-PROZESS (film_producer.py): eigener Python-
    Prozess besitzt die GPU und schreibt in einen Shared-Memory-Ring —
    kein GIL-Sharing. Der Server liest Batches in Mikrosekunden direkt
    aus dem Ring, waehrend die GPU mit vollem Durchsatz rechnet und nur
    pausiert, wenn der Ring voll ist (Ueberschreib-Schutz vor dem
    Player-Playhead)."""

    MAX_BYTES = 4 << 30          # Ringpuffer-Obergrenze (~4 GB)

    def __init__(self, t0_days: float, raster_days: float,
                 x, y, vx, vy, mass, real_r, visible, is_ast,
                 is_star_bh, ast_bounce: bool = False):
        self.raster_days = max(0.1, raster_days)
        self.t0 = t0_days
        self.n = len(x)
        # Reines Punkte-Streaming: Sample = nur x|y f32 (8 Bytes/Koerper);
        # Masse/Sichtbarkeit laufen als Ereignisse im Event-Ring.
        self.sample_bytes = 8 * self.n
        self.capacity = max(2000, int(self.MAX_BYTES // self.sample_bytes))
        self.shm = shared_memory.SharedMemory(
            create=True, size=self.capacity * self.sample_bytes)
        self.ev_cap = 65536
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
        self.view = (0.0, 0.0, -1.0, -1.0)   # cx, cy, halbW, halbH (Auto)
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
                  self.shatter_t, _SESSION_SEQ[0]),
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

    def batch(self, t_days: float, spacing_days: float, count: int) -> bytes:
        head_abs = self.head_val.value
        if head_abs == 0:
            raise ValueError("Puffer noch leer")
        # Playhead an den Producer melden (Vorlauf-/Ueberschreib-Schutz)
        self.playhead_val.value = t_days
        step = max(1, int(round(spacing_days / self.raster_days)))
        i0 = int((t_days - self.t0) / self.raster_days) - 1
        i0 = max(self.tail_abs, min(i0, head_abs - 1))
        # Nahe der Kante den Abstand aufs Raster kollabieren: lieber die
        # real existierenden Samples dicht liefern als 1 Sample pro Batch
        # (sonst steht der Playhead und springt pro Roundtrip).
        avail = head_abs - i0
        if avail < step * 8:
            step = max(1, avail // 8)
        idxs = list(range(i0, head_abs, step))[:max(2, min(count, 120))]
        times = np.asarray(
            [self.t0 + (i + 1) * self.raster_days for i in idxs], "<f8")
        head = struct.pack("<IIII", 2, self.n, len(idxs),
                           int(self.coll_val.value))
        meta = struct.pack("<dd", self.tail, self.head)
        s = self.sample_bytes
        buf = self.shm.buf
        parts = [bytes(buf[(i % self.capacity) * s:
                           (i % self.capacity) * s + s]) for i in idxs]
        return head + meta + times.tobytes() + b"".join(parts)

    # ---- Streaming (Push) ----
    sub_rate = 60.0          # Tage/s Playback-Tempo des Clients
    stream_task = None
    sent_abs = None          # aktuelle Stream-Position (absoluter Sample-Index)
    _bw = 4e6                # gemessene Leitungs-Bandbreite (Bytes/s, EWMA)

    sent_ev = 0              # bereits gestreamte Ereignisse

    def build_frame(self, idxs: list) -> bytes:
        """v4-Frame: pro Sample nur die Koerper in der Referenz-Box
        (Culling!), Koordinaten als u16 relativ zur Box (User-Design:
        Integer-Streaming). Der Client dekodiert zurueck in Welt-
        koordinaten — die Kamera bleibt frei, die Box bestimmt nur
        Culling und Praezision."""
        # Neue Ereignisse einsammeln + lokalen vis-Spiegel pflegen
        ev_total = int(self.ev_count_val.value)
        ev_from = max(self.sent_ev, ev_total - self.ev_cap + 8)
        # 1024/Frame: Bounce-Stroeme (kind=1) erzeugen deutlich mehr
        # Ereignisse als Merges — Backlog holt ueber Folgeframes auf.
        ev_n = min(1024, ev_total - ev_from)
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
        s = self.sample_bytes
        buf = self.shm.buf
        times = np.asarray(
            [self.t0 + (i + 1) * self.raster_days for i in idxs], "<f8")
        blocks = []
        box = None
        for i in idxs:
            raw = np.frombuffer(buf, "<f4", 2 * self.n,
                                (i % self.capacity) * s)
            t_i = self.t0 + (i + 1) * self.raster_days
            alive_i = self._kill_t > t_i
            x = raw[0:self.n]
            y = raw[self.n:2 * self.n]
            if box is None:
                if hw <= 0 or hh <= 0:
                    # Auto-Box: gesamte Koerperverteilung (+5% Rand)
                    bx0, bx1 = float(x.min()), float(x.max())
                    by0, by1 = float(y.min()), float(y.max())
                    mx = 0.05 * max(bx1 - bx0, 1e-6)
                    my = 0.05 * max(by1 - by0, 1e-6)
                    box = (bx0 - mx, by0 - my,
                           max(bx1 - bx0 + 2 * mx, 1e-6),
                           max(by1 - by0 + 2 * my, 1e-6))
                else:
                    # Referenz-Box = 2x Viewport um die Kamera
                    box = (cx - 2 * hw, cy - 2 * hh,
                           max(4 * hw, 1e-6), max(4 * hh, 1e-6))
            x0, y0, spanx, spany = box
            sel = np.flatnonzero(
                alive_i & (x >= x0) & (x <= x0 + spanx)
                & (y >= y0) & (y <= y0 + spany))
            # Dichte-LOD: mehr sichtbare Punkte, als Bandbreite und
            # Client-Dekodierung bei 20 Samples/s verkraften -> nur jeden
            # stride-ten Asteroiden streamen (deterministisch ueber den
            # ORIGINAL-Index: dieselben Koerper bleiben ueber Samples
            # stabil gestreamt, die Interpolation reisst nicht). Massive
            # Koerper werden nie ausgeduennt; nicht gestreamte versteckt
            # der Client ueber die vorhandene _filmInView-Mechanik.
            # Physik und Ereignisse bleiben exakt — reine Darstellung.
            # Obergrenze 120k: mehr Punkte pro Sample schafft der
            # Client-Dekodier-/Interpolations-Loop nicht bei 60 FPS.
            lod_max = min(120_000,
                          max(20000, int(self._bw * 0.7 / 20.0 / 8.0)))
            # Hysterese: den Ausduennungsfaktor nur wechseln, wenn die
            # Punktzahl deutlich aus dem Zielband laeuft — sonst pendelt
            # er bei wandernden Wolken zwischen zwei Stufen und die
            # Dichte springt sichtbar hin und her.
            stride = getattr(self, "_lod_stride", 1)
            n_mit_altem = len(sel) / stride
            if n_mit_altem > lod_max * 1.15 or n_mit_altem < lod_max * 0.5:
                stride = max(1, int(np.ceil(len(sel) / lod_max)))
                self._lod_stride = stride
            if stride > 1:
                sel = sel[(sel % stride == 0) | ~self._is_ast[sel]]
            qx = np.clip((x[sel] - x0) / spanx * 65535.0,
                         0, 65535).astype("<u2")
            qy = np.clip((y[sel] - y0) / spany * 65535.0,
                         0, 65535).astype("<u2")
            block = struct.pack("<I", len(sel)) + \
                sel.astype("<u4").tobytes() + qx.tobytes() + qy.tobytes()
            block += b"\x00" * ((-len(block)) % 4)
            blocks.append(block)

        head = struct.pack("<IIII", 4, self.n, len(idxs), ev_n)
        meta = struct.pack("<dddddd", self.tail, self.head,
                           box[0], box[1], box[2], box[3])
        return head + meta + times.tobytes() + \
            b"".join(blocks) + b"".join(ev_parts)

    async def stream(self, ws) -> None:
        """Kontinuierlicher Sample-Push: haelt den Client-Puffer ~5 s
        Playback voll. Kleine Frames (<=256 KB) — TCP-Backpressure via
        await send drosselt automatisch auf Leitungstempo."""
        try:
            while self.running_val.value:
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
                if self.sent_abs is None:
                    self.sent_abs = max(self.tail_abs,
                                        int((ph - self.t0) /
                                            self.raster_days) - 1)
                sent_abs = self.sent_abs
                head_abs = self.head_val.value
                # Puffer-Ziel: 5 s Playback voraus (mind. 8 Raster)
                target = max(8 * self.raster_days, self.sub_rate * 5.0)
                sent_t = self.t0 + sent_abs * self.raster_days
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
                sps = min(20.0, max(0.5,
                    self._bw * 0.7 / self.sample_bytes))
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
                max_count = max(1, min(24, budget // self.sample_bytes))
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
                if len(frame) > 65536:
                    # Warmup: die ersten Messungen staerker gewichten,
                    # damit die Dichte nicht sekundenlang auf dem
                    # konservativen Startwert verharrt (LOD "springt").
                    self._bw_n = getattr(self, "_bw_n", 0) + 1
                    w = 0.5 if self._bw_n <= 8 else 0.3
                    self._bw = (1 - w) * self._bw + \
                        w * (len(frame) / max(dur, 0.002))
                self.sent_abs = idxs[-1] + step
        except Exception:
            pass

    def resubscribe(self, t_days: float, rate: float,
                    jump: bool = False) -> None:
        self.playhead_val.value = t_days
        self.sub_rate = max(0.1, rate)
        # Sprung wird vom CLIENT deklariert (Scrub/LIVE/Start) — eine
        # Heuristik ueber Zeitfenster erkannte kleine Ruck-Scrubs nicht
        # und streamte von der alten Position weiter (Wiedergabe hing).
        # Heartbeats (jump=False) fassen den laufenden Stream nie an.
        if jump:
            self.sent_abs = None

    async def dump_state(self) -> bytes | None:
        """Exakten f64-Zustand vom Producer anfordern (Engine-Uebergabe)."""
        if self.dump_req_val.value == 2 and not self.proc.is_alive():
            # Producer hat sich selbst beendet (Zerbersten) — sein
            # letzter Dump liegt bereit, niemand wuerde noch antworten.
            return struct.pack("<IId", 5, self.n, self.head) + \
                bytes(self.dump_shm.buf[0:4 * 8 * self.n])
        self.dump_req_val.value = 1
        for _ in range(150):
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
    ast_bounce = False
    if off + 1 == len(buf):               # Flag-Byte: Bits 4-7 Version
        flags = buf[off]
        if flags >> 4 != 1:
            raise ValueError(
                "Film-Protokollversion veraltet — Seite neu laden "
                "(Strg+F5)")
        ast_bounce = (flags & 1) != 0
        off += 1
    if off != len(buf):
        raise ValueError(f"Protokollfehler: {len(buf)} Bytes, erwartet {off}")
    return (raster_days, t0_days, arrays, visible, is_ast, is_star_bh,
            ast_bounce)


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
                # Textnachricht = Ping des Frontends bei der Auto-Detection
                await ws.send('{"backend":"cuda","device":"%s"}'
                              % device_name(_device))
                continue
            try:
                typ, _n, dt_years = HEADER.unpack_from(message, 0)
                if typ == MSG_FILM_START:
                    raster_days, t0_days, (x, y, vx, vy, mass, real_r), \
                        visible, is_ast, is_star_bh, ast_bounce = \
                        parse_film_start(message)
                    if film:
                        film.stop()
                    film = FilmSession(t0_days, raster_days, x, y, vx, vy,
                                       mass, real_r, visible, is_ast,
                                       is_star_bh, ast_bounce)
                    fulls += 1
                    log.info("Film gestartet: N=%d, Raster %.2f Tage",
                             len(x), film.raster_days)
                    continue
                if typ == MSG_FILM_STOP:
                    if film:
                        # Exakten Endzustand an den Client — sonst startet
                        # die naechste Engine mit eingefrorenen Impulsen
                        state_msg = await film.dump_state()
                        if state_msg:
                            await ws.send(state_msg)
                        film.stop()
                        film = None
                    continue
                if typ == MSG_FILM_SUB:
                    if film is None:
                        raise ValueError("kein Film aktiv")
                    rate, vcx, vcy, vhw, vhh = struct.unpack_from(
                        "<ddddd", message, HEADER.size)
                    film.view = (vcx, vcy, vhw, vhh)
                    film.resubscribe(dt_years, rate, jump=bool(_n))
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
    ap.add_argument("--device", type=int, default=None,
                    help="CUDA-Device-Index (Default: beste f64-GPU)")
    ap.add_argument("--idle-exit", type=int, default=0, metavar="SEK",
                    help="Selbst beenden nach SEK ohne Client (fuer "
                         "systemd socket activation); 0 = nie")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    global _device
    _device = args.device if args.device is not None else pick_device()

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
    serve_kwargs = dict(max_size=64 * 1024 * 1024, ping_interval=None,
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
