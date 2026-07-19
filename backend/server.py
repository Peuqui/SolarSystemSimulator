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
import struct

import numpy as np
import websockets

from nbody_kernel import NBodyCuda, pick_device

log = logging.getLogger("solarsim-cuda")

HEADER = struct.Struct("<IId")   # typ, N/pad, dtYears
MSG_FULL = 0
MSG_STEP = 1
MSG_DELTA = 2
MSG_FILM_START = 3   # u32 typ | u32 N | f64 rasterTage | f64 t0Tage | FULL-Arrays
MSG_FILM_STOP = 4    # nur Header
MSG_FILM_REQ = 5     # u32 typ | u32 pad | f64 tTage (gewuenschte Sim-Zeit)
# Delta-Record: u32 idx | u32 pad | f64 x | f64 y | f64 vx | f64 vy
DELTA_REC = np.dtype([("idx", "<u4"), ("pad", "<u4"), ("v", "<f8", (4,))])

# Film-Antwort (Batch): u32 status=2 | u32 N | u32 count | u32 pad |
#   f64 tail | f64 head | count x f64 zeiten | count x sample (4N f32)


class FilmSession:
    """Freilaufender Producer: rechnet die Simulation mit maximalem
    Durchsatz voraus und fuellt einen Ringpuffer aus f32-Samples in festem
    Sim-Zeit-Raster. Der Client liest daraus wie aus einem Video (Player).
    Stufe 1: ohne Kollisionen — die Koerperliste bleibt konstant."""

    MAX_BYTES = 4 << 30          # Ringpuffer-Obergrenze (~4 GB)
    MIN_KEEP = 1000              # nie unter diese Sample-Zahl evikten

    def __init__(self, sim: NBodyCuda, state: dict,
                 t0_days: float, raster_days: float):
        self.sim = sim
        self.state = state
        self.raster_days = max(0.1, raster_days)
        self.t0 = t0_days        # Sim-Zeit des Startzustands
        self.samples: list[np.ndarray] = []
        self.bytes = 0
        self.running = True
        self.task: asyncio.Task | None = None

    @property
    def tail(self) -> float:
        return self.t0

    @property
    def head(self) -> float:
        return self.t0 + len(self.samples) * self.raster_days

    async def produce(self) -> None:
        dt_years = self.raster_days / 365.25
        try:
            while self.running:
                out = await asyncio.to_thread(self.sim.step, self.state, dt_years)
                self.samples.append(out)
                self.bytes += out.nbytes
                if self.bytes > self.MAX_BYTES and len(self.samples) > self.MIN_KEEP:
                    drop = len(self.samples) // 8
                    for s in self.samples[:drop]:
                        self.bytes -= s.nbytes
                    del self.samples[:drop]
                    self.t0 += drop * self.raster_days
        except Exception:
            log.exception("Film-Producer abgebrochen")
            self.running = False

    def stop(self) -> None:
        self.running = False
        if self.task:
            self.task.cancel()

    def batch(self, t_days: float, spacing_days: float, count: int) -> bytes:
        """Sample-Fenster ab t in gewuenschter Dichte (Videoplayer-Prinzip:
        der Client holt ganze Batches in Playback-Aufloesung voraus, statt
        pro Roundtrip ein einzelnes Paar — sonst Diashow bei schnellem
        Playback uebers Netz)."""
        n_s = len(self.samples)
        if n_s == 0:
            raise ValueError("Puffer noch leer")
        step = max(1, int(round(spacing_days / self.raster_days)))
        i0 = int((t_days - self.t0) / self.raster_days) - 1
        i0 = max(0, min(i0, n_s - 1))
        idxs = list(range(i0, n_s, step))[:max(2, min(count, 120))]
        times = np.asarray(
            [self.t0 + (i + 1) * self.raster_days for i in idxs], "<f8")
        head = struct.pack("<IIII", 2, len(self.samples[0]) // 4,
                           len(idxs), 0)
        meta = struct.pack("<dd", self.tail, self.head)
        return head + meta + times.tobytes() + \
            b"".join(self.samples[i].tobytes() for i in idxs)


def parse_film_start(buf: bytes):
    _typ, n, raster_days = HEADER.unpack_from(buf, 0)
    (t0_days,) = struct.unpack_from("<d", buf, HEADER.size)
    off = HEADER.size + 8
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
    return raster_days, t0_days, arrays, visible, is_ast


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


async def handle(ws, sim: NBodyCuda):
    peer = ws.remote_address
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
                await ws.send('{"backend":"cuda","device":"%s"}' % sim.name())
                continue
            try:
                typ, _n, dt_years = HEADER.unpack_from(message, 0)
                if typ == MSG_FILM_START:
                    raster_days, t0_days, (x, y, vx, vy, mass), visible, \
                        is_ast = parse_film_start(message)
                    if film:
                        film.stop()
                    f_state = await asyncio.to_thread(
                        sim.load_state, x, y, vx, vy, mass, visible, is_ast)
                    film = FilmSession(sim, f_state, t0_days, raster_days)
                    film.task = asyncio.create_task(film.produce())
                    fulls += 1
                    log.info("Film gestartet: N=%d, Raster %.2f Tage",
                             len(x), film.raster_days)
                    continue
                if typ == MSG_FILM_STOP:
                    if film:
                        film.stop()
                        film = None
                    continue
                if typ == MSG_FILM_REQ:
                    if film is None:
                        raise ValueError("kein Film aktiv")
                    # Layout: u32 typ | u32 count | f64 tTage | f64 spacingTage
                    (spacing_days,) = struct.unpack_from(
                        "<d", message, HEADER.size)
                    # bei leerem Puffer kurz auf den Producer warten
                    for _ in range(200):
                        if film.samples or not film.running:
                            break
                        await asyncio.sleep(0.02)
                    await ws.send(film.batch(dt_years, spacing_days, _n))
                    frames += 1
                    continue
                if typ == MSG_FULL:
                    dt_years, (x, y, vx, vy, mass), visible, is_ast = \
                        parse_full(message)
                    state = await asyncio.to_thread(
                        sim.load_state, x, y, vx, vy, mass, visible, is_ast)
                    fulls += 1
                elif typ == MSG_DELTA:
                    recs = np.frombuffer(message, DELTA_REC, _n, HEADER.size)
                    await asyncio.to_thread(
                        sim.apply_updates, state,
                        recs["idx"].astype(np.int64), recs["v"])
                elif typ != MSG_STEP:
                    raise ValueError(f"unbekannter Nachrichtentyp {typ}")
                out = await asyncio.to_thread(sim.step, state, dt_years)
                frames += 1
                await ws.send(build_response(len(out) // 4, out))
            except Exception as e:          # Fehler zum Client melden
                log.exception("Frame-Fehler")
                await ws.send(build_error(str(e)))
    finally:
        if film:
            film.stop()
        log.info("Client getrennt: %s (%d Frames, davon %d FULL-Uploads)",
                 peer, frames, fulls)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", type=int, default=None,
                    help="CUDA-Device-Index (Default: beste f64-GPU)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    device = args.device if args.device is not None else pick_device()
    sim = NBodyCuda(device)
    log.info("CUDA-Backend bereit auf Device %d (%s), Port %d",
             device, sim.name(), args.port)

    async with websockets.serve(
            lambda ws: handle(ws, sim), "127.0.0.1", args.port,
            max_size=64 * 1024 * 1024, ping_interval=None):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
