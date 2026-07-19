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
# Delta-Record: u32 idx | u32 pad | f64 x | f64 y | f64 vx | f64 vy
DELTA_REC = np.dtype([("idx", "<u4"), ("pad", "<u4"), ("v", "<f8", (4,))])


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
