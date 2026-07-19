"""WebSocket-Backend: CUDA-f64-Physik fuer den Sonnensystem-Simulator.

Start:  ./venv/bin/python backend/server.py [--port 8765] [--device N]

Der Browser (index.html) verbindet sich auf ws://127.0.0.1:<port> und
schickt pro Frame den kompletten Koerperzustand; das Backend rechnet den
Frame (adaptive Yoshida-Substeps, f64) auf der GPU und schickt die neuen
Positionen/Geschwindigkeiten zurueck. Kollisionen, Trails und UI bleiben
im Browser — das Backend ist ein reiner Physik-Beschleuniger, die HTML
funktioniert ohne ihn unveraendert (hardwareagnostisch).

Binaerprotokoll (Little-Endian):
  Request:  f64 dtYears | u32 N | u32 pad |
            x[N] f64 | y[N] | vx[N] | vy[N] | mass[N] |
            visible[N] u8 | isAst[N] u8
  Response: u32 status (0=ok) | u32 pad | x[N] f64 | y[N] | vx[N] | vy[N]
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

HEADER = struct.Struct("<dII")   # dtYears, N, pad


def parse_request(buf: bytes):
    dt_years, n, _pad = HEADER.unpack_from(buf, 0)
    off = HEADER.size
    f64 = np.dtype("<f8")
    arrays = []
    for _ in range(5):                      # x, y, vx, vy, mass
        arrays.append(np.frombuffer(buf, f64, n, off))
        off += 8 * n
    visible = np.frombuffer(buf, np.uint8, n, off); off += n
    is_ast = np.frombuffer(buf, np.uint8, n, off); off += n
    if off != len(buf):
        raise ValueError(f"Protokollfehler: {len(buf)} Bytes, erwartet {off}")
    return dt_years, arrays, visible, is_ast


def build_response(x, y, vx, vy) -> bytes:
    head = struct.pack("<II", 0, 0)
    return head + x.tobytes() + y.tobytes() + vx.tobytes() + vy.tobytes()


def build_error(msg: str) -> bytes:
    return struct.pack("<II", 1, 0) + msg.encode()


async def handle(ws, sim: NBodyCuda):
    peer = ws.remote_address
    log.info("Client verbunden: %s", peer)
    try:
        async for message in ws:
            if isinstance(message, str):
                # Textnachricht = Ping des Frontends bei der Auto-Detection
                await ws.send('{"backend":"cuda","device":"%s"}' % sim.name())
                continue
            try:
                dt_years, (x, y, vx, vy, mass), visible, is_ast = \
                    parse_request(message)
                nx, ny, nvx, nvy = await asyncio.to_thread(
                    sim.advance, dt_years, x, y, vx, vy, mass, visible, is_ast)
                await ws.send(build_response(nx, ny, nvx, nvy))
            except Exception as e:          # Fehler zum Client melden
                log.exception("Frame-Fehler")
                await ws.send(build_error(str(e)))
    finally:
        log.info("Client getrennt: %s", peer)


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
