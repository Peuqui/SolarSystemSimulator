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

import cupy as cp
import numpy as np
import websockets

from nbody_kernel import G_AU, NBodyCuda, pick_device

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
                 t0_days: float, raster_days: float,
                 mass: np.ndarray, visible: np.ndarray,
                 real_r: np.ndarray, is_ast: np.ndarray):
        self.sim = sim
        self.state = state
        self.raster_days = max(0.1, raster_days)
        self.t0 = t0_days        # Sim-Zeit des Startzustands
        self.n = state["N"]
        # Host-Spiegel (Originalreihenfolge) fuer Kollisionslogik und
        # Sample-Zusammenbau — aendern sich nur bei Merges.
        self.mass = np.array(mass, dtype=np.float64, copy=True)
        self.vis = np.array(visible, dtype=np.uint8, copy=True)
        self.real_r = np.array(real_r, dtype=np.float64, copy=True)
        self.is_ast = np.array(is_ast, dtype=np.uint8, copy=True) != 0
        # GPU-Spiegel fuer die Kollisionserkennung — die Positionsdaten
        # liegen nach jedem Step ohnehin im residenten out_f32-Puffer;
        # die (M,N)-Masken rechnen auf der Karte, nur Treffer-Indizes
        # (fast immer leer) kommen zum Host.
        with cp.cuda.Device(sim.device):
            self._g_vis = cp.asarray(self.vis)
            self._g_mass = cp.asarray(self.mass.astype(np.float32))
            self._g_rr = cp.asarray(self.real_r.astype(np.float32))
            self._g_ast = cp.asarray(self.is_ast)
            self._g_prev = None
        self._mass_f32 = self.mass.astype("<f4").tobytes()
        self._vis_pad = self.vis.tobytes() + b"\x00" * ((-self.n) % 4)
        self.collisions = 0
        self._prev_xy = None     # Positionen des Vor-Samples (Swept-Check)
        self.samples: list[bytes] = []
        self.bytes = 0
        self.running = True
        self.task: asyncio.Task | None = None
        # Vorlauf-Fenster (Videostreaming-Prinzip): der Producer rechnet
        # nur begrenzt ueber den Player-Playhead hinaus und legt sich dann
        # schlafen — statt ungebremst zukunft zu produzieren, die der
        # naechste inject verwirft und deren eviction den zuschauer
        # ueberholt. last_req_t = zuletzt angefragter Playhead;
        # lookahead_days wird aus dem Konsumtempo der Batch-Anfragen
        # abgeleitet (~90 s Playback beim aktuellen Tempo).
        self.last_req_t = t0_days
        self.lookahead_days = 730.0

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
                # Vorlauf voll -> schlafen, bis der Player aufholt
                if self.head - self.last_req_t > self.lookahead_days:
                    await asyncio.sleep(0.05)
                    continue
                sample = await asyncio.to_thread(self._produce_one, dt_years)
                self.samples.append(sample)
                self.bytes += len(sample)
                # GIL-Luft fuer den Event-Loop: mit komfortablem Polster
                # vor dem Player kurz durchatmen, damit Batch-Anfragen
                # sofort beantwortet werden (sonst Stand-und-Sprung beim
                # Client trotz produzierender GPU). Beim Aufholen: Vollgas.
                if self.head - self.last_req_t > 30 * self.raster_days:
                    await asyncio.sleep(0.004)
                else:
                    await asyncio.sleep(0)
                if self.bytes > self.MAX_BYTES and len(self.samples) > self.MIN_KEEP:
                    # Eviction-Schutz: nie ueber (Playhead - Vorlauf) hinaus
                    # loeschen — Position und Rueckspul-Marge des Zuschauers
                    # sind heilig. Ist nichts loeschbar, pausiert der
                    # Producer (Speicher-Deckel statt Zuschauer-Ueberholen).
                    guard_t = self.last_req_t - self.lookahead_days
                    max_drop = int((guard_t - self.t0) / self.raster_days)
                    drop = min(len(self.samples) // 8, max(0, max_drop))
                    if drop == 0:
                        await asyncio.sleep(0.2)
                        continue
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

    def _detect_and_merge(self, sample: np.ndarray) -> None:
        """Kollisionen (massive x alle, Swept + ballistische Prognose) und
        Numerik-Waechter — komplett auf der GPU: die Positionsdaten liegen
        nach jedem Step ohnehin im residenten out_f32-Puffer, nur Treffer-
        Indizes (fast immer leer) kommen zum Host. Merges wirken auf den
        residenten f64-Zustand UND die Host-/GPU-Spiegel; aeltere Samples
        bleiben unveraendert (rueckspulbar)."""
        st = self.state
        n = self.n
        x = sample[0:n]
        y = sample[n:2 * n]
        vx = sample[2 * n:3 * n]
        vy = sample[3 * n:4 * n]
        hit_pairs = []
        runaway_np = None
        with cp.cuda.Device(self.sim.device):
            g = st["out_f32"]
            gx = g[0:n]
            gy = g[n:2 * n]
            gvx = g[2 * n:3 * n]
            gvy = g[3 * n:4 * n]
            if self._g_prev is None:
                px, py = gx, gy
            else:
                px, py = self._g_prev[0:n], self._g_prev[n:2 * n]
            m_idx_all = st["m_idx_h"]
            m_alive = m_idx_all[(self.vis[m_idx_all] != 0)
                                & (self.mass[m_idx_all] > 0)]
            if len(m_alive):
                gm = cp.asarray(m_alive)
                cx = gx[gm][:, None]
                cy = gy[gm][:, None]
                rsum = self._g_rr[gm][:, None] + self._g_rr[None, :]
                rsum2 = rsum * rsum
                alive = self._g_vis != 0

                def seg_hit(p0x, p0y, p1x, p1y):
                    sx = (p1x - p0x)[None, :]
                    sy = (p1y - p0y)[None, :]
                    seg2 = sx * sx + sy * sy
                    tt = ((cx - p0x[None, :]) * sx +
                          (cy - p0y[None, :]) * sy) / cp.where(
                              seg2 > 0, seg2, cp.float32(1.0))
                    tt = cp.clip(tt, 0.0, 1.0)
                    ddx = p0x[None, :] + tt * sx - cx
                    ddy = p0y[None, :] + tt * sy - cy
                    return ddx * ddx + ddy * ddy <= rsum2

                dt_y = cp.float32(self.raster_days / 365.25)
                hit2d = (seg_hit(px, py, gx, gy) |
                         seg_hit(gx, gy, gx + gvx * dt_y, gy + gvy * dt_y)) \
                    & alive[None, :]
                hit2d[cp.arange(len(gm)), gm] = False
                rows, cols = cp.nonzero(hit2d)
                if rows.size:
                    hit_pairs = list(zip(cp.asnumpy(rows).tolist(),
                                         cp.asnumpy(cols).tolist()))
                # Numerik-Waechter (siehe unten) mit Schwerpunkt der massiven
                mw = self.mass[m_alive]
                msum = float(mw.sum())
                if msum > 0:
                    bx = float((x[m_alive] * mw).sum() / msum)
                    by = float((y[m_alive] * mw).sum() / msum)
                    r = cp.maximum(cp.hypot(gx - cp.float32(bx),
                                            gy - cp.float32(by)),
                                   cp.float32(1e-6))
                    v2 = gvx * gvx + gvy * gvy
                    vesc2 = cp.float32(2.0 * G_AU * msum) / r
                    runaway = (v2 > 9.0 * vesc2) & self._g_ast \
                        & (self._g_vis != 0)
                    ridx = cp.flatnonzero(runaway)
                    if ridx.size:
                        runaway_np = cp.asnumpy(ridx)
            self._g_prev = g.copy()
        changed = False
        for row, j in hit_pairs:
            mi = int(m_alive[row])
            j = int(j)
            if not self.vis[mi] or not self.vis[j] or self.mass[mi] <= 0:
                continue
            a, b = (mi, j) if self.mass[mi] >= self.mass[j] else (j, mi)
            m_a, m_b = self.mass[a], self.mass[b]
            m_ges = m_a + m_b
            if m_ges <= 0:
                continue
            nx = (x[a] * m_a + x[b] * m_b) / m_ges
            ny = (y[a] * m_a + y[b] * m_b) / m_ges
            nvx = (vx[a] * m_a + vx[b] * m_b) / m_ges
            nvy = (vy[a] * m_a + vy[b] * m_b) / m_ges
            self.sim.apply_body_state(self.state, a, nx, ny, nvx, nvy, m_ges)
            self.sim.deactivate_body(self.state, b)
            self.mass[a] = m_ges
            self.mass[b] = 0.0
            self.vis[b] = 0
            self.real_r[a] = (self.real_r[a] ** 3 +
                              self.real_r[b] ** 3) ** (1.0 / 3.0)
            with cp.cuda.Device(self.sim.device):
                self._g_vis[b] = 0
                self._g_mass[a] = np.float32(m_ges)
                self._g_mass[b] = 0
                self._g_rr[a] = np.float32(self.real_r[a])
            self.collisions += 1
            changed = True
        if runaway_np is not None:
            for j in runaway_np:
                j = int(j)
                if not self.vis[j]:
                    continue
                self.sim.deactivate_body(self.state, j)
                self.mass[j] = 0.0
                self.vis[j] = 0
                with cp.cuda.Device(self.sim.device):
                    self._g_vis[j] = 0
                    self._g_mass[j] = 0
                self.collisions += 1
                changed = True
        if changed:
            self._mass_f32 = self.mass.astype("<f4").tobytes()
            self._vis_pad = self.vis.tobytes() + b"\x00" * ((-self.n) % 4)

    def _produce_one(self, dt_years: float) -> bytes:
        """Ein Producer-Schritt komplett im Worker-Thread: Kernel-Step,
        GPU-Kollisionserkennung, Sample-Zusammenbau — ein Thread-Hop
        statt drei."""
        out = self.sim.step(self.state, dt_years)
        self._detect_and_merge(out)
        return out.tobytes() + self._mass_f32 + self._vis_pad

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
        head = struct.pack("<IIII", 2, self.n, len(idxs), self.collisions)
        meta = struct.pack("<dd", self.tail, self.head)
        return head + meta + times.tobytes() + \
            b"".join(self.samples[i] for i in idxs)


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
                    raster_days, t0_days, (x, y, vx, vy, mass, real_r), \
                        visible, is_ast = parse_film_start(message)
                    if film:
                        film.stop()
                    f_state = await asyncio.to_thread(
                        sim.load_state, x, y, vx, vy, mass, visible, is_ast)
                    film = FilmSession(sim, f_state, t0_days, raster_days,
                                       mass, visible, real_r, is_ast)
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
                    # Playhead + Konsumtempo fuer das Vorlauf-Fenster:
                    # ein Batch deckt ~3 s Playback -> Faktor 30 = ~90 s.
                    # Direkte Zuweisung (kein max): nach einem Rueck-Scrub
                    # muss der Eviction-Schutz die AKTUELLE Position decken.
                    film.last_req_t = dt_years
                    film.lookahead_days = min(
                        36500.0, max(365.0, 30.0 * spacing_days * _n))
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
