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

Sample-Format (identisch zum Protokoll): [x|y|vx|vy|masse] als f32 in
Originalreihenfolge + Sichtbarkeit u8 + Padding auf 4 Bytes.
"""
from __future__ import annotations

import time

import numpy as np


EV_BYTES = 24    # Ereignis: f64 tTage | u32 a (Ueberlebender/0xFFFFFFFF) |
#                  u32 b (Verlierer) | f32 neueMasse | u32 pad


def producer_main(shm_name: str, sample_bytes: int, capacity: int,
                  ev_name: str, ev_cap: int, ev_count_val,
                  head_val, playhead_val, coll_val, running_val,
                  state: dict, raster_days: float, t0_days: float) -> None:
    # CUDA erst IM Kindprozess initialisieren (spawn-Kontext!)
    from multiprocessing import shared_memory

    import cupy as cp

    from nbody_kernel import G_AU, NBodyCuda, pick_device

    sim = NBodyCuda(pick_device())
    x = state["x"]
    y = state["y"]
    vx = state["vx"]
    vy = state["vy"]
    mass = np.array(state["mass"], dtype=np.float64, copy=True)
    real_r = np.array(state["realR"], dtype=np.float64, copy=True)
    vis = np.array(state["visible"], dtype=np.uint8, copy=True)
    is_ast = np.array(state["isAst"], dtype=np.uint8, copy=True) != 0
    st = sim.load_state(x, y, vx, vy, mass, vis, state["isAst"])
    n = len(x)
    pad = (-n) % 4
    mass_f32 = mass.astype("<f4").tobytes()
    vis_pad = vis.tobytes() + b"\x00" * pad
    collisions = 0
    dt_years = raster_days / 365.25

    with cp.cuda.Device(sim.device):
        g_vis = cp.asarray(vis)
        g_rr = cp.asarray(real_r.astype(np.float32))
        g_ast = cp.asarray(is_ast)
        g_prev = None

    shm = shared_memory.SharedMemory(name=shm_name)
    buf = shm.buf
    ev_shm = shared_memory.SharedMemory(name=ev_name)
    ev_buf = ev_shm.buf
    k = 0

    import struct as _struct

    def emit_event(a: int, b: int, new_mass: float) -> None:
        # Merge/Kill als Ereignis in den Event-Ring — Samples selbst
        # tragen nur noch Positionen (reines Punkte-Streaming).
        i = ev_count_val.value % ev_cap
        t_ev = t0_days + (k + 1) * raster_days
        ev_buf[i * EV_BYTES:(i + 1) * EV_BYTES] = _struct.pack(
            "<dIIfI", t_ev, a & 0xFFFFFFFF, b, new_mass, 0)
        ev_count_val.value += 1

    def detect_and_merge(sample: np.ndarray) -> None:
        nonlocal collisions, mass_f32, vis_pad, g_prev
        st_local = st
        sx = sample[0:n]
        sy = sample[n:2 * n]
        svx = sample[2 * n:3 * n]
        svy = sample[3 * n:4 * n]
        hit_pairs = []
        runaway_np = None
        m_alive = np.empty(0, dtype=np.int64)
        with cp.cuda.Device(sim.device):
            g = st_local["out_f32"]
            gx = g[0:n]
            gy = g[n:2 * n]
            gvx = g[2 * n:3 * n]
            gvy = g[3 * n:4 * n]
            px, py = (gx, gy) if g_prev is None else \
                (g_prev[0:n], g_prev[n:2 * n])
            m_idx_all = st_local["m_idx_h"]
            m_alive = m_idx_all[(vis[m_idx_all] != 0) & (mass[m_idx_all] > 0)]
            if len(m_alive):
                gm = cp.asarray(m_alive)
                cx = gx[gm][:, None]
                cy = gy[gm][:, None]
                rsum = g_rr[gm][:, None] + g_rr[None, :]
                rsum2 = rsum * rsum
                alive = g_vis != 0

                def seg_hit(p0x, p0y, p1x, p1y):
                    ssx = (p1x - p0x)[None, :]
                    ssy = (p1y - p0y)[None, :]
                    seg2 = ssx * ssx + ssy * ssy
                    tt = ((cx - p0x[None, :]) * ssx +
                          (cy - p0y[None, :]) * ssy) / cp.where(
                              seg2 > 0, seg2, cp.float32(1.0))
                    tt = cp.clip(tt, 0.0, 1.0)
                    ddx = p0x[None, :] + tt * ssx - cx
                    ddy = p0y[None, :] + tt * ssy - cy
                    return ddx * ddx + ddy * ddy <= rsum2

                g_dt = cp.float32(dt_years)
                hit2d = (seg_hit(px, py, gx, gy) |
                         seg_hit(gx, gy, gx + gvx * g_dt, gy + gvy * g_dt)) \
                    & alive[None, :]
                hit2d[cp.arange(len(gm)), gm] = False
                rows, cols = cp.nonzero(hit2d)
                if rows.size:
                    hit_pairs = list(zip(cp.asnumpy(rows).tolist(),
                                         cp.asnumpy(cols).tolist()))
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
                    runaway = (v2 > 9.0 * vesc2) & g_ast & (g_vis != 0)
                    ridx = cp.flatnonzero(runaway)
                    if ridx.size:
                        runaway_np = cp.asnumpy(ridx)
            g_prev = g.copy()
        changed = False
        for row, j in hit_pairs:
            mi = int(m_alive[row])
            j = int(j)
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
            with cp.cuda.Device(sim.device):
                g_vis[b] = 0
                g_rr[a] = np.float32(real_r[a])
            emit_event(a, b, float(m_ges))
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
                with cp.cuda.Device(sim.device):
                    g_vis[j] = 0
                emit_event(0xFFFFFFFF, j, 0.0)   # reiner Kill (Numerik-Waechter)
                collisions += 1
                changed = True
        if changed:
            coll_val.value = collisions

    try:
        while running_val.value:
            # Ueberschreib-Schutz: Slot (k - capacity) wird gleich
            # ueberschrieben — er muss hinter dem Player-Playhead liegen.
            ph_abs = int((playhead_val.value - t0_days) / raster_days)
            # 70% des Rings als Vorlauf, 30% bleiben Rueckspul-Historie —
            # sonst frisst die Eviction sich bis an den Playhead heran
            # und der Player reitet stotternd auf der Abbruchkante.
            if k - ph_abs >= int(capacity * 0.7):
                time.sleep(0.01)
                continue
            out = sim.step(st, dt_years)
            detect_and_merge(out)
            # Reines Punkte-Streaming: nur x|y (8 Bytes/Koerper) — alles
            # andere (Masse/Sichtbarkeit) laeuft als Ereignis.
            sample = out[0:2 * n].tobytes()
            slot = k % capacity
            buf[slot * sample_bytes:(slot + 1) * sample_bytes] = sample
            k += 1
            head_val.value = k
    finally:
        shm.close()
        ev_shm.close()
