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


EV_BYTES = 32    # Ereignis: f64 tTage | u32 a (Ueberlebender/0xFFFFFFFF) |
#                  u32 b (Verlierer) | f32 neueMasse | u32 kind |
#                  f32 x | f32 y (exakter Ereignis-Ort — der Client kann
#                  ihn NICHT aus dem Stream rekonstruieren: das Opfer
#                  fehlt im Folge-Sample, seine interpolierte Position
#                  stammt je nach Stream-Dichte von Tagen davor)
#                  kind 0 = merge/kill (b verschwindet), 1 = bounce (nur
#                  Zaehler + Visual, niemand stirbt)


def producer_main(shm_name: str, sample_bytes: int, capacity: int,
                  ev_name: str, ev_cap: int, ev_count_val,
                  dump_name: str, dump_req_val,
                  head_val, playhead_val, coll_val, running_val,
                  state: dict, raster_days: float, t0_days: float,
                  ast_bounce: bool = False,
                  shatter_flag=None, shatter_a=None, shatter_b=None,
                  shatter_t=None, det_rank: int = 0) -> None:
    # CUDA erst IM Kindprozess initialisieren (spawn-Kontext!)
    from multiprocessing import shared_memory

    import cupy as cp

    from concurrent.futures import ThreadPoolExecutor

    from nbody_kernel import (G_AU, NBodyCuda, pick_detect_device,
                              pick_devices)

    # Multi-GPU lohnt erst, wenn die Rechenlast den Barrier-Overhead
    # (~PCIe-Roundtrips pro Substep) klar uebersteigt — gemessen ab
    # ~30k Asteroiden. Darunter ist die beste Einzelkarte schneller.
    n_ast_total = int(np.count_nonzero(
        np.asarray(state["isAst"], dtype=np.uint8)))
    phys_devs = pick_devices() if n_ast_total >= 30000 \
        else pick_devices()[:1]
    sim = NBodyCuda(phys_devs)
    # Kollisions-/Bounce-Erkennung auf die zweitbeste f64-GPU auslagern:
    # sie laeuft dann UEBERLAPPT mit dem naechsten Kernel-Step (Pipeline).
    # Preis: Erkennungs-Ergebnisse von Sample k werden erst vor Step k+2
    # angewandt (1 Raster Versatz) — feiner als der Live-JS-Algo, der
    # Kollisionen einmal pro Frame (oft 1-2 Tage) aufloest. Ohne zweite
    # GPU laeuft die Analyse im selben Muster auf der Physik-GPU.
    det_dev = pick_detect_device(phys_devs, det_rank)
    ana_dev = det_dev if det_dev is not None else sim.device
    print(f"[film] physik auf gpus {phys_devs}, erkennung auf gpu "
          f"{ana_dev}" + (" (pipelined)" if det_dev is not None else
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
    st = sim.load_state(x, y, vx, vy, mass, vis, state["isAst"])
    n = len(x)
    collisions = 0
    dt_years = raster_days / 365.25

    with cp.cuda.Device(ana_dev):
        g_ast = cp.asarray(is_ast)
        prev_det = None          # voriges Sample (fuer den Merge-Sweep)
        g_vis_det = cp.asarray(vis)
        g_rr_det = cp.asarray(real_r.astype(np.float32))
    # vis/real_r aendern sich nur bei Merges/Kills — nur dann neu auf die
    # Erkennungs-GPU laden statt bei jedem Sample (spart 2 H2D/Sample).
    det_dirty = [False]

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

    def emit_event(a: int, b: int, new_mass: float, kind: int,
                   k_ev: int, ex: float, ey: float) -> None:
        # Merge/Kill/Bounce als Ereignis in den Event-Ring — Samples selbst
        # tragen nur noch Positionen (reines Punkte-Streaming). k_ev ist
        # der Sample-Zaehler der ANALYSE (Pipeline: Anwendung 1 spaeter).
        i = ev_count_val.value % ev_cap
        t_ev = t0_days + (k_ev + 1) * raster_days
        ev_buf[i * EV_BYTES:(i + 1) * EV_BYTES] = _struct.pack(
            "<dIIfIff", t_ev, a & 0xFFFFFFFF, b, new_mass, kind,
            float(ex), float(ey))
        ev_count_val.value += 1

    def analyze_merge(sx, sy, gx, gy, gvx, gvy, g_vis_a, g_rr_a):
        """Merge-Kandidaten + Runaways auf der Erkennungs-GPU bestimmen.
        Reine Analyse — mutiert nichts; Anwendung im Hauptloop."""
        nonlocal prev_det
        hit_pairs = []
        runaway_np = None
        m_alive = np.empty(0, dtype=np.int64)
        if True:
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
                    runaway = (v2 > 9.0 * vesc2) & g_ast & (g_vis_a != 0)
                    ridx = cp.flatnonzero(runaway)
                    if ridx.size:
                        runaway_np = cp.asnumpy(ridx)
            prev_det = (gx.copy(), gy.copy())
        pairs = [(int(m_alive[row]), int(j)) for row, j in hit_pairs]
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
        for mi, j in pairs:
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
                    shatter_t.value = t0_days + (k_ev + 1) * raster_days
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
            emit_event(a, b, float(m_ges), 0, k_ev, nx, ny)
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
                emit_event(0xFFFFFFFF, j, 0.0, 0, k_ev,
                           float(sx[j]), float(sy[j]))   # Numerik-Waechter
                collisions += 1
                changed = True
        if changed:
            coll_val.value = collisions
            det_dirty[0] = True

    BOUNCE_E = 0.6            # Restitution (wie BOUNCE_RESTITUTION im JS)
    bounce_count = 0

    def analyze_bounce(gx32, gy32, gvx32, gvy32, g_vis_a, g_rr_a):
        """Asteroid-x-Asteroid-Stoesse nach dem bisherigen Algo (Bounce,
        Restitution 0,6, Ein-Stoss pro Sample, Ueberlapp-Aufloesung 1,1x).
        Kandidaten + Swept-Check laufen auf der Erkennungs-GPU; Rueckgabe
        sind die Treffer-Paare (Anwendung als Deltas im Hauptloop)."""
        dt_y = raster_days / 365.25
        hits_host = None
        if True:
            # VORFILTER komplett in f32 (auf der f64-schwachen
            # Erkennungskarte ~30x schneller). Der Beruehrungsradius wird
            # um mehr als den f32-Rundungsfehler (~3e-7 AU bei 5 AU)
            # aufgeweitet — kein echter Treffer kann verloren gehen. Die
            # wenigen Kandidaten werden danach exakt in f64 nachgeprueft.
            gx = gx32
            gy = gy32
            gvx = gvx32
            gvy = gvy32
            F32_TOL = cp.float32(1e-6)
            g_alive = g_ast & (g_vis_a != 0)
            ai = cp.flatnonzero(g_alive)
            if int(ai.size) < 2:
                return None
            axp = gx[ai].astype(cp.float64)
            ayp = gy[ai].astype(cp.float64)
            sp = cp.hypot(gvx[ai], gvy[ai]).astype(cp.float64)
            h = max(1e-4, 2.0 * float(cp.percentile(sp, 95)) * dt_y)
            ix = cp.floor(axp / h).astype(cp.int64)
            iy = cp.floor(ayp / h).astype(cp.int64)

            def cell_key(cx_, cy_):
                return cx_ * cp.int64(73856093) ^ cy_ * cp.int64(19349663)

            key = cell_key(ix, iy)
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
                # Fenster BEIDSEITIG: [-dt, +dt] — der Live-Algo prueft
                # per CCD rueckwaerts ueber den Frame; nur vorwaerts
                # verpasste frontales Tunneling im letzten Kernel-Step.
                tmin = cp.clip(-(dpx * dvx_ + dpy * dvy_) /
                               cp.where(dv2 > 0, dv2, cp.float32(1.0)),
                               cp.float32(-dt_y), cp.float32(dt_y))
                cxm = dpx + dvx_ * tmin
                cym = dpy + dvy_ * tmin
                rsum = g_rr_a[pi] + g_rr_a[pj] + F32_TOL
                hitm = cxm * cxm + cym * cym <= rsum * rsum
                hidx = cp.flatnonzero(hitm)
                return pi[hidx], pj[hidx]

            # Kompakte Wolken (Injektion) haetten beim Voll-Materialisieren
            # aller Kandidaten zweistellige Millionen Paare x mehrere
            # Arrays (GBs VRAM-Peak, den der CuPy-Pool dauerhaft behielte).
            # Daher Chunk-Verarbeitung: expandieren + sweepen in Stuecken,
            # nur Treffer ueberleben. Physik identisch, Reihenfolge der
            # Treffer identisch (Offsets + Chunks aufsteigend).
            CHUNK = 4_000_000
            hit_i_parts = []
            hit_j_parts = []
            for ox, oy in ((0, 0), (1, 0), (0, 1), (1, 1), (1, -1)):
                k2 = cell_key(ix + ox, iy + oy)
                lo = cp.searchsorted(ks, k2, side="left")
                hi_r = cp.searchsorted(ks, k2, side="right")
                lens = hi_r - lo
                tot = int(lens.sum())
                if tot == 0:
                    continue
                cum = cp.cumsum(lens)
                starts = cum - lens
                for c0 in range(0, tot, CHUNK):
                    c1 = min(c0 + CHUNK, tot)
                    r = cp.arange(c0, c1, dtype=cp.int64)
                    rows = cp.searchsorted(cum, r, side="right")
                    cols = r - starts[rows] + lo[rows]
                    a = ai[rows]
                    b = ai[order[cols]]
                    keep = (a < b) if (ox == 0 and oy == 0) else (a != b)
                    hi_c, hj_c = sweep_hits(a[keep], b[keep])
                    if int(hi_c.size):
                        hit_i_parts.append(hi_c)
                        hit_j_parts.append(hj_c)
            if not hit_i_parts:
                return None
            # Exakte f64-Nachpruefung NUR der Vorfilter-Kandidaten
            # (wenige tausend statt Millionen Paare).
            ci = cp.concatenate(hit_i_parts)
            cj = cp.concatenate(hit_j_parts)
            x64i = gx[ci].astype(cp.float64)
            y64i = gy[ci].astype(cp.float64)
            x64j = gx[cj].astype(cp.float64)
            y64j = gy[cj].astype(cp.float64)
            dvx_ = gvx[cj].astype(cp.float64) - gvx[ci].astype(cp.float64)
            dvy_ = gvy[cj].astype(cp.float64) - gvy[ci].astype(cp.float64)
            dpx = x64j - x64i
            dpy = y64j - y64i
            dv2 = dvx_ * dvx_ + dvy_ * dvy_
            tmin = cp.clip(-(dpx * dvx_ + dpy * dvy_) /
                           cp.where(dv2 > 0, dv2, 1.0), -dt_y, dt_y)
            cxm = dpx + dvx_ * tmin
            cym = dpy + dvy_ * tmin
            rsum = g_rr_a[ci].astype(cp.float64) + \
                g_rr_a[cj].astype(cp.float64)
            hitm = cxm * cxm + cym * cym <= rsum * rsum
            hidx = cp.flatnonzero(hitm)
            if int(hidx.size) == 0:
                return None
            hits_host = (cp.asnumpy(ci[hidx]), cp.asnumpy(cj[hidx]))
        return hits_host

    def bounce_deltas(sample: np.ndarray, hits_host):
        """Host-Teil des Bounce-Algos: Ein-Stoss-Filter, Impuls und
        Ueberlapp-Push — wie im JS, aber als DELTAS (dx, dy, dvx, dvy),
        weil die Anwendung pipelined ein Sample spaeter erfolgt."""
        dt_y = raster_days / 365.25
        hi, hj = hits_host
        x = sample[0:n].astype(np.float64)
        y = sample[n:2 * n].astype(np.float64)
        vxa = sample[2 * n:3 * n].astype(np.float64)
        vya = sample[3 * n:4 * n].astype(np.float64)
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
        dpx = x[hj] - x[hi]
        dpy = y[hj] - y[hi]
        dvx_ = vxa[hj] - vxa[hi]
        dvy_ = vya[hj] - vya[hi]
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
        pdx = x[hj] - x[hi]
        pdy = y[hj] - y[hi]
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

    def analyze_batch(outs: np.ndarray, k0: int) -> list:
        # Alle K Samples des Batches sequenziell analysieren (der
        # Pipeline-Thread hat sie als Host-Kopien) — Ergebnisse werden
        # gesammelt im Hauptloop angewandt.
        return [analyze_sample(outs[i], k0 + i) for i in range(len(outs))]

    def analyze_sample(out_np: np.ndarray, k_ev: int) -> dict:
        """Komplette Erkennung eines Samples auf der Erkennungs-GPU —
        laeuft im Pipeline-Thread, mutiert nichts. vis/mass/real_r werden
        hier nur GELESEN; der Hauptloop mutiert sie erst nach dem
        Einsammeln des Ergebnisses (keine Gleichzeitigkeit)."""
        nonlocal g_vis_det, g_rr_det
        with cp.cuda.Device(ana_dev):
            gx = cp.asarray(out_np[0:n])
            gy = cp.asarray(out_np[n:2 * n])
            gvx = cp.asarray(out_np[2 * n:3 * n])
            gvy = cp.asarray(out_np[3 * n:4 * n])
            if det_dirty[0]:
                g_vis_det = cp.asarray(vis)
                g_rr_det = cp.asarray(real_r.astype(np.float32))
                det_dirty[0] = False
            g_vis_a = g_vis_det
            g_rr_a = g_rr_det
            pairs, runaway_np = analyze_merge(
                out_np[0:n], out_np[n:2 * n],
                gx, gy, gvx, gvy, g_vis_a, g_rr_a)
            hits = analyze_bounce(gx, gy, gvx, gvy, g_vis_a, g_rr_a) \
                if ast_bounce else None
        bounce = bounce_deltas(out_np, hits) if hits is not None else None
        return {"k": k_ev, "sample": out_np, "pairs": pairs,
                "runaways": runaway_np, "bounce": bounce}

    def apply_analysis(res: dict) -> None:
        apply_merges(res["sample"], res["pairs"], res["runaways"], res["k"])
        if res["bounce"] is not None:
            apply_bounce(res["bounce"], res["k"], res["sample"])

    executor = ThreadPoolExecutor(max_workers=1)
    future = None
    # Batch-Groesse: K Raster pro Kernel-Launch. Erkennungs-Ergebnisse
    # werden nach dem Batch angewandt — mit K=8 (4 Tage) bleibt der
    # Versatz in der Groessenordnung grosser Live-Frames, halbiert aber
    # den Launch-/Pipeline-Overhead nochmals.
    K = 8
    try:
        while running_val.value:
            if shatter_flag is not None and shatter_flag.value:
                break        # Zerbersten erkannt — Client uebernimmt
            # Pipeline: Erkennungs-Ergebnisse des VORIGEN Batches
            # einsammeln und anwenden, bevor der naechste Launch startet.
            if future is not None:
                for res in future.result():
                    apply_analysis(res)
                future = None
            if dump_req_val.value == 1:
                dump_state()
                dump_req_val.value = 2
            # Ueberschreib-Schutz: Slot (k - capacity) wird gleich
            # ueberschrieben — er muss hinter dem Player-Playhead liegen.
            ph_abs = int((playhead_val.value - t0_days) / raster_days)
            # 70% des Rings als Vorlauf, 30% bleiben Rueckspul-Historie —
            # sonst frisst die Eviction sich bis an den Playhead heran
            # und der Player reitet stotternd auf der Abbruchkante.
            if k + K - ph_abs >= int(capacity * 0.7):
                time.sleep(0.01)
                continue
            outs = sim.step_batch(st, dt_years, K)
            # Erkennung laeuft UEBERLAPPT auf der Erkennungs-GPU, waehrend
            # die Physik-GPUs schon den naechsten Batch rechnen.
            future = executor.submit(analyze_batch, outs, k)
            # Reines Punkte-Streaming: nur x|y (8 Bytes/Koerper) — alles
            # andere (Masse/Sichtbarkeit) laeuft als Ereignis.
            for i in range(K):
                slot = (k + i) % capacity
                buf[slot * sample_bytes:(slot + 1) * sample_bytes] = \
                    outs[i][0:2 * n].tobytes()
            k += K
            head_val.value = k
            if ast_bounce and k % 500 == 0 and bounce_count:
                print(f"[film] {bounce_count} asti-bounces nach "
                      f"{k} samples", flush=True)
    finally:
        # Letzter Zustand fuer die Engine-Uebergabe, dann sauber schliessen.
        # Nach einem Zerbersten ist der SHATTER-Dump massgeblich — dann
        # weder ausstehende Analysen anwenden noch neu dumpen.
        try:
            executor.shutdown(wait=False)
            if shatter_flag is not None and shatter_flag.value:
                # Shatter-Dump liegt bereits — nur als bereit markieren,
                # damit der FILM_STOP-Pfad des Servers nicht wartet.
                dump_req_val.value = 2
            else:
                if future is not None:
                    for res in future.result():
                        apply_analysis(res)
                dump_state()
                dump_req_val.value = 2
        except Exception:
            pass
        shm.close()
        ev_shm.close()
        dump_shm.close()
