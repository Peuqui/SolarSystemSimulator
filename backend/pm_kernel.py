"""Particle-Mesh-Kraftberechnung (isolierte Randbedingung, 2D-Ebene).

Ersetzt die O(N²)-all-pairs-Summe des selbstgravitierenden Kernels durch
eine Faltung ueber ein Gitter — O(N log N) statt O(N²). Erster Schritt der
PM/TreePM-Roadmap (siehe TODO.md); hier NUR die Kraft, gegen all-pairs in
`test_pm.py` validiert, bevor sie in `film_producer.py` eingebunden wird.

WICHTIG — 3D-Gravitation in der 2D-Ebene:
Der Simulator ist 2D (x, y), aber die Kraft faellt mit 1/r² (der all-pairs-
Kernel rechnet a = G·m·(r_j−r_i)/(|Δ|²+ε²)^1,5). Die Faltungs-Green-Funktion
ist deshalb die geglaettete 3D-Kraft K(Δ) = −G·Δ/(|Δ|²+ε²)^1,5, NICHT der 2D-
ln(r)-Kern. So stimmt PM grossraeumig exakt mit all-pairs ueberein.

ISOLIERTE Randbedingung (Hockney/James, kein periodisches Universum):
Das Rechengitter wird auf 2N×2N genullt (Zero-Padding), die Masse liegt nur
im N×N-Quadranten, und der Green-Kern wird so ueber die 2N-Periode gelegt,
dass die umgewickelten Kopien in den Nullbereich fallen. Nur der [0:N,0:N]-
Ausschnitt der Faltung ist gueltig. Eine naive periodische FFT wuerde die
periodischen Bilder umwickeln und am Rand falsche Kraefte liefern.

Faltung DIREKT mit der Kraft-Green-Funktion (zwei Komponenten K_x, K_y)
statt ueber ein Potential mit anschliessender Differenziation: reproduziert
die geglaettete Kraft treuer und vermeidet den zusaetzlichen Gitter-Fehler
der Ableitung.
"""
from __future__ import annotations

import time

import cupy as cp
import numpy as np

import gpu_bench
from nbody_kernel import G_AU


def build_force_kernels(grid_n: int, h: float, eps2: float,
                        G: float = G_AU, dtype=cp.float64) -> tuple:
    """FFT der beiden Kraft-Green-Funktionen auf dem 2N×2N-Padding-Gitter.

    Einmal vorab bauen, dann in jeder Kraftauswertung wiederverwenden — die
    Kerne haengen nur an Gittergroesse, Zellweite und Softening, nicht am
    Zustand. Rueckgabe: (FKx, FKy) als rfft2-Spektren (2N, N+1).
    """
    n_pad = 2 * grid_n
    # off[m] bildet den FFT-Index m auf den ECHTEN Versatz ab: 0..N-1 bleibt,
    # N..2N-1 wird negativ (−N..−1). So liegt Versatz 0 auf Index 0, und die
    # negativen Versaetze fuellen die obere Haelfte — genau die Anordnung,
    # die die zirkulare FFT-Faltung mit dem Zero-Padding braucht.
    off = cp.arange(n_pad, dtype=cp.float64)
    off = cp.where(off < grid_n, off, off - n_pad)
    dx = (off * h)[:, None]           # (2N, 1) — Versatz entlang Achse 0 (x)
    dy = (off * h)[None, :]           # (1, 2N) — Versatz entlang Achse 1 (y)
    r2 = dx * dx + dy * dy + eps2
    inv_r3 = 1.0 / (r2 * cp.sqrt(r2))  # 1/(|Δ|²+ε²)^1,5
    # K(Δ) = −G·Δ/(...)^1,5. Das Minus macht die Faltung (ρ ⊛ K)[p] direkt
    # zur Beschleunigung a(p) = Σ_j m_j·G·(r_j−p)/... (nachgerechnet: der
    # Versatz Δ = p − q = Feld − Quelle, also Δ_x = x_p − x_q, und
    # −G·Δ_x/... = G·(x_q − x_p)/... = Zug zur Quelle). Bei Δ=0 ist Δ_x=0 →
    # K_x=0: keine Selbstkraft.
    # Die Kerne selbst IMMER in f64 aufbauen — 1/(r²+ε²)^1,5 ueberstreicht
    # ueber das ganze Gitter viele Groessenordnungen, und der Aufbau kostet
    # nur einmal je h. Erst das Spektrum geht in die Rechengenauigkeit.
    kx = (-G) * dx * inv_r3
    ky = (-G) * dy * inv_r3
    return (cp.fft.rfft2(kx.astype(dtype, copy=False)),
            cp.fft.rfft2(ky.astype(dtype, copy=False)))


# Die CIC-Zellindizes samt Gewichten — identisch fuer Deposit und Gather.
# Steht als C-Fragment nur EINMAL da: laufen die beiden Schemata
# auseinander, entsteht eine Selbstkraft (jedes Teilchen zoege an seinem
# eigenen Beitrag im Gitter).
_CIC_ECKEN = r"""
    double fx = (px - x0) * inv_h;
    double fy = (py - y0) * inv_h;
    // zi/zj statt i/j: `i` ist in ElementwiseKernel die Laufvariable der
    // erzeugten Schleife und darf nicht verdeckt werden.
    int zi = (int)floor(fx);
    int zj = (int)floor(fy);
    // In [0, N-2] halten, damit zi+1 / zj+1 gueltig bleiben. Bei
    // ausreichendem Rand (siehe grid_fuer) greift die Klemmung nicht.
    zi = min(max(zi, 0), grid_n - 2);
    zj = min(max(zj, 0), grid_n - 2);
    double tx = fx - zi;
    double ty = fy - zj;
    double w00 = (1.0 - tx) * (1.0 - ty);
    double w10 = tx * (1.0 - ty);
    double w01 = (1.0 - tx) * ty;
    double w11 = tx * ty;
    int b = zi * grid_n + zj;
"""

# Deposit als EIN Kernel mit atomicAdd statt concatenate + bincount ueber
# 4N Eintraege: das sparte bei 1M Massen acht temporaere Arrays zu je 8 MB.
# Konflikte sind selten — das Gitter ist so gewaehlt, dass rund ein
# Teilchen je Zelle liegt (siehe load_state).
_cic_deposit_kernel = cp.ElementwiseKernel(
    "float64 px, float64 py, float64 masse, int32 grid_n, "
    "float64 x0, float64 y0, float64 inv_h",
    "raw T rho",
    _CIC_ECKEN + """
    atomicAdd(&rho[b], (T)(masse * w00));
    atomicAdd(&rho[b + grid_n], (T)(masse * w10));
    atomicAdd(&rho[b + 1], (T)(masse * w01));
    atomicAdd(&rho[b + grid_n + 1], (T)(masse * w11));
    """,
    "cic_deposit")

# Gather fuer BEIDE Kraftkomponenten in einem Zug. Vorher lief je Komponente
# ein eigener Aufruf, der Zellindex und Gewichte neu ausrechnete und vier
# Fancy-Index-Zugriffe plus ein Dutzend Temporaerarrays erzeugte — gemessen
# 2,37 ms fuer 1M Massen, der groesste Einzelposten des Substeps.
_cic_gather_kernel = cp.ElementwiseKernel(
    "raw T fx_feld, raw T fy_feld, float64 px, float64 py, int32 grid_n, "
    "float64 x0, float64 y0, float64 inv_h",
    "float64 ax, float64 ay",
    _CIC_ECKEN + """
    ax = w00 * fx_feld[b] + w10 * fx_feld[b + grid_n]
       + w01 * fx_feld[b + 1] + w11 * fx_feld[b + grid_n + 1];
    ay = w00 * fy_feld[b] + w10 * fy_feld[b + grid_n]
       + w01 * fy_feld[b + 1] + w11 * fy_feld[b + grid_n + 1];
    """,
    "cic_gather2")


def _cic_deposit(gx, gy, gm, grid_n, x0, y0, h, dtype=cp.float64):
    """Cloud-in-Cell: jede Masse auf die 4 umliegenden Gitterpunkte
    verteilt. Rueckgabe: Massengitter (N, N) — Summe = Gesamtmasse."""
    rho = cp.zeros(grid_n * grid_n, dtype)
    _cic_deposit_kernel(gx, gy, gm, np.int32(grid_n),
                        float(x0), float(y0), 1.0 / float(h), rho)
    return rho.reshape(grid_n, grid_n)


def grid_fuer(gx, gy, grid_n: int, rand_zellen: float = 3.0):
    """Gitter-Geometrie aus der Punktwolke: Ursprung (x0, y0) und Zellweite
    h, sodass alle Punkte mit `rand_zellen` Zellen Rand hineinpassen.

    Gibt (x0, y0, h) zurueck. Im Betrieb ruft `NBodyPM._geometrie` das
    einmal je Batch auf (die vier min/max sind synchrone Transfers), fuers
    Kraft-Testen leitet es sich aus der jeweiligen Wolke ab."""
    xmin = float(cp.min(gx))
    xmax = float(cp.max(gx))
    ymin = float(cp.min(gy))
    ymax = float(cp.max(gy))
    spanne = max(xmax - xmin, ymax - ymin, 1e-30)
    # Rand auf beiden Seiten -> nutzbare Zellen = N - 2*rand
    h = spanne / (grid_n - 2 * rand_zellen)
    x0 = xmin - rand_zellen * h
    y0 = ymin - rand_zellen * h
    return x0, y0, h


def pm_force_field(gx, gy, gm, grid_n, x0, y0, h, eps2,
                   G: float = G_AU, kernels=None, dtype=cp.float64):
    """Das Beschleunigungs-Feld (ax_grid, ay_grid) auf dem N×N-Gitter.

    Getrennt vom Gather, damit MEHRERE Teilchenmengen aus demselben Feld
    schoepfen koennen — die Massen bauen das Feld, Massen UND masselose
    Tracer lesen es an ihren Positionen ab (die Tracer wirken nicht
    zurueck, tauchen also nicht im Deposit auf).

    `dtype` ist die Rechengenauigkeit der FALTUNG (f32 im Betrieb, siehe
    NBodyPM.kraft_f32); der Zustand bleibt davon unberuehrt."""
    if kernels is None:
        kernels = build_force_kernels(grid_n, h, eps2, G, dtype)
    fkx, fky = kernels
    n_pad = 2 * grid_n

    rho_pad = cp.zeros((n_pad, n_pad), dtype)
    rho_pad[:grid_n, :grid_n] = _cic_deposit(gx, gy, gm, grid_n,
                                             x0, y0, h, dtype)
    fr = cp.fft.rfft2(rho_pad)
    ax_grid = cp.fft.irfft2(fr * fkx, s=(n_pad, n_pad))[:grid_n, :grid_n]
    ay_grid = cp.fft.irfft2(fr * fky, s=(n_pad, n_pad))[:grid_n, :grid_n]
    # irfft2 liefert bei f32-Eingang f32 zurueck; ascontiguousarray, weil
    # der Gather-Kernel linear indiziert (der Ausschnitt ist ein View).
    return (cp.ascontiguousarray(ax_grid), cp.ascontiguousarray(ay_grid))


def gather_accel(ax_grid, ay_grid, px, py, grid_n, x0, y0, h):
    """Feld an Teilchenpositionen (px, py) ablesen — fuer Massen und
    Tracer gleich. BEIDE Komponenten in einem Kernel: Zellindex und
    Gewichte werden nur einmal berechnet und das Feld nur einmal
    indiziert."""
    ax = cp.empty(px.shape, cp.float64)
    ay = cp.empty(px.shape, cp.float64)
    _cic_gather_kernel(ax_grid, ay_grid, px, py, np.int32(grid_n),
                       float(x0), float(y0), 1.0 / float(h), ax, ay)
    return ax, ay


def pm_accelerations(gx, gy, gm, grid_n, x0, y0, h, eps2,
                     G: float = G_AU, kernels=None):
    """Beschleunigung (ax, ay) je Masse via Particle-Mesh — Feld bauen und
    an denselben Positionen ablesen. Bequemlichkeit fuer Kraft-Tests."""
    ax_grid, ay_grid = pm_force_field(gx, gy, gm, grid_n, x0, y0, h, eps2,
                                      G, kernels)
    return gather_accel(ax_grid, ay_grid, gx, gy, grid_n, x0, y0, h)


class NBodyPM:
    """Selbstgravitierender Verbund via Particle-Mesh — auf EINER Karte.

    Aussenverhalten wie `NBodySelfGrav` (`load_state` / `step_batch` /
    `export_f64`), damit der Film-Producer beide Kernel gleich aufruft. Die
    Kraft kommt aber aus einer FFT-Faltung ueber ein Gitter (O(N log N))
    statt aus der all-pairs-Summe (O(N²)).

    WARUM EINE Karte: Das Gitter ueber die Karten zu teilen hiesse, es pro
    Schritt ueber die ×4-PCIe-Links zu reduzieren/broadcasten — ~1 s gegen
    ~6 ms Rechnung, auf Hardware ohne NVLink ein Verlust. PM auf einer Karte
    ist gegen all-pairs auf fuenf Karten trotzdem um Groessenordnungen
    schneller. Massen UND Tracer schoepfen aus DEMSELBEN Kraftfeld auf
    dieser Karte.

    ADAPTIVES Gitter: Jeder Schritt leitet Ursprung und Zellweite aus der
    aktuellen Massen-Ausdehnung ab (feste Zellenzahl `grid_n`). So folgt das
    Gitter der expandierenden Wolke — mitbewegte Koordinaten, kein
    Rausfallen. Das Softening ist `softening_zellen` × Zellweite und waechst
    damit mit; unter ~1 Zelle kann PM nicht aufloesen (das leistet spaeter
    der Baum in TreePM).
    """

    def __init__(self, devices, softening_au: float = 0.0,
                 kraft_f32: bool = True, grid_n: int | None = None,
                 softening_zellen: float = 1.5, rand_zellen: float = 4.0):
        # `devices` darf eine Liste sein (Producer uebergibt alle) — PM
        # nimmt die erste. `softening_au` wird angenommen (API-Kompat) und
        # nur als Untergrenze verwendet: das Gitter bestimmt das Softening.
        #
        # grid_n=None: aus der Massenzahl ableiten (~√N → rund 1 Teilchen je
        # Zelle, Zellweite ≈ Teilchenabstand ≈ Softening). Ein festes,
        # feines Gitter waere bei wenigen Teilchen schrotrausch-dominiert
        # (viele leere Zellen) und wuerde das Softening unter den
        # Teilchenabstand druecken — genau die Zweikoerper-Streuung, die es
        # zu daempfen gilt. Ein fester Wert bleibt fuers Testen erzwingbar.
        if isinstance(devices, int):
            devices = [devices]
        self.device = devices[0]
        self.devices = [self.device]
        self._grid_n_fest = None if grid_n is None else int(grid_n)
        self.grid_n = self._grid_n_fest or 512   # bis load_state N kennt
        self.softening_zellen = float(softening_zellen)
        self.rand_zellen = float(rand_zellen)
        self.softening_floor = float(softening_au)
        # Rechengenauigkeit der FALTUNG. Der Zustand (x, y, v) bleibt f64 —
        # wie in Kernel A, wo dieselbe Trennung gemessen ist. Das Gitter
        # traegt ohnehin nur ~3 gueltige Stellen (Diskretisierungsfehler der
        # CIC-Interpolation), da sind die 7 Stellen von f32 reichlich; f64
        # kostet dafuer den doppelten Speicherverkehr in Deposit, FFT und
        # Gather — und genau die sind der Engpass, nicht die Rechenwerke.
        self.kraft_f32 = bool(kraft_f32)
        self.feld_dtype = cp.float32 if self.kraft_f32 else cp.float64
        self._kernels = None      # (FKx, FKy) — gecacht, neu bei h-Wechsel
        self._kernel_h = None
        self._host = None         # pinned Download-Puffer (siehe _host_puffer)

    def name(self) -> str:
        with cp.cuda.Device(self.device):
            n = cp.cuda.runtime.getDeviceProperties(self.device)["name"]
        return n.decode() + " (PM)"

    def _geometrie(self, st) -> tuple:
        """Gitter-Geometrie (x0, y0, h, eps2) aus der aktuellen Massen-
        Ausdehnung — und die passenden Green-Kerne, falls sich h geaendert
        hat.

        KOSTET VIER SYNCHRONE GPU->CPU-TRANSFERS (min/max je Achse in
        `grid_fuer`). Jeder davon laesst die CPU warten, bis alle offenen
        Kernel durch sind, und in dieser Zeit startet niemand neue — genau
        das war der Grund fuer `step=93 %` bei nur 55 % GPU. Darum EINMAL
        je Batch, nicht je Substep.

        Dass das Gitter waehrend eines Batches steht, ist unkritisch: Bei
        Raster 10 Tagen und K=8 vergehen 80 Tage, in denen sich der
        Hubble-Fluss um rund 1 AE ausdehnt — gegen eine Zellweite von
        ~140 AE und 4 Zellen Rand. Die Kerne MUESSEN ohnehin zu genau dem
        h passen, mit dem gefaltet wird; ein Gitter, das mitten im Batch
        wandert, waere also selbst dann falsch, wenn es nichts kostete.
        """
        x0, y0, h = grid_fuer(st["x"], st["y"], self.grid_n, self.rand_zellen)
        eps = max(self.softening_zellen * h, self.softening_floor)
        eps2 = eps * eps
        # Kerne haengen nur an (h, eps) — bei nahezu gleichem h wiederverwenden.
        if self._kernel_h is None or abs(h - self._kernel_h) > 1e-6 * h:
            self._kernels = build_force_kernels(self.grid_n, h, eps2,
                                                dtype=self.feld_dtype)
            self._kernel_h = h
        return x0, y0, h, eps2

    def _accel_into(self, st, geo: tuple):
        """Kraftfeld aus den Massen bauen und an Massen- UND Tracer-
        Positionen ablesen. Setzt st['ax','ay','tax','tay']. Die Geometrie
        kommt von aussen (`_geometrie`) — hier wird nicht gemessen."""
        gx, gy, gm = st["x"], st["y"], st["m"]
        x0, y0, h, eps2 = geo
        axg, ayg = pm_force_field(gx, gy, gm, self.grid_n, x0, y0, h, eps2,
                                  kernels=self._kernels,
                                  dtype=self.feld_dtype)
        st["ax"], st["ay"] = gather_accel(axg, ayg, gx, gy,
                                          self.grid_n, x0, y0, h)
        if st["T"]:
            st["tax"], st["tay"] = gather_accel(axg, ayg, st["tx"], st["ty"],
                                                self.grid_n, x0, y0, h)

    def load_state(self, x, y, vx, vy, mass, *_egal,
                   tracer=None, **_auch_egal) -> dict:
        # Gitteraufloesung aus der Massenzahl: naechste Zweierpotenz zu √N,
        # gedeckelt auf [256, 2048]. So liegt rund ein Teilchen je Zelle und
        # die Zellweite trifft den Teilchenabstand. Der Deckel haelt die FFT
        # bezahlbar (2048² gepaddet = 4096²).
        if self._grid_n_fest is None:
            import math
            g = 1 << max(8, round(math.log2(max(math.isqrt(len(x)), 1))))
            self.grid_n = int(min(g, 2048))
            self._kernel_h = None      # Neu-Cache erzwingen
        with cp.cuda.Device(self.device):
            st = {
                "x": cp.asarray(x, cp.float64),
                "y": cp.asarray(y, cp.float64),
                "vx": cp.asarray(vx, cp.float64),
                "vy": cp.asarray(vy, cp.float64),
                "m": cp.asarray(mass, cp.float64),
                "N": len(x),
            }
            if tracer is not None and len(tracer[0]):
                st["tx"] = cp.asarray(tracer[0], cp.float64)
                st["ty"] = cp.asarray(tracer[1], cp.float64)
                st["tvx"] = cp.asarray(tracer[2], cp.float64)
                st["tvy"] = cp.asarray(tracer[3], cp.float64)
                st["T"] = len(tracer[0])
            else:
                st["tx"] = st["ty"] = st["tvx"] = st["tvy"] = None
                st["T"] = 0
            # Anfangsbeschleunigung fuer das erste Kick
            self._accel_into(st, self._geometrie(st))
        return st

    def step_batch(self, st: dict, dt_years: float, steps: int) -> np.ndarray:
        """`steps` Leapfrog-Schritte (Kick-Drift-Kick). Rueckgabe wie
        NBodySelfGrav: f32 (steps, 4N+2T) [x|y|vx|vy | tx|ty]."""
        if steps < 1:
            raise ValueError(f"steps muss >= 1 sein: {steps}")
        n, t = st["N"], st["T"]
        dt = float(dt_years)
        hdt = 0.5 * dt
        # Ergebnisse AUF DER GPU sammeln und in EINEM Transfer holen. Frueher
        # lief pro Substep je ein cp.asnumpy (6 Stueck) -> 6*steps (=48 bei K=8)
        # synchrone GPU->CPU-Transfers je Batch, jeder mit implizitem Device-
        # Sync. CUDAs Default-Sync ist Spin-Wait: verbrennt einen CPU-Kern bei
        # 100%, waehrend die GPU zwischen den winzigen Schritten idlet (gemessen
        # step=96%, GPU nur ~40%). Das war der Produktions-Flaschenhals.
        with cp.cuda.Device(self.device):
            # raus_gpu MUSS auf self.device liegen (nicht dem Default-Device),
            # sonst laufen die Slice-Assigns cross-device und racen mit asnumpy.
            raus_gpu = cp.empty((steps, 4 * n + 2 * t), cp.float32)
            # Gitter EINMAL je Batch (siehe `_geometrie`): die vier
            # min/max-Transfers dort sind der zweite Sync-Posten neben dem
            # asnumpy — je Substep gemessen kosteten sie mehr als die
            # Faltung selbst.
            geo = self._geometrie(st)
            for s in range(steps):
                st["vx"] += hdt * st["ax"]
                st["vy"] += hdt * st["ay"]
                st["x"] += dt * st["vx"]
                st["y"] += dt * st["vy"]
                if t:
                    st["tvx"] += hdt * st["tax"]
                    st["tvy"] += hdt * st["tay"]
                    st["tx"] += dt * st["tvx"]
                    st["ty"] += dt * st["tvy"]
                self._accel_into(st, geo)
                st["vx"] += hdt * st["ax"]
                st["vy"] += hdt * st["ay"]
                if t:
                    st["tvx"] += hdt * st["tax"]
                    st["tvy"] += hdt * st["tay"]
                # dtype-Cast f64->f32 passiert implizit beim Slice-Assign (GPU-
                # Kernel, kein Sync); der einzige Sync ist das asnumpy am Ende.
                raus_gpu[s, 0:n] = st["x"]
                raus_gpu[s, n:2 * n] = st["y"]
                raus_gpu[s, 2 * n:3 * n] = st["vx"]
                raus_gpu[s, 3 * n:4 * n] = st["vy"]
                if t:
                    raus_gpu[s, 4 * n:4 * n + t] = st["tx"]
                    raus_gpu[s, 4 * n + t:4 * n + 2 * t] = st["ty"]
            raus = self._host_puffer(raus_gpu.shape)
            raus_gpu.get(out=raus)        # EIN Transfer statt 6*steps
        return raus

    def _host_puffer(self, form: tuple) -> np.ndarray:
        """Wiederverwendeter PINNED Host-Puffer fuer den Batch-Download.

        Ueber einen pageable numpy-Puffer (wie `cp.asnumpy` ihn anlegt) muss
        der Treiber zusaetzlich durch einen eigenen Staging-Bereich kopieren.
        Gemessen an 152 MB je Batch ueber den ×4-Link: 70,4 ms pageable
        gegen 47,3 ms pinned (2,12 gegen 3,15 GB/s) — bei 89 ms Batchzeit
        der groesste Einzelposten ueberhaupt.

        ACHTUNG: Der Puffer wird wiederverwendet, das Ergebnis gilt also nur
        BIS ZUM NAECHSTEN `step_batch`. Der Film-Producer baut es im
        selbstgravitierenden Pfad sofort in ein eigenes Array um; wer es
        laenger halten oder nebenlaeufig auswerten will (wie der alte Kernel
        seine Erkennung), muss vorher kopieren.
        """
        if self._host is None or self._host.shape != form:
            roh = cp.cuda.alloc_pinned_memory(
                int(np.prod(form)) * np.dtype(np.float32).itemsize)
            self._host = np.frombuffer(
                roh, np.float32, int(np.prod(form))).reshape(form)
        return self._host

    def export_f64(self, st: dict) -> np.ndarray:
        """Exakter f64-Zustand [x|y|vx|vy] (4N), Massen nur (keine Tracer,
        keine Verschmelzungen). Gleiche Form wie NBodySelfGrav."""
        n = st["N"]
        raus = np.empty(4 * n, dtype="<f8")
        with cp.cuda.Device(self.device):
            raus[0:n] = cp.asnumpy(st["x"])
            raus[n:2 * n] = cp.asnumpy(st["y"])
            raus[2 * n:3 * n] = cp.asnumpy(st["vx"])
            raus[3 * n:4 * n] = cp.asnumpy(st["vy"])
        return raus


# --- Kartenwahl fuer PM: eigener Mikro-Benchmark -------------------------
# `selfgrav_kernel.miss_gewicht` (all-pairs, 4096 Koerper) taugt fuer PM
# NICHT: bei so kleinem N ist die Last compute-bound, wo die RTX 8000 knapp
# vorn liegt. PM ist bei grossem N FFT- und bandbreiten-bound — da gewinnt
# die V100 (HBM2). Gemessen (Schritte/s je Kraftauswertung):
#     N        RTX 8000   V100
#   200.000       190      237   (V100 ×1,25)
#   1.000.000      41       54   (V100 ×1,31)
# Also eine EIGENE Last, gross genug, dass sie im richtigen Regime misst und
# baugleiche Karten gleich rankt (kurze Laeufe verrauschen — siehe die
# 7-%-Streuung zweier baugleicher RTX beim all-pairs-Proxy).
KALIBRIER_PM_N = 200_000
KALIBRIER_PM_SCHRITTE = 20
KALIBRIER_PM_RUNDEN = 3


def miss_gewicht_pm(device: int) -> float:
    """PM-Durchsatz einer Karte (Schritte/s), gemessen im echten Betriebspfad
    (`NBodyPM.step_batch`). Nur als VERHAELTNIS zwischen Karten aussagekraeftig.
    MEHRERE Runden, davon die beste — Stoerungen (kalte Karte, fremde Last)
    wirken nur nach unten, deshalb ist das Maximum der robuste Schaetzer."""
    n = KALIBRIER_PM_N
    rng = np.random.default_rng(0)
    r = 60000.0 * np.sqrt(rng.uniform(0, 1, n))
    th = rng.uniform(0, 2 * np.pi, n)
    x = r * np.cos(th)
    y = r * np.sin(th)
    null = np.zeros(n)
    m = np.full(n, 5.6e9 / n)
    kern = NBodyPM([device], softening_au=0.0)
    st = kern.load_state(x, y, null, null, m)
    kern.step_batch(st, 1e-6, 3)          # warmlaufen: FFT-Plan, Takt hoch
    cp.cuda.Device(device).synchronize()
    beste = 0.0
    for _ in range(KALIBRIER_PM_RUNDEN):
        t0 = time.perf_counter()
        kern.step_batch(st, 1e-6, KALIBRIER_PM_SCHRITTE)
        cp.cuda.Device(device).synchronize()
        dauer = time.perf_counter() - t0
        beste = max(beste, KALIBRIER_PM_SCHRITTE / max(dauer, 1e-9))
    return beste


def waehle_karte_pm(devices: list[int]) -> int:
    """Die schnellste EINE Karte fuer den PM-Pfad — aus dem persistenten
    Cache oder frisch gemessen. Eine Karte: trivial (kein Benchmark noetig).
    Mehrere: die mit dem hoechsten gemessenen PM-Score."""
    devices = list(devices)
    if len(devices) == 1:
        return devices[0]
    scores = gpu_bench.hole_gewichte("pm_fft", miss_gewicht_pm, devices)
    return max(scores, key=scores.get)
