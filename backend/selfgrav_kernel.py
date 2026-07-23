"""Selbstgravitierende Massen auf beliebig vielen GPUs (Kernel A).

Zweck: kosmische Strukturbildung — Filamente, Knoten und Voids, die aus
der Simulation ENTSTEHEN. Dafuer muss jede Masse jede andere spueren,
und davon braucht es zehntausende bis hunderttausende.

Der Kernel in `nbody_kernel.py` kann das nicht und soll es nicht: Er ist
fuer wenige Massen und enge Begegnungen gebaut (f64, adaptive Feinschleife,
Bahnen auf der Messerschneide). Seine Massen liegen im Shared Memory
(`M_MAX = 64`) und rechnen massiv x massiv seriell auf Thread 0. Dieser
Kernel tritt DANEBEN, nicht an seine Stelle — zwei physikalische Fragen,
zwei Antworten.

Was hier anders ist
-------------------
* **Tiling statt Shared-Memory-Tabelle** (Nyland/Harris): Jeder Thread
  besitzt einen Zielkoerper, der Block laedt Quellen kachelweise
  kooperativ. Shared Memory = blockDim * 3 Werte, UNABHAENGIG von n.
  Damit faellt jede feste Obergrenze fuer die Koerperzahl weg.
* **Gemischte Genauigkeit**, gemessen statt gewaehlt: Der ZUSTAND laeuft
  f64, die KRAFTSCHLEIFE f32. Entscheidend ist die Akkumulation —
  `v += a*dt/2` addiert bei v ~ 1680 AU/a Inkremente von ~0,2, und die
  f32-Aufloesung liegt dort schon bei 1e-4. Die Kraftsumme dagegen
  vertraegt f32 muehelos (Energiefehler 7,9e-5 gegen 7,4e-5 bei voller
  f64-Rechnung, NumPy-Referenz 4,6e-5). So rechnen ALLE Karten
  dasselbe, und die in f64 auf 1/32 gebremsten RTX 8000 sind
  vollwertig. `kraft_f32=False` schaltet auf durchgaengiges f64.
* **Plummer-Softening.** Ohne dominiert Zweikoerper-Streuung das
  Ergebnis. Damit gibt es keine engen Begegnungen mehr, und ein FESTER
  Zeitschritt genuegt — die halbe Komplexitaet des alten Kernels
  (Klassifikation, private Feinschleife, hierarchische Zeitschritte)
  entfaellt ersatzlos.
* **Newtons drittes Gesetz wird NICHT ausgenutzt.** Es spart die halbe
  Rechnung, erzwingt aber Abstimmung zwischen Threads und macht das
  Ergebnis von der Ausfuehrungsreihenfolge abhaengig. Doppelt rechnen ist
  auf der GPU billiger als sich abzustimmen — und bleibt deterministisch.

Multi-GPU
---------
Die Koerper werden nach GEMESSENER Rechenleistung (`miss_gewicht`) in
zusammenhaengende Segmente geteilt; Karte g integriert nur ihr Segment,
sieht aber alle Quellen. Nach jedem Drift tauschen die Karten ihre
Segmente ueber gemappten Host-Speicher aus (kein NVLink im Zielsystem,
alle Karten haengen ueber PHB an x4-Adaptern).

Die Barrier und der Blick auf gemappten Host-Speicher stehen in
`gpu_verbund.py` — sie sind mit `nbody_kernel.py` geteilt.

**Determinismus ueber Kartenzahlen:** Jeder Zielkoerper summiert ueber
alle Quellen in derselben Kachelreihenfolge, egal welche Karte ihn
besitzt. Ein Lauf auf einer Karte und einer auf fuenf liefern darum
bitgleiche Ergebnisse — `test_selfgrav.py` prueft genau das.
"""
from __future__ import annotations

import ctypes
import os
import time

import cupy as cp
import cupy.cuda.compiler as _cupy_compiler
import numpy as np
import nvidia.cuda_runtime

import gpu_bench
from gpu_verbund import BARRIER_SRC, G_MAX, device_view_of_host
from nbody_kernel import G_AU

_cupy_compiler._cudadevrt = os.path.join(
    nvidia.cuda_runtime.__path__[0], "lib", "libcudadevrt.a")

import warnings  # noqa: E402

warnings.filterwarnings("ignore", message="The grid size will be reduced")

BLOCK = 256                    # Threads je Block = Kachelbreite
KALIBRIER_N = 4096             # Koerperzahl des Mikro-Benchmarks
KALIBRIER_SCHRITTE = 12
KALIBRIER_RUNDEN = 3           # davon zaehlt die BESTE (siehe miss_gewicht)

# Gemessene Gewichte je (Device, kraft_f32). Modulweit, weil sich die
# Hardware waehrend eines Laufs nicht aendert — sonst zahlte jede neue
# Session die Messung erneut.
_gewicht_cache: dict[tuple[int, bool], float] = {}


def miss_gewicht(device: int, kraft_f32: bool = True) -> float:
    """Rechenleistung einer Karte fuer diesen Kernel — aus dem persistenten
    Hardware-Cache (`gpu_bench`) oder frisch gemessen. Zusaetzlich modulweit
    im Speicher gehalten, damit wiederholte Aufrufe je Lauf gar nicht erst die
    Datei anfassen. Die Last-Art trennt f32- und f64-Kraftschleife, weil sie
    verschiedene Karten kroenen koennen."""
    schluessel = (device, bool(kraft_f32))
    if schluessel in _gewicht_cache:
        return _gewicht_cache[schluessel]
    art = "allpairs_f32" if kraft_f32 else "allpairs_f64"
    wert = gpu_bench.hole_gewichte(
        art, lambda d: _miss_gewicht_roh(d, kraft_f32), [device])[device]
    _gewicht_cache[schluessel] = wert
    return wert


def _miss_gewicht_roh(device: int, kraft_f32: bool = True) -> float:
    """Rechenleistung einer Karte FUER DIESEN KERNEL, gemessen.

    Eine Datenblatt-Metrik taugt hier nicht — sie lag nachweislich falsch
    herum: Nach SM-Zahl x Takt x FP32-Lanes muesste eine RTX 8000 die
    V100 schlagen (7,47e9 zu 7,07e9), gemessen ist die V100 aber
    1,67-mal schneller. Der Kernel liest den f64-Zustand aus dem VRAM,
    und dort gewinnt HBM2 gegen GDDR6; das steht in keiner FLOP-Tabelle.

    Darum ein Mikro-Benchmark: 4096 Koerper, ein Dutzend Schritte.

    MEHRERE RUNDEN, davon die beste. Eine einzelne Messung misst
    zuverlaessig das Falsche: Eine ruhende Karte steht in einer
    Idle-Taktstufe und braucht Bruchteile einer Sekunde auf vollen Takt.
    Gemessen ueber fuenf Runden lag die erste Runde 10-20 % unter allen
    folgenden, und zwar je nach Reihenfolge bei einer anderen Karte —
    die erste Fassung dieser Funktion bildete damit die MESSREIHENFOLGE
    ab statt der Leistung und haette einer baugleichen Karte ein um ein
    Fuenftel kleineres Segment gegeben. Stoerungen (kalte Karte, fremde
    Last) wirken nur nach unten, deshalb ist das Maximum der robuste
    Schaetzer, nicht der Mittelwert.

    Rueckgabe: Schritte pro Sekunde (nur als VERHAELTNIS aussagekraeftig,
    die absolute Zahl haengt an KALIBRIER_N).
    """
    n = KALIBRIER_N
    rng = np.random.default_rng(0)
    x = rng.uniform(-1e4, 1e4, n)
    y = rng.uniform(-1e4, 1e4, n)
    null = np.zeros(n)
    mass = np.full(n, 1.0)
    einzel = NBodySelfGrav([device], softening_au=100.0,
                           kraft_f32=kraft_f32)
    st = einzel.load_state(x, y, null, null, mass, _kalibrieren=True)
    beste = 0.0
    for _ in range(KALIBRIER_RUNDEN):
        einzel.step_batch(st, 1e-6, KALIBRIER_SCHRITTE)   # warmlaufen
        cp.cuda.Device(device).synchronize()
        t0 = time.perf_counter()
        einzel.step_batch(st, 1e-6, KALIBRIER_SCHRITTE)
        cp.cuda.Device(device).synchronize()
        dauer = time.perf_counter() - t0
        beste = max(beste, KALIBRIER_SCHRITTE / max(dauer, 1e-9))
    return beste


def waehle_karten(kraft_f32: bool = True,
                  mindest_anteil: float = 0.25) -> list[int]:
    """Alle Karten, die gemessen mindestens `mindest_anteil` der besten
    Leistung bringen, absteigend sortiert.

    Bewusst NICHT `pick_devices` aus `nbody_kernel`: das filtert nach
    f64-Score und sortiert damit genau die Karten aus, die dieser Kernel
    mit seiner f32-Kraftschleife am besten nutzt (RTX 8000: in f64 1/32
    und unbrauchbar, in f32 gemessen 655 gegen 1092 Schritte/s einer
    V100 — voll konkurrenzfaehig).

    Schwaechere Karten bleiben draussen, weil die Barrier auf die
    langsamste wartet."""
    n = cp.cuda.runtime.getDeviceCount()
    g = {d: miss_gewicht(d, kraft_f32) for d in range(n)}
    best = max(g.values())
    return sorted((d for d in g if g[d] >= mindest_anteil * best),
                  key=lambda d: -g[d])[:G_MAX]


def segmentiere(n: int, gewichte: list[float]) -> list[tuple[int, int]]:
    """Koerper nach Rechenleistung auf die Karten aufteilen.

    Zusammenhaengende Segmente (nicht round-robin wie bei den
    Asteroiden-Shards): Hier muss jede Karte ihr Segment als EINEN Block
    in den Austauschpuffer schreiben, und alle Koerper kosten gleich viel
    — es gibt keine heissen Ausreisser, die sich verteilen muessten.

    Rueckgabe: [(start, laenge)] je Karte, luecken- und ueberlappungsfrei.
    """
    ges = sum(gewichte)
    grenzen = [0]
    lauf = 0.0
    for w in gewichte[:-1]:
        lauf += w
        grenzen.append(int(round(n * lauf / ges)))
    grenzen.append(n)
    # Monoton halten: bei sehr kleinem n koennen zwei Grenzen
    # zusammenfallen — dann bekommt eine Karte eben nichts.
    for i in range(1, len(grenzen)):
        grenzen[i] = max(grenzen[i], grenzen[i - 1])
    return [(grenzen[i], grenzen[i + 1] - grenzen[i])
            for i in range(len(gewichte))]


_SRC = BARRIER_SRC + r"""
// Austauschbereich im gemappten Host-Speicher. Anders als beim alten
// Kernel steht hier NUR der Rundenzaehler drin — die Positionssegmente
// gehen ueber einen eigenen Puffer, weil ihre Groesse an n haengt und
// nicht an G_MAX.
struct GSync {
    unsigned int round_[G_MAX * PAD];
};

// Rechentyp der KRAFTSCHLEIFE (-DKRAFT_F32 schaltet auf float). Der
// ZUSTAND bleibt in beiden Faellen f64.
#ifdef KRAFT_F32
typedef float kreal;
__device__ inline float wurzel_inv(float v) { return rsqrtf(v); }
#else
typedef double kreal;
__device__ inline double wurzel_inv(double v) { return rsqrt(v); }
#endif

// Gemischte Genauigkeit, gemessen statt gewaehlt. An N=400 ueber 8000
// Schritte, Energieerhaltung dE/E gegen eine f64-NumPy-Referenz
// (-4,55e-5):
//
//   alles f32                -1,27e-1   unbrauchbar
//   f64-Zustand, f32-Kraft   -7,89e-5   gleichwertig
//   alles f64                -7,42e-5
//
// Entscheidend ist die AKKUMULATION, nicht die Kraft: `v += a*dt/2`
// addiert bei v ~ 1680 AU/a Inkremente von ~0,2, und die f32-Aufloesung
// liegt dort schon bei 1e-4. Ueber tausende Schritte geht das unter.
// Die Kraftsumme dagegen ist durch das Softening ohnehin geglaettet.
//
// Damit rechnen ALLE Karten dasselbe, und die in f64 auf 1/32 gebremsten
// RTX 8000 sind vollwertig (655 gegen 1092 Schritte/s einer V100).
//
// Die Kachel haelt ABSOLUTE Positionen. Sie relativ zum Zielkoerper zu
// speichern waere praeziser, geht aber nicht: Shared Memory ist
// blockweit geteilt, und jeder Thread im Block hat einen anderen
// Zielkoerper — er wuerde die Kachel mit seinem eigenen Bezugspunkt
// ueberschreiben. (Genau dieser Fehler kostete beim Bau eine Runde: bei
// n=2 wurde die Kachel zu exakt 0 und die Koerper flogen geradeaus.)
__device__ inline void beschleunigung(
    const double px, const double py,
    const double* __restrict__ gx, const double* __restrict__ gy,
    const double* __restrict__ gm, const int n,
    const double G, const double eps2,
    kreal* sx, kreal* sy, kreal* sm,
    cg::thread_block& block,
    double& ax, double& ay)
{
    kreal sax = 0, say = 0;
    const kreal pxk = (kreal)px, pyk = (kreal)py;
    const kreal Gk = (kreal)G, eps2k = (kreal)eps2;
    const int p = blockDim.x;
    for (int kach = 0; kach < n; kach += p) {
        const int j = kach + threadIdx.x;
        // Ueberhang mit Masse 0 fuellen: traegt nichts bei und haelt die
        // Schleifenlaenge fuer alle Threads gleich (kein divergenter
        // __syncthreads).
        sx[threadIdx.x] = (j < n) ? (kreal)gx[j] : (kreal)0;
        sy[threadIdx.x] = (j < n) ? (kreal)gy[j] : (kreal)0;
        sm[threadIdx.x] = (j < n) ? (kreal)gm[j] : (kreal)0;
        block.sync();
        const int gueltig = min(p, n - kach);
        #pragma unroll 4
        for (int q = 0; q < gueltig; q++) {
            const kreal dx = sx[q] - pxk;
            const kreal dy = sy[q] - pyk;
            // Plummer: a = G m d / (d^2 + eps^2)^(3/2). Der Selbstterm
            // (dx = dy = 0) liefert 0/eps^3 = 0 und braucht keinen Test.
            const kreal r2 = dx * dx + dy * dy + eps2k;
            const kreal inv = wurzel_inv(r2);
            const kreal f = Gk * sm[q] * inv * inv * inv;
            sax += f * dx;
            say += f * dy;
        }
        block.sync();
    }
    ax = (double)sax; ay = (double)say;
}

extern "C" __global__ void selfgrav_kernel(
    // Vollkopien im lokalen VRAM (alle n Koerper)
    double* __restrict__ gx, double* __restrict__ gy,
    const double* __restrict__ gm,
    // Geschwindigkeiten + Beschleunigungen NUR des eigenen Segments
    double* __restrict__ vx, double* __restrict__ vy,
    double* __restrict__ ax, double* __restrict__ ay,
    // Austausch: [2n] im gemappten Host-Speicher (x-Block, y-Block)
    double* __restrict__ tausch,
    GSync* gs,
    float* __restrict__ snap,      // Ausgabe [steps][4][nSeg]
    // TRACER (Kernel B): masselose Testteilchen, die dem Feld der Massen
    // folgen. Nur das eigene Segment, keine Vollkopie — sie sind fuer
    // niemanden Quelle. Deshalb auch kein Austausch und keine eigene
    // Barrier: Ein Tracer schreibt ausschliesslich seine eigenen Werte.
    double* __restrict__ tx, double* __restrict__ ty,
    double* __restrict__ tvx, double* __restrict__ tvy,
    double* __restrict__ tax, double* __restrict__ tay,
    float* __restrict__ tsnap,     // Ausgabe [steps][2][tSeg] (nur x|y)
    const int tSeg,
    const int seg0, const int nSeg, const int n,
    const int gpuId, const int nGpus,
    const double G, const double eps2, const double dt, const int steps)
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    extern __shared__ char sh_roh[];
    kreal* sx = (kreal*)sh_roh;
    kreal* sy = sx + blockDim.x;
    kreal* sm = sx + 2 * blockDim.x;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int schritt = gridDim.x * blockDim.x;
    unsigned int barRound = 0;

    // ACHTUNG, Kollektiv-Regel: `beschleunigung` enthaelt block.sync().
    // Deshalb laeuft die Schleife ueber BASIS, nicht ueber tid — jeder
    // Thread des Blocks nimmt an jeder Iteration teil, auch wenn er
    // keinen gueltigen Zielkoerper hat. Ein `for (i = tid; i < nSeg;
    // i += schritt)` liesse Threads jenseits von nSeg die Barrier
    // ueberspringen: divergentes block.sync() ist undefiniert, und es
    // hat sich als grob falsche Bahnen geaeussert (Median 1,9e4 AU
    // Abweichung), nicht als sauberer Absturz.
    for (int basis = 0; basis < nSeg; basis += schritt) {
        const int i = basis + tid;
        const bool aktiv = (i < nSeg);
        double a0, a1;
        beschleunigung(aktiv ? gx[seg0 + i] : 0.0,
                       aktiv ? gy[seg0 + i] : 0.0,
                       gx, gy, gm, n, G, eps2, sx, sy, sm, block, a0, a1);
        if (aktiv) { ax[i] = a0; ay[i] = a1; }
    }
    // Startbeschleunigung der Tracer — dieselbe Kachelschleife ueber
    // dieselben Massen. Sie laufen NACH den Massen und im selben Kernel:
    // Ein eigener Launch bekaeme die Positionen der Zwischenschritte nie
    // zu sehen (der Massen-Launch rechnet `steps` Schritte am Stueck),
    // und pro Schritt zu synchronisieren waere genau die Barrier, die
    // sich bei kleiner Koerperzahl schon als teurer erwiesen hat als die
    // Rechnung selbst.
    for (int basis = 0; basis < tSeg; basis += schritt) {
        const int i = basis + tid;
        const bool aktiv = (i < tSeg);
        double a0, a1;
        beschleunigung(aktiv ? tx[i] : 0.0, aktiv ? ty[i] : 0.0,
                       gx, gy, gm, n, G, eps2, sx, sy, sm, block, a0, a1);
        if (aktiv) { tax[i] = a0; tay[i] = a1; }
    }
    // Ohne diese Barrier koennte eine schnelle Karte schon Positionen
    // schreiben, waehrend eine langsame sie noch als Quellen liest.
    sys_barrier(gs->round_, gpuId, nGpus, &barRound, grid, tid);

    for (int s = 0; s < steps; s++) {
        // --- Kick (halb) + Drift, nur auf dem eigenen Segment ---
        for (int i = tid; i < nSeg; i += schritt) {
            vx[i] += ax[i] * (0.5 * dt);
            vy[i] += ay[i] * (0.5 * dt);
            const double nx = gx[seg0 + i] + vx[i] * dt;
            const double ny = gy[seg0 + i] + vy[i] * dt;
            gx[seg0 + i] = nx;
            gy[seg0 + i] = ny;
            // Direkt in den Austauschpuffer: erspart einen zweiten Lauf
            // ueber das Segment.
            tausch[seg0 + i] = nx;
            tausch[n + seg0 + i] = ny;
        }
        sys_barrier(gs->round_, gpuId, nGpus, &barRound, grid, tid);

        // --- Fremde Segmente einsammeln (einmal n Loads ueber PCIe;
        //     die Kraftschleife rechnet danach rein aus dem VRAM) ---
        if (nGpus > 1) {
            for (int i = tid; i < n; i += schritt) {
                if (i < seg0 || i >= seg0 + nSeg) {
                    gx[i] = tausch[i];
                    gy[i] = tausch[n + i];
                }
            }
            // Erst wenn ALLE gelesen haben, darf der naechste Drift den
            // Austauschpuffer wieder ueberschreiben.
            sys_barrier(gs->round_, gpuId, nGpus, &barRound, grid, tid);
        }

        // --- Kraft am neuen Ort + zweiter halber Kick ---
        // Wieder ueber BASIS: block.sync() in `beschleunigung` ist
        // kollektiv (siehe oben).
        for (int basis = 0; basis < nSeg; basis += schritt) {
            const int i = basis + tid;
            const bool aktiv = (i < nSeg);
            double a0, a1;
            beschleunigung(aktiv ? gx[seg0 + i] : 0.0,
                           aktiv ? gy[seg0 + i] : 0.0,
                           gx, gy, gm, n, G, eps2, sx, sy, sm, block,
                           a0, a1);
            if (!aktiv) continue;
            ax[i] = a0; ay[i] = a1;
            vx[i] += a0 * (0.5 * dt);
            vy[i] += a1 * (0.5 * dt);
            // Ausgabe bleibt f32: der Client quantisiert ohnehin, und
            // der Ring traegt f32. Die f64-Wahrheit bleibt auf der Karte.
            float* z = snap + (size_t)s * 4 * nSeg;
            z[i]            = (float)gx[seg0 + i];
            z[nSeg + i]     = (float)gy[seg0 + i];
            z[2 * nSeg + i] = (float)vx[i];
            z[3 * nSeg + i] = (float)vy[i];
        }

        // --- TRACER, kompletter Leapfrog-Schritt am Stueck ---
        // Sie duerfen erst hier laufen: Die Massen stehen jetzt an ihren
        // NEUEN Orten, und genau dagegen sollen die Tracer fallen. Weil
        // sie fuer niemanden Quelle sind, braucht es weder Austausch noch
        // Barrier — Kick, Drift und Kick passen in eine Schleife.
        //
        // Nur x|y in die Ausgabe: Die Geschwindigkeit der Tracer
        // interessiert niemanden. Sie halbiert die Ausgabemenge, und bei
        // hunderttausenden Tracern ist die Ausgabe der teure Teil.
        for (int basis = 0; basis < tSeg; basis += schritt) {
            const int i = basis + tid;
            const bool aktiv = (i < tSeg);
            double px = 0.0, py = 0.0;
            if (aktiv) {
                tvx[i] += tax[i] * (0.5 * dt);
                tvy[i] += tay[i] * (0.5 * dt);
                tx[i] += tvx[i] * dt;
                ty[i] += tvy[i] * dt;
                px = tx[i]; py = ty[i];
            }
            double a0, a1;
            beschleunigung(px, py, gx, gy, gm, n, G, eps2,
                           sx, sy, sm, block, a0, a1);
            if (!aktiv) continue;
            tax[i] = a0; tay[i] = a1;
            tvx[i] += a0 * (0.5 * dt);
            tvy[i] += a1 * (0.5 * dt);
            float* zt = tsnap + (size_t)s * 2 * tSeg;
            zt[i]        = (float)tx[i];
            zt[tSeg + i] = (float)ty[i];
        }
        grid.sync();
    }
}
"""


class NBodySelfGrav:
    """Selbstgravitierender Verbund aus 1..G_MAX Karten.

    Aussenverhalten wie `NBodyCuda`: `load_state` uebernimmt den Zustand
    (bleibt danach GPU-resident), `step_batch` rechnet mehrere Schritte in
    EINEM cooperative Launch und liefert f32-Snapshots."""

    def __init__(self, devices, softening_au: float,
                 kraft_f32: bool = True):
        if isinstance(devices, int):
            devices = [devices]
        if not devices or len(devices) > G_MAX:
            raise ValueError(f"1..{G_MAX} Devices erwartet: {devices}")
        if not softening_au > 0:
            raise ValueError(f"Softening muss positiv sein: {softening_au}")
        self.devices = list(devices)
        self.device = self.devices[0]
        self.eps2 = float(softening_au) ** 2
        self.kraft_f32 = bool(kraft_f32)
        self._block = BLOCK
        self._kreal_bytes = 4 if self.kraft_f32 else 8
        self._kerns = {}
        self._mods = {}
        opts = ("--std=c++17",)
        if self.kraft_f32:
            opts += ("-DKRAFT_F32",)
        for d in self.devices:
            with cp.cuda.Device(d):
                mod = cp.RawModule(code=_SRC, options=opts,
                                   enable_cooperative_groups=True)
                self._mods[d] = mod
                self._kerns[d] = mod.get_function("selfgrav_kernel")

    def name(self) -> str:
        return " + ".join(
            cp.cuda.runtime.getDeviceProperties(d)["name"].decode()
            for d in self.devices)


    def load_state(self, x: np.ndarray, y: np.ndarray,
                   vx: np.ndarray, vy: np.ndarray,
                   mass: np.ndarray, *_egal,
                   tracer: tuple | None = None,
                   _kalibrieren: bool = False, **_auch_egal) -> dict:
        """Vollzustand uebernehmen. Jede Karte haelt ALLE Positionen und
        Massen, integriert aber nur ihr nach GEMESSENER Leistung
        gewichtetes Segment.

        Die zusaetzlichen Argumente von `NBodyCuda.load_state`
        (Sichtbarkeit, Asteroiden-Flag, Beruehrungsradien) werden
        angenommen und ignoriert: Hier ist jeder Koerper eine Masse, es
        gibt keine Testteilchen und keine Beruehrungen. So kann der
        Producer beide Kernel gleich aufrufen.

        `_kalibrieren` bricht die Rekursion beim Mikro-Benchmark: der
        legt selbst einen Zustand an und braucht keine Gewichtung, weil
        er immer auf genau einer Karte laeuft."""
        n = len(x)
        if n < 1:
            raise ValueError("leerer Zustand")
        ng = len(self.devices)
        gewichte = [1.0] * ng if (_kalibrieren or ng == 1) \
            else [miss_gewicht(d, self.kraft_f32) for d in self.devices]
        segmente = segmentiere(n, gewichte)

        x64 = np.ascontiguousarray(x, np.float64)
        y64 = np.ascontiguousarray(y, np.float64)
        vx64 = np.ascontiguousarray(vx, np.float64)
        vy64 = np.ascontiguousarray(vy, np.float64)
        m64 = np.ascontiguousarray(mass, np.float64)

        # Austauschpuffer + Rundenzaehler im gemappten Host-Speicher:
        # cudaHostAllocPortable(1) | cudaHostAllocMapped(2) macht ihn in
        # JEDEM Kartenkontext zero-copy erreichbar.
        tausch_bytes = 2 * n * 8
        gs_bytes = 4 * G_MAX * 16
        if ng > 1:
            tausch_ptr = cp.cuda.runtime.hostAlloc(tausch_bytes, 3)
            gs_ptr = cp.cuda.runtime.hostAlloc(gs_bytes, 3)
            ctypes.memset(gs_ptr, 0, gs_bytes)
            ctypes.memset(tausch_ptr, 0, tausch_bytes)
        else:
            tausch_ptr = gs_ptr = None

        # Tracer: masselose Testteilchen. Sie werden GLEICHMAESSIG
        # aufgeteilt und nicht nach den Massen-Segmenten — ihre Kosten
        # haengen nur an ihrer eigenen Zahl, nicht daran, welche Masse wo
        # liegt. Jede Karte rechnet ihren Block gegen ALLE Massen.
        if tracer is not None:
            tx64 = np.ascontiguousarray(tracer[0], np.float64)
            ty64 = np.ascontiguousarray(tracer[1], np.float64)
            tvx64 = np.ascontiguousarray(tracer[2], np.float64)
            tvy64 = np.ascontiguousarray(tracer[3], np.float64)
            n_tracer = len(tx64)
            t_segmente = segmentiere(n_tracer, gewichte)
        else:
            n_tracer = 0
            t_segmente = [(0, 0)] * ng

        shards = []
        for g, d in enumerate(self.devices):
            seg0, n_seg = segmente[g]
            t0, t_seg = t_segmente[g]
            with cp.cuda.Device(d):
                sh = {
                    "dev": d, "gpu_id": g, "seg0": seg0, "n_seg": n_seg,
                    "gx": cp.asarray(x64), "gy": cp.asarray(y64),
                    "gm": cp.asarray(m64),
                    "vx": cp.asarray(vx64[seg0:seg0 + n_seg]),
                    "vy": cp.asarray(vy64[seg0:seg0 + n_seg]),
                    "ax": cp.zeros(max(n_seg, 1), cp.float64),
                    "ay": cp.zeros(max(n_seg, 1), cp.float64),
                    "t0": t0, "t_seg": t_seg,
                    "tx": cp.asarray(tx64[t0:t0 + t_seg]) if t_seg
                          else cp.zeros(1, cp.float64),
                    "ty": cp.asarray(ty64[t0:t0 + t_seg]) if t_seg
                          else cp.zeros(1, cp.float64),
                    "tvx": cp.asarray(tvx64[t0:t0 + t_seg]) if t_seg
                           else cp.zeros(1, cp.float64),
                    "tvy": cp.asarray(tvy64[t0:t0 + t_seg]) if t_seg
                           else cp.zeros(1, cp.float64),
                    "tax": cp.zeros(max(t_seg, 1), cp.float64),
                    "tay": cp.zeros(max(t_seg, 1), cp.float64),
                }
                if ng > 1:
                    sh["tausch"] = device_view_of_host(tausch_ptr, tausch_bytes,
                                                d, cp.float64)
                    sh["gs"] = device_view_of_host(gs_ptr, gs_bytes, d, cp.uint8)
                else:
                    sh["tausch"] = cp.zeros(2 * n, cp.float64)
                    sh["gs"] = cp.zeros(gs_bytes, cp.uint8)
                shards.append(sh)
        return {"N": n, "T": n_tracer, "shards": shards,
                "segmente": segmente,
                "_host": (tausch_ptr, gs_ptr)}

    def step_batch(self, st: dict, dt_years: float,
                   steps: int) -> np.ndarray:
        """`steps` Leapfrog-Schritte in einem cooperative Launch je Karte.

        Rueckgabe: f32-Array (steps, 4n) [x|y|vx|vy] in
        Originalreihenfolge. Erst werden ALLE Karten gelauncht — sie
        warten in der Barrier aufeinander —, dann eingesammelt."""
        if st is None:
            raise ValueError("kein Zustand geladen")
        if steps < 1:
            raise ValueError(f"steps muss >= 1 sein: {steps}")
        n = st["N"]
        ng = len(st["shards"])
        for sh in st["shards"]:
            n_seg = sh["n_seg"]
            if n_seg == 0:
                continue
            with cp.cuda.Device(sh["dev"]):
                sh["snap"] = cp.empty(steps * 4 * n_seg, cp.float32)
                t_seg = sh["t_seg"]
                sh["tsnap"] = cp.empty(steps * 2 * max(t_seg, 1), cp.float32)
                # Grid nach dem GROESSEREN der beiden Segmente: Massen und
                # Tracer laufen im selben Kernel, und beide Schleifen
                # brauchen genug Threads. Bei 400.000 Tracern gegen 20.000
                # Massen bestimmen die Tracer das Grid.
                grid = max(1, (max(n_seg, t_seg) + self._block - 1)
                           // self._block)
                self._kerns[sh["dev"]](
                    (grid,), (self._block,),
                    (sh["gx"], sh["gy"], sh["gm"],
                     sh["vx"], sh["vy"], sh["ax"], sh["ay"],
                     sh["tausch"], sh["gs"], sh["snap"],
                     sh["tx"], sh["ty"], sh["tvx"], sh["tvy"],
                     sh["tax"], sh["tay"], sh["tsnap"],
                     cp.int32(sh["t_seg"]),
                     cp.int32(sh["seg0"]), cp.int32(n_seg), cp.int32(n),
                     cp.int32(sh["gpu_id"]), cp.int32(ng),
                     cp.float64(G_AU), cp.float64(self.eps2),
                     cp.float64(dt_years), cp.int32(steps)),
                    shared_mem=3 * self._block * self._kreal_bytes)
        out = np.empty((steps, 4 * n), np.float32)
        for sh in st["shards"]:
            n_seg = sh["n_seg"]
            if n_seg == 0:
                continue
            with cp.cuda.Device(sh["dev"]):
                snap = cp.asnumpy(sh["snap"]).reshape(steps, 4, n_seg)
            s0 = sh["seg0"]
            for f in range(4):
                out[:, f * n + s0:f * n + s0 + n_seg] = snap[:, f, :]
        if not st["T"]:
            return out
        # Tracer hinten anhaengen: [x_massen | y_.. | vx | vy | x_tracer |
        # y_tracer]. Der Producer schreibt ohnehin nur x|y in den Ring,
        # und die Tracer-Geschwindigkeit braucht niemand.
        m = st["T"]
        raus = np.empty((steps, 4 * n + 2 * m), np.float32)
        raus[:, :4 * n] = out
        for sh in st["shards"]:
            t_seg = sh["t_seg"]
            if t_seg == 0:
                continue
            with cp.cuda.Device(sh["dev"]):
                ts = cp.asnumpy(sh["tsnap"]).reshape(steps, 2, t_seg)
            t0 = sh["t0"]
            raus[:, 4 * n + t0:4 * n + t0 + t_seg] = ts[:, 0, :]
            raus[:, 4 * n + m + t0:4 * n + m + t0 + t_seg] = ts[:, 1, :]
        return raus

    def export_f64(self, st: dict) -> np.ndarray:
        """Exakten f64-Zustand [x|y|vx|vy] (4n) in Originalreihenfolge.

        Gleiche Form wie `NBodyCuda.export_f64` — der Film-Producer
        dumpt damit den Zustand fuer die Engine-Uebergabe und darf nicht
        wissen muessen, welcher Kernel gerechnet hat. Die Masse fehlt
        bewusst: Sie aendert sich hier nie (keine Verschmelzungen)."""
        n = st["N"]
        raus = np.empty(4 * n, dtype="<f8")
        erste = st["shards"][0]
        with cp.cuda.Device(erste["dev"]):
            raus[0:n] = cp.asnumpy(erste["gx"])
            raus[n:2 * n] = cp.asnumpy(erste["gy"])
        for sh in st["shards"]:
            n_seg = sh["n_seg"]
            if n_seg == 0:
                continue
            s0 = sh["seg0"]
            with cp.cuda.Device(sh["dev"]):
                raus[2 * n + s0:2 * n + s0 + n_seg] = cp.asnumpy(sh["vx"])
                raus[3 * n + s0:3 * n + s0 + n_seg] = cp.asnumpy(sh["vy"])
        return raus
