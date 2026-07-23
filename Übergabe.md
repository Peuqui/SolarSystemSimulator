# Übergabe — Film-Streaming: Freeze/Stocken behoben, GPU-Leak offen

**Branch:** `feat/particle-mesh`
**Kontext:** Film-Streaming für den neuen Particle-Mesh-Kernel (Millionen Massen +
Tracer, Galaxien/Filamente). Der Producer rechnet auf der GPU voraus (Ringpuffer),
der Browser spielt die Timeline wie ein Video ab. Siehe auch die Memory-Dateien
`project_zwei_kernel_pm`, `project_film_freeze_diagnose`, `design_lod_deterministisch`.

## In dieser Session behoben (committet)

1. **LOD deterministisch** (`_dichte_filter`/`_lod_auswahl` in `backend/server.py`):
   dichteabhängiges LOD raus, jetzt globale Hash-Stichprobe `budget/N` — pro Sample
   dieselben Objekte, kein Flackern. Test `backend/test_lod_auswahl.py` (umbenannt
   von `test_lod_dichte.py`). `test_film_protokoll.py` auf v7-Layout nachgezogen.

2. **Film-Freeze bei hohem Abspieltempo** (`filmTick` in `index.html`): Die
   Kantenraten-Bremse (`min(rate, _filmHeadRate)·FILM_BREMSE`) fiel bei gedrosseltem
   Producer auf 0 → Playhead stand → Producer↔Playhead-Deadlock. Bremse raus.

3. **Diashow-Stocken** (`filmTick`): Playhead sprang 0↔100 Tage. Jetzt
   **puffer-basierte Glättung** — je näher der Playhead der Datenkante, desto
   langsamer (`follow = rate · min(1, istVorratS/FILM_PUFFER_S)`), er pendelt sich
   glatt auf die Produktionsrate ein. Gemessen: 0 Nullschritte, gleichmäßig.

4. **Dangling-Producer** (tote Clients): `ping_interval` war `None` → weggebrochene
   Browser (halboffene TCP) blieben unerkannt, Producer lief verwaist weiter und
   hielt GPU-Speicher. Jetzt `ping_interval=20, ping_timeout=30` in `server.py`.

5. **PM-Kernel `asnumpy`-Bündelung** (`backend/pm_kernel.py` `step_batch`): 48
   synchrone GPU→CPU-Transfers je Batch → 1. Korrekt (Test bestanden), aber nur
   56→63 Tage/s — der Engpass liegt woanders (siehe unten).

## Aktueller Stand (verifiziert per DevTools bei 1M+500k Objekten)

- **Kein Freeze mehr**, **kein Diashow-Stocken** — läuft glatt.
- ABER bei hohem Tempo läuft es **in Zeitlupe (~63 Tage/s)**: das ist die
  **Produktionsgrenze** des PM-Kernels bei 447k Objekten. `step=96 %`, GPU nur
  ~45 % → **launch-overhead-gebunden** (viele kleine cupy-Kernel je Substep).

## Offen — Reihenfolge

### 1. GPU-Kontext-Leak (NÄCHSTES, sauber schließen)
Der Film-Producer hält CUDA-Kontexte auf **zwei** V100 (z. B. 308 MiB idle + 484
MiB rechnend), nutzt aber nur **eine**. Bei jedem neuen Browser-Tab wieder.
- **Ausgeschlossen:** die Kartenwahl. Isoliert getestet — `waehle_karten` +
  `waehle_karte_pm` belegen KEINE GPU (Cache `~/.cache/solar-system/gpu_bench.json`
  greift). Der zweite Kontext entsteht erst im **PM-Kernel-Betrieb**, obwohl der
  Code durchweg `with cp.cuda.Device(self.device)` nutzt — per Code-Lesen nicht
  auffindbar.
- **Nächster Schritt:** Device-Touch-Instrumentierung im laufenden Producer
  (welche Devices werden berührt?). **Sauberster Fix vermutlich:** die PM-Karte im
  **Server** wählen (via `gpu_bench`-Cache, kein CUDA-Touch) und dem Producer-
  Prozess per **`CUDA_VISIBLE_DEVICES`** nur diese eine Karte zeigen → ein zweiter
  Kontext ist dann unmöglich.
- Hinweis: CUDA-Reihenfolge (V100,V100,V100,RTX,RTX) ≠ nvidia-smi-Reihenfolge
  (RTX,V100,RTX,V100,V100) — beim Debuggen aufpassen.

### 2. PM-Kernel-Durchsatz (der große Hebel gegen die Zeitlupe)
`step=96 %`, GPU nur ~45 %. Viele kleine cupy-Kernel je Substep (grid_fuer,
`_cic_deposit`, FFT, `_cic_gather`, Leapfrog × 8). Ansatz: **Kernel-Fusion /
CUDA-Graphs** (Launch-Overhead senken). Plus geplant: **Tracer auf separate Karte
(RTX)** auslagern — entlastet die Rechenkarte (Teil B).

### 3. Tracer-Kreis (visuell)
Beim Neustart (Inject/Handover) würfelt der Producer die Tracer auf die t=0-Scheibe
zurück (`wuerfle_tracer(seed=0)`, harte Kante bei 72000 AE), während die Massen ihren
Zustand behalten → scharfer Kreis + Halo. Fix: Tracer-Zustand server-intern über den
Handover mitführen statt neu würfeln (Tracer bleiben im Client anonym).

### 4. Farben konfigurierbar (B)
Massen und Tracer haben dieselbe Farbe (`_TRACER_RGB = '#cfe0ff'`). Gewünscht:
Massen hell-hellgelb, Tracer hellblau — als **benannte Konstanten** (SSOT, keine
verstreuten Hex-Literale).

## Diagnose-Werkzeuge (für die nächste Session)
- **Stack eines hängenden Producers ohne root** (ptrace_scope=1 blockiert gdb,
  py-spy fehlt): `faulthandler.register(signal.SIGUSR1, all_threads=True)` in
  `producer_main` einbauen, dann `kill -USR1 <pid>` → Stack ins Journal. Feuert
  NICHT bei blockierendem C-Call → dann `/proc/<pid>/wchan` lesen.
- **Producer-Diagnose-Logs:** `FilmSession.DIAG = True` (temporär) → `[film-diag]`
  (prod/play-Rate, Vorlauf, drossel%) und `[stream]`-Zeilen.
- **Live-Reproduktion:** Debug-Chrome `DISPLAY=:0 XAUTHORITY=~/.Xauthority
  chrome --remote-debugging-port=9222 http://127.0.0.1/solar-system/?v=dbg`,
  dann per chrome-devtools MCP `filmSpeed` über den log-Slider `film-speed`
  (Wert 3 = 1000 Tage/s) setzen und `filmPlayheadDays` messen.
- `/var/www/html/solar-system` → **Symlink** aufs Projekt (Client-Fix nach Reload
  sofort aktiv). Server: `systemctl --user restart solar-cuda.service`.
