# Solar System Simulator

[Deutsch](README.md) | English

Browser-based N-body gravity simulator. The **default mode** is a single
HTML file — no build, no backend, just open it in a browser. For massive
scenarios (hundreds of thousands of bodies) there is an **optional CUDA
backend** with multi-GPU physics and a decoupled film mode.

![Render](https://img.shields.io/badge/render-Canvas%202D%20%2B%20WebGL-blueviolet)
![Standalone](https://img.shields.io/badge/frontend-single--file-success)
![CUDA](https://img.shields.io/badge/optional-CUDA%20multi--GPU-76b900)
![Mobile](https://img.shields.io/badge/mobile-ready-orange)

## Features

- **16 predefined scenarios**: Solar System, Solar System with asteroid
  belts (main and Kuiper belt, count configurable via slider), TRAPPIST-1,
  Alpha Centauri, Kepler-16, Kepler-47 (binary + 3 planets), Trisolaris
  (3 suns), Lagrange constellations (stable/unstable), Trojans (L4),
  figure-8 choreography, Butterfly I, Moth I, Goggles, Yarn, and an
  "Empty system" to build freely
- **Inject perturber masses interactively** — position, mass (10⁻³ to 10⁶
  Earth masses, including stars above ~80 M⊕), speed and direction
- **Inject asteroid clouds** — whole swarms of small bodies with count
  (logarithmic, **up to 50,000 per injection**), total mass, density and
  spread via sliders; Shift+click the inject button (or long-press on
  mobile) spawns a cloud instead of a single rogue
- **Five physics engines** via toggle — main thread, WebWorker, WebGPU,
  Hybrid and a native **CUDA backend (f64, multi-GPU)**, see
  [Physics engines](#physics-engines)
- **Film mode** — compute and display fully decoupled: the GPU simulates
  ahead at maximum throughput, the browser plays the timeline like a video
  (scrub, rewind, follow live), see [Film mode](#film-mode)
- **Asteroid collisions** (optional toggle) — asteroids bounce off each
  other like balls of different mass (mass-dependent slingshot: the heavier
  turns blue, the lighter red and is flung away). On hitting a planet/star
  an asteroid is absorbed; body × body at high relative speed **shatters**
  into fragments. All three paths apply live and in film mode
- **Real-time N-body integration** with configurable time step and
  slow-down factor; pause, reset and single-step
- **Log-zoom mode** for seeing the sun and outer planets at once
- **Orbit trails** (variable length), **force vectors**, **barycenter
  centering** and **grid** toggleable
- **Save, export and import configurations** locally — stores every body
  with its state **and all sliders and switches** (time step, film
  settings, display options)
- **Live stats**: kinetic / potential / total energy, angular momentum,
  barycenter drift, escape-trajectory detection; live benchmark badge
  (FPS, days/s, active engine)
- **Full mobile support** — long-press, pinch-zoom, two-finger pan,
  dedicated mobile toolbar and bottom sheet
- **Bilingual German / English** — toggle top-left in the side menu, choice
  persists in `localStorage`

## Physics engines

The engine is switchable in the side menu; the choice persists in
`localStorage`.

| Engine | Runs on | Best for |
|---|---|---|
| **Main thread (CPU)** | Browser, single thread | debug/reference |
| **WebWorker** | Browser, own thread (f64) | default — the fastest in-browser path on most devices |
| **WebGPU** | integrated/discrete GPU (f32) | strong GPUs; slower than the worker on a weak iGPU |
| **Hybrid** | Worker + targeted finer asteroid substeps | more precise close encounters |
| **CUDA backend** | NVIDIA GPUs, native f64, multi-GPU | hundreds of thousands of bodies, film mode |

Interestingly the **WebWorker** is the fastest in-browser path on many
(even weaker) machines: native f64 arithmetic on the CPU beats a weak iGPU
at ~7,000 bodies with only ~10 massive sources, whose per-substep
dispatch/sync latencies exceed the actual compute time. Only a strong GPU
(WebGPU) or the CUDA backend flip the balance.

## Film mode

Requires the active CUDA backend. Compute and display are fully decoupled:

- A dedicated **GPU producer process** simulates ahead at full throughput
  into a ring buffer and only stops when it runs too far ahead.
- The browser reads the buffer as a **push-streamed point film** and plays
  the timeline like a video player: **scrub, rewind, follow live** at the
  production edge.
- Only positions are transmitted, as **integer screen coordinates**
  (server-side view culling, u16 quantization); mass/visibility travel as
  compact events. The reference box follows the viewport rather than the
  scene, so a single body flung far out does not spoil the resolution at
  the centre.
- **Measured in-between frames instead of guessed curves**: for closely
  encountering asteroids the kernel records waypoints on the real
  trajectory during its fine loop (slider, 8 per interval by default);
  the browser only interpolates in straight lines between them. A
  perihelion passage is therefore a measurement. Everything else uses
  Catmull-Rom through four sample points — for those bodies the chord
  error stays below the visibility threshold anyway. The server spans the
  waypoints across whatever interval it streams, so the bandwidth per
  body is independent of playback speed.
- **Density LOD**: when the point budget cannot cover every body, a
  priority applies — massive bodies (sun, planets, rogues, stars, black
  holes) are **never** thinned, then come the asteroids of the loaded
  system, and injected clouds get whatever is left. Within each tier
  thinning is **density-aware** (kept ~ count^0.5 per grid cell): a belt
  of a few hundred objects no longer vanishes next to a cloud of hundreds
  of thousands, while dense regions still read as visibly denser. The
  selection runs over a hash of the body index — continuous, free of grid
  artifacts, and stable across samples. The budget is adjustable via a
  slider (Auto … "All"). **Physics and collisions always run for every
  body** — only the display is thinned.
- Collision events (merge, kill, bounce, shatter) carry their **exact
  location and time** and are replayed as explosions at the right moment.
  Contact with a massive body is tested per substep inside the kernel's
  fine loop — a body plunging into a star at 50 AU/year therefore
  disappears *inside* it rather than one sample interval short (0.07 AU,
  fifteen times a solar radius).
- On leaving film mode or switching engines the exact f64 state is handed
  over to the new engine — no loss of momentum.

This keeps a cloud of **> 250,000 bodies** playing smoothly while the GPU
keeps computing in the background.

## CUDA backend

Optional, for the last two engines and film mode. Requires NVIDIA GPU(s)
with a CUDA 12 driver. The frontend works fully without the backend
(WebWorker/WebGPU) — the backend only unlocks the CUDA engine and film
mode.

### Key points

- **Native f64 physics** in a CuPy/NVRTC kernel (Yoshida 4th order,
  identical to the worker reference); the whole frame runs as one
  cooperative launch (grid sync).
- **Multi-GPU sharding, hardware-agnostic**: asteroids are distributed
  across all suitable cards weighted by f64 score, the massive bodies are
  replicated and stay bit-identical. Substep synchronization runs
  atomics-free over PCIe-mapped host memory — no NVLink needed. With one
  card the same code path degenerates without overhead.
- **Hierarchical time steps**: only sun-diving bodies run in private fine
  loops, the rest on the coarse raster — instead of a single sun-diver
  dragging every body onto the minimum step.
- **Collision/bounce detection pipelined** on separate GPUs (f32 pre-filter
  + exact f64 verification), overlapped with the physics. The bounce search
  is the bottleneck (75–93 % of batch time) and is **split spatially across
  up to two cards**: each checks an x-stripe plus a halo and keeps only
  pairs whose left-hand partner belongs to it — gapless, non-overlapping,
  with no data exchange between the cards. Whether this pays off depends on
  the **candidate-pair count per sample**, not on the body count: the same
  250 k asteroids yield over 10⁹ pairs to check as a dense clump and under
  10⁶ once a belt has formed. The producer therefore adds and drops the
  second card **per batch** (thresholds with hysteresis, measured gain up
  to 1.7×) — no film restart required.
- **Server lifecycle driven by the browser**: selecting the CUDA engine
  starts the server on demand (optionally via systemd socket activation),
  deselecting stops it after a short idle grace — GPUs are free at rest.

### Setup & start

```bash
cd backend
# venv in the project root, with CuPy for CUDA 12 (toolkit bundled in wheel):
python3 -m venv ../venv
../venv/bin/pip install "cupy-cuda12x[ctk]" websockets numpy
# start the server (default port 8765, auto-picks the best f64 GPU):
../venv/bin/python server.py
```

Then pick the engine **"CUDA backend (native, f64)"** in the browser; the
frontend connects via WebSocket (local `127.0.0.1:8765`, remote via a
reverse proxy).

Useful switches: `--ring-gib` (film ring-buffer size, determines how much
past is navigable), `--det-gpus` (detection cards per session), `--diag`
(log time shares of producer and stream loop — for bottleneck hunting, off
in normal operation).

Tests and measurement tools (standalone scripts, no pytest):

```bash
cd backend
../venv/bin/python test_kernel.py             # kernel vs. NumPy, 1/2/3 GPUs
../venv/bin/python test_film_golden.py        # collision chain end-to-end
../venv/bin/python test_erkennung_streifen.py # stripes == a single card
../venv/bin/python test_lod_dichte.py         # priority + density selection
../venv/bin/python bench_erkennung.py         # switch threshold, 2nd card
../venv/bin/python bench_film.py --szene knoedel --det-gpus 1 2
```

## Controls

### Desktop

| Input                        | Action                                 |
|------------------------------|----------------------------------------|
| Left-click + drag            | Set position + velocity vector         |
| Right/Middle-click + drag    | Pan the view                           |
| Mouse wheel                  | Zoom                                   |
| `P`                          | Pause / Resume                         |
| `R`                          | Reset scenario                         |
| `Space`                      | Single step (while paused)             |

### Mobile

| Gesture                     | Action                                |
|-----------------------------|---------------------------------------|
| Hold one finger             | Arm position (long-press)             |
| Hold + drag                 | Set velocity                          |
| Two-finger pinch            | Zoom                                  |
| Two-finger drag             | Pan the view                          |
| Bottom toolbar              | Pause, zoom ±, reset, center          |
| Right action buttons        | Remove escaping / perturbers / clouds |
| Red FAB bottom-right        | Inject (long-press: cloud)            |
| Hamburger top-left          | Settings sheet                        |

The world coordinate of a planned injection stays stable during zoom, pan
and pinch — the marker shows it before each "Inject" click, and a repeated
click spawns another mass at exactly the same spot.

## Running locally

For the default mode any static web server works, e.g.:

```bash
python3 -m http.server 8080
# Browser: http://localhost:8080/
```

Or just open `index.html` directly in the browser (`file://` — the
simulation itself runs fully offline; only the optional footer elements
(GoatCounter pageview counter, GitHub stars badge) are inactive then). For
the CUDA engine + film mode see [CUDA backend](#cuda-backend).

## Browser performance

With large asteroid belts (~1400 bodies) there are differences between
browsers. For in-browser operation (without the CUDA backend):

**Desktop:**

| Browser | Recommendation | Note |
|---|---|---|
| **Edge / Chrome** | WebGPU **on** (strong GPU) or WebWorker | Dawn backend + DirectX 12 path; on a weak iGPU the worker is often faster |
| **Firefox** | WebWorker | `wgpu` backend still immature; Canvas ops costlier per call in Firefox |

**Android:**

| Browser | Recommendation | Note |
|---|---|---|
| **Chrome** | WebWorker or WebGPU | on most devices the worker gives the steadiest 60 fps |
| **Firefox** | WebWorker | weaker `wgpu` backend |

With few bodies (default scenarios without belts) the difference is
irrelevant; all browsers run smoothly at 60 fps. For hundreds of thousands
of bodies the CUDA backend is the only way.

## Technical

### Frontend (single-file)

- **Pure HTML / CSS / vanilla JS** in one file; asteroids are rendered via
  a **WebGL point batch** in a single draw call (instead of tens of
  thousands of canvas calls), the rest over Canvas 2D.
- **N-body integration** with symplectic Verlet (Yoshida 4th order);
  asteroid-heavy scenarios skip asteroid-asteroid gravity for O(N) instead
  of O(N²) cost.
- **Continuous collision detection (CCD)** via parametric trajectory
  intersection — fast encounters (cloud vs. belt head-on) are resolved at
  the contact time, without bodies tunneling through each other.
- **No build pipeline, no npm, no libraries** — the sim engine loads
  nothing. Only the footer optionally fetches GoatCounter (privacy-friendly
  pageview counter) and the GitHub stars value.
- **Network worker in film mode**: WebSocket reception and frame decoding
  run in a dedicated worker, and the payload is transferred rather than
  copied. Otherwise the main thread's render loop blocks socket draining,
  the TCP receive window collapses and the link goes underused.
- **`localStorage`** for saved configurations and UI settings.

### Backend (optional, `backend/`)

- **Python + CuPy/NVRTC** — `server.py` (WebSocket server, session
  management, v4 integer streaming with culling and LOD),
  `nbody_kernel.py` (multi-GPU f64 kernel with hierarchical time steps),
  `film_producer.py` (producer process: ring buffer, collision / bounce /
  shatter detection).
- **Process split**: the producer owns the GPU exclusively, the server
  answers buffer requests from shared memory in microseconds — the physics
  GPU never starves on the GIL.
- Tests: `test_kernel.py` (kernel vs. NumPy reference, 1/2/3 GPU +
  benchmarks), `test_film_golden.py` (collision chain end-to-end),
  `test_erkennung_streifen.py` (the spatially split bounce search returns
  exactly the same hits as a single card), `test_lod_dichte.py` (priority
  and density-aware point selection). Measurement tools:
  `bench_erkennung.py`, `bench_film.py`.

## Stargazers over time

[![Star History Chart](https://api.star-history.com/svg?repos=Peuqui/SolarSystemSimulator&type=Date)](https://star-history.com/#Peuqui/SolarSystemSimulator&Date)

## License

MIT
