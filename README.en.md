# Solar System Simulator

[Deutsch](README.md) | English

Browser-based N-body gravitational simulator. A single HTML file, no
build step, no backend — just open it in a browser or serve it through
any static web server.

![Render](https://img.shields.io/badge/render-Canvas%202D-blueviolet)
![Standalone](https://img.shields.io/badge/single--file-100kB-success)
![Mobile](https://img.shields.io/badge/mobile-ready-orange)

## Features

- **15 built-in scenarios**: Solar System, TRAPPIST-1, Alpha Centauri,
  Kepler-16, Kepler-47 (binary star with 3 planets), Trisolaris (3
  suns), stable/unstable Lagrange configurations, L4 Trojans, the
  figure-8 choreography, Butterfly I, Moth I, Goggles, Yarn, and an
  empty system for sandboxing
- **Inject perturber masses interactively** — position, mass (10⁻³ to
  10⁶ Earth masses, including stars above ~80 M⊕), speed and direction
- **Real-time N-body integration** with configurable time step and
  slowdown factor; pause, reset and single-step controls
- **Log-zoom mode** so the Sun and the outer planets fit on screen at
  the same time
- **Toggles** for trails (variable length), force vectors, barycenter
  lock and grid
- **Save/export/import** configurations locally
- **Live statistics**: kinetic / potential / total energy, angular
  momentum, barycenter drift, escape-trajectory detection
- **Full mobile support** — long-press, pinch-to-zoom, two-finger pan,
  dedicated mobile toolbar and bottom sheet

## Controls

### Desktop

| Input                            | Action                                 |
|----------------------------------|----------------------------------------|
| Left-click + drag                | Set position + velocity vector         |
| Right-/middle-click + drag       | Pan view                               |
| Scroll wheel                     | Zoom                                   |
| `P`                              | Pause / resume                         |
| `R`                              | Reset scenario                         |
| `Space`                          | Single step (while paused)             |

### Mobile

| Gesture                          | Action                                 |
|----------------------------------|----------------------------------------|
| 1 finger, long press             | Arm a position                         |
| Long press + drag                | Set velocity                           |
| 2-finger pinch                   | Zoom                                   |
| 2-finger drag                    | Pan view                               |
| Toolbar buttons                  | Pause, zoom ±, inject, reset           |

The world-space coordinate of an armed injection stays stable across
zoom, pan and pinch — clicking *Inject* spawns a new mass at exactly
that location, regardless of camera state.

## Running locally

Any static web server works, e.g.:

```bash
python3 -m http.server 8080
# Browser: http://localhost:8080/
```

Alternatively just open `index.html` from disk (`file://`) — the
simulation itself runs fully offline; only the optional footer widgets
(GoatCounter pageview hit, GitHub stars badge) stay inactive without
network access.

## Technical notes

- **Pure HTML / CSS / Canvas 2D / vanilla JS**, single file
- **Symplectic Verlet integrator** for the N-body step
- **No build pipeline, no npm, no libraries** — the sim engine itself
  fetches nothing. The footer optionally pulls GoatCounter (privacy-
  friendly pageview counter) and the current GitHub star count from the
  GitHub API; offline this only hides the stars badge
- **`localStorage`** for saved configurations and UI settings

## Stargazers over time

[![Star History Chart](https://api.star-history.com/svg?repos=Peuqui/SolarSystemSimulator&type=Date)](https://star-history.com/#Peuqui/SolarSystemSimulator&Date)

## License

MIT
