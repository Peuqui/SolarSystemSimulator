# Solar System Simulator

Deutsch | [English](README.en.md)

Browser-basierter N-Körper-Gravitationssimulator. Der **Standard-Modus**
ist eine einzige HTML-Datei — kein Build, kein Backend, einfach im Browser
öffnen. Für massive Szenarien (Hunderttausende Körper) gibt es ein
**optionales CUDA-Backend** mit Multi-GPU-Physik und einem entkoppelten
Film-Modus.

![Render](https://img.shields.io/badge/render-Canvas%202D%20%2B%20WebGL-blueviolet)
![Standalone](https://img.shields.io/badge/frontend-single--file-success)
![CUDA](https://img.shields.io/badge/optional-CUDA%20multi--GPU-76b900)
![Mobile](https://img.shields.io/badge/mobile-ready-orange)

## Features

- **16 vordefinierte Szenarien**: Sonnensystem, Sonnensystem mit
  Asteroidengürteln (Asteroiden- und Kuipergürtel, Anzahl per Slider
  konfigurierbar), TRAPPIST-1, Alpha Centauri, Kepler-16, Kepler-47
  (Doppelstern + 3 Planeten), Trisolaris (3 Sonnen), Lagrange-
  Konstellationen (stabil/instabil), Trojaner (L4), Figur-8-Choreografie,
  Butterfly I, Moth I, Goggles, Yarn, „Leeres System" zum freien Bauen
- **Störmassen interaktiv injizieren** — Position, Masse (10⁻³ bis 10⁶
  Erdmassen, inkl. Sternen ab ~80 M⊕), Geschwindigkeit und Richtung
- **Asteroiden-Wolken injizieren** — ganze Schwärme aus Kleinkörpern mit
  Anzahl (logarithmisch **bis 50 000 pro Injektion**), Gesamtmasse, Dichte
  und Streuung per Slider; Shift+Klick auf den Inject-Button (bzw. langer
  Druck auf Mobile) spawnt statt einer Rogue eine Wolke
- **Fünf Physik-Engines** per Toggle — Hauptthread, WebWorker, WebGPU,
  Hybrid und ein natives **CUDA-Backend (f64, Multi-GPU)**, siehe
  [Physik-Engines](#physik-engines)
- **Film-Modus** — Compute und Darstellung vollständig entkoppelt: die
  GPU rechnet mit maximalem Durchsatz voraus, der Browser spielt die
  Timeline wie ein Video (scrubben, zurückspulen, live folgen), siehe
  [Film-Modus](#film-modus)
- **Asteroiden-Kollisionen** (optional per Toggle) — Asteroiden prallen
  wie Kugeln unterschiedlicher Masse voneinander ab (massenabhängiger
  Slingshot: der schwerere wird blau, der leichtere rot markiert und
  fliegt weg). Beim Treffer auf einen Planeten/Stern wird ein Asteroid
  absorbiert; Körper × Körper mit hoher Relativgeschwindigkeit **zerbirst**
  in Fragmente. Alle drei Pfade greifen live wie im Film
- **Echtzeit-N-Body-Integration** mit konfigurierbarem Zeitschritt und
  Verlangsamungsfaktor; Pause, Reset und Einzelschritt
- **Log-Zoom-Modus** für gleichzeitige Sicht auf Sonne und äußere Planeten
- **Bahnspuren** (variable Länge), **Kraftvektoren**, **Schwerpunkt-
  Zentrierung** und **Gitternetz** ein-/ausblendbar
- **Konfigurationen** lokal speichern, exportieren und importieren
- **Live-Statistik**: kinetische / potenzielle / Gesamtenergie, Drehimpuls,
  Schwerpunkt-Drift, Fluchtkurs-Erkennung; Live-Benchmark-Badge (FPS,
  Tage/s, aktive Engine)
- **Mobile vollständig unterstützt** — Long-Press, Pinch-Zoom,
  2-Finger-Pan, dedizierte mobile Toolbar und Bottom-Sheet
- **Mehrsprachig Deutsch / Englisch** — Toggle oben links im Seitenmenü,
  Sprachwahl bleibt im `localStorage` erhalten

## Physik-Engines

Die Engine ist im Seitenmenü umschaltbar; die Wahl bleibt im
`localStorage` erhalten.

| Engine | Läuft auf | Eignung |
|---|---|---|
| **Hauptthread (CPU)** | Browser, ein Thread | Debug/Referenz |
| **WebWorker** | Browser, eigener Thread (f64) | Standard — auf den meisten Geräten der schnellste browserinterne Pfad |
| **WebGPU** | integrierte/dedizierte GPU (f32) | starke GPUs; auf schwacher iGPU langsamer als der Worker |
| **Hybrid** | Worker + gezielt genauere Asteroiden-Substeps | nahe Begegnungen präziser |
| **CUDA-Backend** | NVIDIA-GPUs, nativ f64, Multi-GPU | Hunderttausende Körper, Film-Modus |

Interessanterweise ist der **WebWorker** auf vielen (auch schwächeren)
Rechnern der schnellste browserinterne Pfad: native f64-Arithmetik auf der
CPU schlägt bei ~7 000 Körpern mit nur ~10 massiven Quellen eine schwache
iGPU, deren Dispatch-/Sync-Latenzen pro Substep die eigentliche Rechenzeit
übersteigen. Erst eine starke GPU (WebGPU) oder das CUDA-Backend drehen das
Verhältnis.

## Film-Modus

Nur mit aktivem CUDA-Backend. Compute und Darstellung sind vollständig
entkoppelt:

- Ein eigener **GPU-Producer-Prozess** rechnet mit vollem Durchsatz in
  einen Ringpuffer voraus und stoppt nur, wenn er zu weit vorläuft.
- Der Browser liest den Puffer als **push-gestreamten Punktfilm** und
  spielt die Timeline wie ein Videoplayer: **scrubben, zurückspulen, live
  folgen** an der Produktionskante.
- Übertragen werden nur Positionen als **Integer-Bildschirmkoordinaten**
  (server-seitiges View-Culling, u16-Quantisierung); Masse/Sichtbarkeit
  laufen als kompakte Ereignisse.
- **Dichte-LOD**: Reicht das Punktbudget nicht für alle Körper, gilt eine
  Rangfolge — massive Körper (Sonne, Planeten, Rogues, Sterne, Schwarze
  Löcher) werden **nie** ausgedünnt, danach kommen die Asteroiden des
  geladenen Systems, den Rest bekommen nachträglich injizierte Wolken.
  Innerhalb jeder Stufe wird **dichteabhängig** gedünnt (behalten ~
  Anzahl^0,5 je Gitterzelle): ein Gürtel aus wenigen hundert Objekten
  verschwindet dadurch nicht neben einer Wolke aus hunderttausenden,
  während dichte Gebiete sichtbar dichter bleiben. Die Auswahl läuft über
  einen Hash des Körperindex — stufenlos, ohne Rasterartefakte, und über
  Samples hinweg stabil. Das Budget ist per Regler einstellbar
  (Auto … „Alle"). **Physik und Kollisionen laufen immer für alle
  Körper** — ausgedünnt wird nur die Darstellung.
- Kollisions-Ereignisse (Merge, Kill, Bounce, Zerbersten) tragen ihren
  **exakten Ort** und werden zeitrichtig als Explosionen abgespielt.
- Beim Verlassen des Films oder Engine-Wechsel wird der exakte
  f64-Zustand an die neue Engine übergeben — kein Impulsverlust.

So bleibt eine Wolke aus **> 250 000 Körpern** flüssig abspielbar, während
die GPU im Hintergrund weiterrechnet.

## CUDA-Backend

Optional, für die letzten beiden Engines und den Film-Modus. Benötigt
NVIDIA-GPU(s) mit CUDA-12-Treiber. Das Frontend funktioniert ohne Backend
vollständig (WebWorker/WebGPU) — das Backend schaltet nur die CUDA-Engine
und den Film-Modus frei.

### Kernpunkte

- **Native f64-Physik** in einem CuPy/NVRTC-Kernel (Yoshida-4.-Ordnung,
  identisch zur Worker-Referenz), der ganze Frame läuft als ein
  cooperative Launch (Grid-Sync).
- **Multi-GPU-Sharding, hardwareagnostisch**: die Asteroiden werden
  gewichtet nach f64-Score auf alle tauglichen Karten verteilt, die
  massiven Körper sind repliziert und bleiben bit-identisch. Die
  Substep-Synchronisation läuft atomics-frei über PCIe-gemappten
  Host-Speicher — kein NVLink nötig. Mit einer Karte degeneriert dieselbe
  Codebahn ohne Overhead.
- **Hierarchische Zeitschritte**: nur sonnennahe Körper („Taucher")
  laufen in privaten Feinschleifen, der Rest im groben Raster — statt dass
  ein einziger Sonnenstürzer die Rate aller Körper drückt.
- **Kollisions-/Bounce-Erkennung pipelined** auf eigenen GPUs
  (f32-Vorfilter + exakte f64-Nachprüfung), überlappt mit der Physik.
  Die Bounce-Suche ist der Engpass (75–93 % der Batchzeit) und wird
  **räumlich auf bis zu zwei Karten aufgeteilt**: jede prüft einen
  x-Streifen plus einen Halo und behält nur Paare, deren linkerer Partner
  ihr gehört — lückenlos, überschneidungsfrei, ohne Datenaustausch
  zwischen den Karten. Ob sich das lohnt, hängt an der **Kandidatenzahl
  pro Sample**, nicht an der Körperzahl: dieselben 250 k Asteroiden
  ergeben als dichter Klumpen über 10⁹ zu prüfende Paare, als
  ausgebildeter Gürtel unter 10⁶. Der Producer schaltet die zweite Karte
  deshalb **pro Batch** zu und ab (Schwellen mit Hysterese, gemessener
  Gewinn bis 1,7×) — ohne Filmneustart.
- **Server-Lebenszyklus vom Browser gesteuert**: Auswahl der CUDA-Engine
  startet den Server on-demand (optional per systemd socket activation),
  Abwahl beendet ihn nach kurzer Leerlauffrist — GPUs sind im Ruhezustand
  frei.

### Setup & Start

```bash
cd backend
# venv im Projekt-Root, mit CuPy für CUDA 12 (Toolkit im Wheel enthalten):
python3 -m venv ../venv
../venv/bin/pip install "cupy-cuda12x[ctk]" websockets numpy
# Server starten (Default-Port 8765, wählt automatisch die beste f64-GPU):
../venv/bin/python server.py
```

Danach im Browser die Engine **„CUDA-Backend (nativ, f64)"** wählen; das
Frontend verbindet sich per WebSocket (lokal `127.0.0.1:8765`, remote über
einen Reverse-Proxy).

Nützliche Schalter: `--ring-gib` (Größe des Film-Ringpuffers, bestimmt die
navigierbare Vergangenheit), `--det-gpus` (Erkennungskarten pro Session),
`--diag` (Zeitanteile von Producer und Stream-Loop mitloggen — für die
Engpass-Suche, im Normalbetrieb aus).

Tests und Messwerkzeuge (Standalone-Skripte, kein pytest):

```bash
cd backend
../venv/bin/python test_kernel.py             # Kernel gegen NumPy, 1/2/3 GPUs
../venv/bin/python test_film_golden.py        # Kollisionskette Ende-zu-Ende
../venv/bin/python test_erkennung_streifen.py # Streifen == eine Karte
../venv/bin/python test_lod_dichte.py         # Rangfolge + Dichte-Auswahl
../venv/bin/python bench_erkennung.py         # Umschaltschwelle der 2. Karte
../venv/bin/python bench_film.py --szene knoedel --det-gpus 1 2
```

## Bedienung

### Desktop

| Eingabe                       | Aktion                                   |
|------------------------------|------------------------------------------|
| Linksklick + ziehen           | Position + Geschwindigkeitsvektor setzen |
| Rechts-/Mittelklick + ziehen  | Ansicht verschieben (Pan)                |
| Mausrad                       | Zoom                                     |
| `P`                           | Pause / Weiter                           |
| `R`                           | Szenario zurücksetzen                    |
| `Leertaste`                   | Einzelschritt (im Pause-Modus)           |

### Mobile

| Geste                       | Aktion                                |
|----------------------------|---------------------------------------|
| 1 Finger lang halten        | Position armieren (Long-Press)        |
| Lang halten + ziehen        | Geschwindigkeit setzen                |
| 2 Finger pinch              | Zoom                                  |
| 2 Finger ziehen             | Ansicht verschieben                   |
| Untere Toolbar              | Pause, Zoom ±, Reset, Zentrieren      |
| Rechte Aktions-Buttons      | Fliehende / Störmassen / Wolken entfernen |
| Roter FAB unten rechts      | Injizieren (langer Druck: Wolke)      |
| Hamburger oben links        | Einstellungen-Sheet                   |

Die Sonnensystem-Koordinate einer geplanten Injektion bleibt während
Zoom, Pan und Pinch stabil — der Marker zeigt sie vor jedem „Injizieren"-
Klick an, und ein erneuter Klick spawnt eine weitere Masse exakt an
derselben Stelle.

## Lokal starten

Für den Standard-Modus reicht jeder statische Webserver, z. B.:

```bash
python3 -m http.server 8080
# Browser: http://localhost:8080/
```

Alternativ einfach `index.html` direkt im Browser öffnen (`file://` —
die Simulation selbst läuft komplett offline; nur die optionalen
Footer-Elemente (GoatCounter-Pageview-Zähler, GitHub-Stars-Badge)
sind dann inaktiv). Für die CUDA-Engine + Film-Modus siehe
[CUDA-Backend](#cuda-backend).

## Browser-Performance

Mit großen Asteroidengürteln (~1400 Körper) gibt es Unterschiede zwischen
den Browsern. Für den browserinternen Betrieb (ohne CUDA-Backend):

**Desktop:**

| Browser | Empfehlung | Bemerkung |
|---|---|---|
| **Edge / Chrome** | WebGPU **an** (starke GPU) oder WebWorker | Dawn-Backend + DirectX-12-Pfad; auf schwacher iGPU ist der Worker oft schneller |
| **Firefox** | WebWorker | `wgpu`-Backend noch unreif; Canvas-Operationen in Firefox teurer pro Call |

**Android:**

| Browser | Empfehlung | Bemerkung |
|---|---|---|
| **Chrome** | WebWorker oder WebGPU | Auf den meisten Geräten liefert der Worker die stabilsten 60 fps |
| **Firefox** | WebWorker | Schwächeres `wgpu`-Backend |

Bei wenigen Bodies (Standard-Szenarien ohne Gürtel) macht der Unterschied
keine Rolle; alle Browser laufen flüssig auf 60 fps. Für Hunderttausende
Körper führt kein Weg am CUDA-Backend vorbei.

## Technik

### Frontend (single-file)

- **Reines HTML / CSS / Vanilla-JS** in einer Datei; Asteroiden werden per
  **WebGL-Punkt-Batch** in einem Draw-Call gerendert (statt zehntausender
  Canvas-Calls), Rest über Canvas-2D.
- **N-Body-Integration** mit symplektischem Verlet (Yoshida-4. Ordnung);
  asteroidenreiche Szenarien überspringen die Asteroid-Asteroid-Gravitation
  für O(N)- statt O(N²)-Aufwand.
- **Kontinuierliche Kollisionserkennung (CCD)** über parametrischen
  Bahnschnitt — schnelle Begegnungen (Wolke gegen Gürtel im Gegenlauf)
  werden zum Kontaktzeitpunkt aufgelöst, ohne dass Körper hindurchtunneln.
- **Keine Build-Pipeline, kein npm, keine Bibliotheken** — die Sim-Engine
  lädt nichts nach. Nur der Footer holt optional GoatCounter (privacy-
  friendly Pageview-Zähler) und den GitHub-Stars-Wert.
- **Netz-Worker im Film-Modus**: WebSocket-Empfang und Frame-Zerlegung
  laufen in einem eigenen Worker, die Nutzlast wird transferiert statt
  kopiert. Sonst blockiert der Renderloop des Hauptthreads das Leeren des
  Sockets, das TCP-Empfangsfenster kollabiert und die Leitung wird nicht
  ausgelastet.
- **`localStorage`** für gespeicherte Konfigurationen und UI-Settings.

### Backend (optional, `backend/`)

- **Python + CuPy/NVRTC** — `server.py` (WebSocket-Server, Session-
  Verwaltung, v4-Integer-Streaming mit Culling und LOD),
  `nbody_kernel.py` (Multi-GPU-f64-Kernel mit hierarchischen
  Zeitschritten), `film_producer.py` (Producer-Prozess: Ringpuffer,
  Kollisions-/Bounce-/Zerberst-Erkennung).
- **Prozess-Split**: der Producer besitzt die GPU exklusiv, der Server
  beantwortet Puffer-Anfragen aus dem Shared Memory in Mikrosekunden — die
  Physik-GPU verhungert nie am GIL.
- Tests: `test_kernel.py` (Kernel gegen NumPy-Referenz, 1/2/3-GPU +
  Benchmarks), `test_film_golden.py` (Kollisionskette Ende-zu-Ende),
  `test_erkennung_streifen.py` (die räumlich aufgeteilte Bounce-Suche
  liefert exakt dieselben Treffer wie eine einzelne Karte),
  `test_lod_dichte.py` (Rangfolge und dichteabhängige Punktauswahl).
  Messwerkzeuge: `bench_erkennung.py`, `bench_film.py`.

## Stargazers über die Zeit

[![Star History Chart](https://api.star-history.com/svg?repos=Peuqui/SolarSystemSimulator&type=Date)](https://star-history.com/#Peuqui/SolarSystemSimulator&Date)

## Lizenz

MIT
