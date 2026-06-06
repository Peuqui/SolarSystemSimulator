# Solar System Simulator

Deutsch | [English](README.en.md)

Browser-basierter N-Körper-Gravitationssimulator. Eine einzige HTML-Datei,
kein Build, kein Backend — einfach im Browser öffnen oder über jeden
statischen Webserver ausliefern.

![Sonnensystem-Szenario](https://img.shields.io/badge/render-Canvas%202D-blueviolet)
![Standalone](https://img.shields.io/badge/single--file-100kB-success)
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
  Anzahl, Gesamtmasse, Dichte und Streuung per Slider; Shift+Klick auf den
  Inject-Button (bzw. langer Druck auf Mobile) spawnt statt einer Rogue
  eine Wolke
- **Elastische Asteroiden-Kollisionen** (optional per Toggle) — Asteroiden
  prallen wie Kugeln unterschiedlicher Masse voneinander ab, mit massen-
  abhängigem Slingshot (der leichtere wird vom schwereren weggeschleudert).
  Sie verschmelzen **nicht**, zerbersten **nicht** und ziehen sich
  untereinander **nicht** gravitativ an — beim Stoß wird der schwerere blau,
  der leichtere rot markiert und fliegt sichtbar weg. Erst beim Treffer auf
  einen Planeten oder Stern wird ein Asteroid absorbiert bzw. löst
  Verschmelzung/Zerbersten aus
- **Echtzeit-N-Body-Integration** mit konfigurierbarem Zeitschritt und
  Verlangsamungsfaktor; Pause, Reset und Einzelschritt
- **Log-Zoom-Modus** für gleichzeitige Sicht auf Sonne und äußere Planeten
- **Bahnspuren** (variable Länge), **Kraftvektoren**, **Schwerpunkt-
  Zentrierung** und **Gitternetz** ein-/ausblendbar
- **Konfigurationen** lokal speichern, exportieren und importieren
- **Live-Statistik**: kinetische / potenzielle / Gesamtenergie, Drehimpuls,
  Schwerpunkt-Drift, Fluchtkurs-Erkennung
- **Mobile vollständig unterstützt** — Long-Press, Pinch-Zoom,
  2-Finger-Pan, dedizierte mobile Toolbar und Bottom-Sheet
- **Mehrsprachig Deutsch / Englisch** — Toggle oben links im Seitenmenü,
  Sprachwahl bleibt im `localStorage` erhalten
- **Optionale GPU-Beschleunigung** — Physik kann wahlweise im WebWorker
  oder via WebGPU Compute Shader laufen, jeweils per Toggle umschaltbar

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
| Toolbar-Knöpfe              | Pause, Zoom ±, Injizieren, Reset      |

Die Sonnensystem-Koordinate einer geplanten Injektion bleibt während
Zoom, Pan und Pinch stabil — der Marker zeigt sie vor jedem „Injizieren"-
Klick an, und ein erneuter Klick spawnt eine weitere Masse exakt an
derselben Stelle.

## Lokal starten

Es reicht jeder statische Webserver, z. B.:

```bash
python3 -m http.server 8080
# Browser: http://localhost:8080/
```

Alternativ einfach `index.html` direkt im Browser öffnen (`file://` —
die Simulation selbst läuft komplett offline; nur die optionalen
Footer-Elemente (GoatCounter-Pageview-Zähler, GitHub-Stars-Badge)
sind dann inaktiv).

## Browser-Performance

Mit großen Asteroidengürteln (~1400 Körper) gibt es deutliche
Unterschiede zwischen den Browsern:

**Desktop (Reihenfolge schnell → langsam):**

| Browser | Empfehlung | Bemerkung |
|---|---|---|
| **Edge** | WebGPU **an** | Aktuelles Dawn-Backend + DirectX-12-Pfad — flüssigste Performance unter Windows |
| **Chrome** | WebGPU **an** | Ähnlich zu Edge, je nach Treiber etwas langsamer; Worker als Fallback |
| **Firefox** | WebGPU **aus** (Worker an) | `wgpu`-Backend noch unreif; Canvas-2D-Path-Operationen sind in Firefox 2–5× teurer pro Call — der WebGPU-Roundtrip kommt erschwerend dazu |

**Android (Reihenfolge schnell → langsam):**

| Browser | Empfehlung | Bemerkung |
|---|---|---|
| **Chrome** | WebGPU **an** | Deutlich performanter als Firefox auf Android |
| **Firefox** | WebGPU **aus** (Worker an) | Auch mobil das schwächere `wgpu`-Backend |

Bei wenigen Bodies (Standard-Szenarien ohne Gürtel) macht der Unterschied
keine Rolle; alle Browser laufen flüssig auf 60 fps.

## Technik

- **Reines HTML / CSS / Canvas-2D / Vanilla-JS** in einer einzigen Datei
- **N-Body-Integration** mit symplektischem Verlet-Schritt (Yoshida-4.
  Ordnung); asteroidenreiche Szenarien überspringen die Asteroid-Asteroid-
  Gravitation für O(N)-Aufwand statt O(N²)
- **Kontinuierliche Kollisionserkennung (CCD)** über parametrischen
  Bahnschnitt — schnelle Begegnungen (Wolke gegen Gürtel im Gegenlauf)
  werden erkannt und korrekt zum Kontaktzeitpunkt aufgelöst, ohne dass
  Körper durcheinander hindurchtunneln
- **Keine Build-Pipeline, kein npm, keine Bibliotheken** — die Sim-Engine
  selbst lädt nichts nach. Nur der Footer holt optional GoatCounter
  (privacy-friendly Pageview-Zähler) und den aktuellen GitHub-Stars-Wert
  vom GitHub-API; ohne Netz fehlt nur das Stars-Badge
- **`localStorage`** für gespeicherte Konfigurationen und UI-Settings

## Stargazers über die Zeit

[![Star History Chart](https://api.star-history.com/svg?repos=Peuqui/SolarSystemSimulator&type=Date)](https://star-history.com/#Peuqui/SolarSystemSimulator&Date)

## Lizenz

MIT
