# Solar System Simulator

Deutsch | [English](README.en.md)

Browser-basierter N-Körper-Gravitationssimulator. Eine einzige HTML-Datei,
kein Build, kein Backend — einfach im Browser öffnen oder über jeden
statischen Webserver ausliefern.

![Sonnensystem-Szenario](https://img.shields.io/badge/render-Canvas%202D-blueviolet)
![Standalone](https://img.shields.io/badge/single--file-100kB-success)
![Mobile](https://img.shields.io/badge/mobile-ready-orange)

## Features

- **15 vordefinierte Szenarien**: Sonnensystem, TRAPPIST-1, Alpha Centauri,
  Kepler-16, Kepler-47 (Doppelstern + 3 Planeten), Trisolaris (3 Sonnen),
  Lagrange-Konstellationen (stabil/instabil), Trojaner (L4), Figur-8-
  Choreografie, Butterfly I, Moth I, Goggles, Yarn, „Leeres System" zum
  freien Bauen
- **Störmassen interaktiv injizieren** — Position, Masse (10⁻³ bis 10⁶
  Erdmassen, inkl. Sternen ab ~80 M⊕), Geschwindigkeit und Richtung
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

## Technik

- **Reines HTML / CSS / Canvas-2D / Vanilla-JS** in einer einzigen Datei
- **N-Body-Integration** mit symplektischem Verlet-Schritt
- **Keine Build-Pipeline, kein npm, keine Bibliotheken** — die Sim-Engine
  selbst lädt nichts nach. Nur der Footer holt optional GoatCounter
  (privacy-friendly Pageview-Zähler) und den aktuellen GitHub-Stars-Wert
  vom GitHub-API; ohne Netz fehlt nur das Stars-Badge
- **`localStorage`** für gespeicherte Konfigurationen und UI-Settings

## Stargazers über die Zeit

[![Star History Chart](https://api.star-history.com/svg?repos=Peuqui/SolarSystemSimulator&type=Date)](https://star-history.com/#Peuqui/SolarSystemSimulator&Date)

## Lizenz

MIT
