# Übergabeprotokoll — Stand 2026-07-20

Arbeitsstand nach einer langen Optimierungs- und Fehlersuchsitzung am
Film-Modus. Commits: `4916dd9` (Fixes), `462d9a7` (Abspieltempo, Lint).

---

## 1. Gemessene Kennzahlen (Referenz für künftige Vergleiche)

Szene: 267k Körper, Raster 0,5 Tage, drei Tesla V100 (Physik) + eine
Quadro RTX 8000 (Erkennung), Client über LAN.

| Posten | Anteil | Bemerkung |
|---|---|---|
| Bounce-Erkennung (`analyze_bounce`) | **75–93 %** | der Engpass |
| Physik (`step_batch`, 3× V100) | 4–20 % | |
| Merge-Erkennung | 2–4 % | |
| Anwenden der Kollisionen | 1–3 % | |
| Ring-Schreiben (`tobytes`) | 1–5 % | |

**Produktionsrate:** 60k → 120 Sim-Tage/s · 117k → 16 · 267k → 3–6
(stark von der räumlichen Verteilung abhängig, nicht nur von N).

**Kandidatenpaare der Bounce-Erkennung:** 46–105 Millionen **pro Sample**.

**Client-Grenze:** Chrome nimmt nur ~35 Mbit/s ab (`rwnd_limited 63 %`,
Empfangsfenster auf 44 KB kollabiert), während die Leitung 856 Mbit/s
liefert. Der Client ist das Nadelöhr, nicht das Netz.

**PCIe:** Alle fünf Karten hängen mit **×4** an M.2-zu-Oculink- bzw.
USB4-Adaptern (Mini-PC). Hardwarebedingt, nicht änderbar.

---

## 2. Offene Punkte, nach Priorität

### 2.1 Kollisionserkennung räumlich auf beide RTX 8000 verteilen
**Größter verbleibender Hebel, erwartet Faktor 2.**

Die zweite RTX 8000 liegt brach. Kollisionen sind lokal — zwei Körper auf
gegenüberliegenden Seiten der Sonne können sich in einem Zeitschritt nicht
treffen. Also: Raum halbieren, jede Karte prüft eine Hälfte plus einen
Halo in Breite der Suchreichweite (~0,02 AE, verschwindend gegen die
Szenengröße).

**Kein Kommunikationsbedarf zwischen den Karten:** Beide bekommen dieselbe
Schnittgrenze (Host berechnet sie, ein Skalar pro Sample) und entscheiden
allein, welche Treffer ihnen gehören:

```
Paar (i,j) gehört zu Karte A, wenn min(x_i, x_j) < grenze, sonst zu B
```

Ein Paar im Halo wird von beiden gefunden, aber nur von einer behalten.
Zusammenführung am Ende auf dem Host (Trefferlisten sind klein).

**Variante:** Beide Karten bekommen *alle* Daten und filtern selbst
(GPU-seitig). Die Alternative — Masken auf dem Host bilden und nur die
jeweilige Hälfte übertragen — spart Transfer, kostet aber CPU-Zeit im
Erkennungs-Thread, und genau dort klemmt der GIL. Empfehlung: GPU-seitig
filtern, danach messen.

**Berührt fünf Stellen** in `film_producer.py`: Device-Auswahl (~Z. 85),
Zustandsarrays je GPU (`g_ast`, `g_vis_det`, `g_rr_det`, ~Z. 106),
`analyze_batch` (Transfer), `analyze_sample`, plus Threading und
Zusammenführung.

⚠️ **Ein Anlauf wurde am 20.07. begonnen und zurückgerollt** — ein
halbfertiger Stand ließ die Golden-Tests fallen („A: kein Bounce
erkannt"). Braucht durchgehende Aufmerksamkeit, nicht nebenbei.

### 2.2 Datenlokalität: GPU-zu-GPU statt über den Host
Pro Batch werden ~34 MB von den V100 zum Host und weiter zur RTX kopiert
(≈17 ms bei ×4). Heute unter 2 % der Batchzeit, nach 2.1 aber ~11 %.
Direkter Karte-zu-Karte-Transfer halbiert den Weg.

Voraussetzung: Die Erkennung muss ohne Host-Rückweg auskommen —
`analyze_merge` bekommt heute Host-Arrays, `bounce_deltas` rechnet
komplett auf der CPU (kostet aber nur 1 %).

### 2.3 Client-Worker
WebSocket-Empfang und Dekodierung in einen eigenen Thread. Der Main-Thread
dekodiert derzeit ~700 KB je Sample (rund 89 000 Körper) und rendert
gleichzeitig — daher das kollabierte Empfangsfenster. Hebt die 35-Mbit/s-
Grenze an, statt nur darunter zu bleiben.

### 2.4 Aufräumen (erst wenn die Optimierung durch ist)
- Diagnose-Logging in `film_producer.py` (Zeitanteile, Kandidatenzahlen,
  Dispersion) und `server.py` (`[stream]`-Zeilen) entfernen. Kostet
  Rechenzeit (`cp.unique`, zwei `bincount`, zwei `percentile` je Sample).
- **Socket-Unit steht auf `0.0.0.0:8765`** statt `127.0.0.1` — Testrest
  der Direktverbindung, das Backend ist damit ohne Auth im LAN erreichbar.
  Original gesichert; auf Wunsch des Nutzers vorerst so belassen.
- `?cudaws=ws://host:8765` im Client hängt den nginx-Proxy ab (Diagnose).
  Kann bleiben, stört im Normalbetrieb nicht.

### 2.5 nginx (braucht sudo)
- `sites-available/narnia` ist seit Mai **nicht mehr synchron** mit
  `sites-enabled/narnia`. Die aktive Datei ist `sites-enabled` (kein
  Symlink!). Zwei Wahrheiten für dieselbe Config.
- `/solar-system/` hat **kein `Cache-Control`** — anders als
  `/haemotrace/` und `/ai-atc/`. Deshalb ist nach jeder Änderung an
  `index.html` ein harter Reload nötig.

---

## 3. Verworfen — nicht erneut versuchen

### 3.1 Bounce-Reichweite auf lokale Dispersion umstellen
Die Suchreichweite `h = 2·v95·dt` richtet sich nach der **absoluten**
Bahngeschwindigkeit, obwohl für Kollisionen nur die **relative** zählt.
Idee: Reichweite aus lokaler Geschwindigkeitsdispersion ableiten.

**Gemessen und verworfen:** Das Maximum der Dispersion liegt 40- bis
100-mal über dem 95-Perzentil (`disp95 = 1,8` gegen `disp_max = 87…209`).
Eine garantiert korrekte Reichweite wäre rund **vierzigmal größer** als
die heutige, nicht kleiner.

**Nebenbefund, wichtig:** Der bestehende Ansatz deckt damit nur die
langsamsten 95 % ab. Kollisionen sehr schneller Körper — Sonnennähe
(`v = √(GM/r)`), durch Stöße ausgeschleuderte Asteroiden — werden
**bereits heute nicht gefunden**. Bewusste Näherung, kein Regressionsfehler,
aber eine bekannte Genauigkeitsgrenze.

### 3.2 Multi-GPU-Schwelle senken
`test_kernel.py` legte nahe, eine Einzelkarte sei schneller als drei
(136 gegen 90 Tage/s bei 200k). Im Betrieb umgestellt → **fünf- bis
sechsmal langsamer** (1,7 gegen 11 Tage/s). Zurückgenommen.

Der Benchmark lief unter paralleler Last; eine saubere Wiederholung ergab
**244 gegen 198 Tage/s zugunsten des Verbunds**. Die Schwelle von 30k in
`film_producer.py` ist korrekt. Warnung steht im Code.

---

## 4. Betriebswissen (spart Fehldiagnosen)

**Was wirkt wann:**

| Geändert | Wirksam durch |
|---|---|
| `index.html` | **Harter Reload** (Strg+Shift+R) — kein Cache-Control! |
| `film_producer.py` | Film aus/an (neuer Prozess per `spawn`) |
| `server.py` | Backend-Prozess muss beendet werden |

Der Server läuft socket-aktiviert mit `--idle-exit 120`: Er beendet sich
erst **120 s nach der letzten getrennten Verbindung**. Ein Reload reicht
nicht — der Browser verbindet sofort neu und setzt den Timer zurück.
Sauber: `systemctl --user stop solar-cuda.service`.

**Prozesse korrekt zählen** (`pgrep -af "server.py"` matcht die eigene
Shell-Zeile mit!):
```bash
pgrep -c -f "SolarSystemSimulator/venv/bin/python server.py"
ps -eo pid,cmd --no-headers | grep "SolarSystemSimulator/venv/bin/python -c from multiprocessing" | grep -vc grep
```

**Logs:** `journalctl --user -u solar-cuda.service -f`

**Tests:** Standalone-Skripte, kein pytest:
```bash
cd backend && ../venv/bin/python test_film_golden.py
cd backend && ../venv/bin/python test_kernel.py
```

**ruff** liegt nicht im Projekt-venv:
`/home/mp/Projekte/AIfred-Intelligence/venv/bin/ruff check backend/`

**GPU-Nummerierung:** `nvidia-smi` zählt nach PCI-Bus, CUDA nach
Rechenleistung. Was der Producer als „gpus [0,1,2]" meldet, sind die drei
V100 — physisch die Karten 1, 3 und 4. Kein Fehler.

**Messfallen, die heute mehrfach zugeschlagen haben:**
- Ein einzelner `nvidia-smi`-Aufruf misst in einem Stop-and-Go-System
  zufällig eine Pause. Immer Zeitreihen nehmen.
- Die Szene verändert sich laufend (Kollisionen, Verdichtung). Zwei
  Messungen sind nur vergleichbar, wenn die Verteilung ähnlich ist —
  dieselbe Objektzahl genügt **nicht**.
- Vor jeder Messung prüfen, ob der laufende Prozess den geänderten Code
  überhaupt geladen hat (Startzeit gegen `stat -c '%y'` der Datei).

---

## 5. Am 20.07. behoben (zur Einordnung künftiger Regressionen)

- **Live-Deadlock:** `_filmHeadRate` wird aus der Kantenbewegung
  geschätzt; stand die Kante, fiel sie auf 0 und der Playhead blieb
  stehen — was dem Server „Puffer voll" bestätigte. Gegenseitiges Warten.
- **Handover-Race:** `filmStart()` legte neue `filmRefs` an, während der
  f64-Dump der alten Session unterwegs war → verworfen → halb
  aktualisierter Zustand → Asteroiden sprangen auf Filmstart-Positionen.
  Jetzt wird der Start aufgeschoben und vom Dump ausgelöst.
- **Neustart-Endlosschleife:** `_filmSentN` wurde auf die gefilterte
  Länge gesetzt, verglichen aber gegen `bodies.length` → in jedem Frame
  ein Neustart.
- **Playhead-Klemmung** im Server (`[tail, head]`) gegen stilles Einfrieren.
- **Interpolation:** Körper nur im älteren Sample bekamen dessen Position
  und galten als aktuell → statische Flecken.
- **Bandbreitenschätzung:** `_bw` maß die Dauer von `await ws.send`
  (Socket-Puffer!) — 724 MB/s statt real ~4. Jetzt über ein Fenster.
- **Frame-Kosten:** Drosselung rechnete mit `sample_bytes` (8·n) statt der
  tatsächlichen Framegröße.
- **Leichen-Filter:** Tote Asteroiden werden beim Filmstart aussortiert.
- **Ringpuffer** per `--ring-gib` konfigurierbar (Default 8 GiB).
