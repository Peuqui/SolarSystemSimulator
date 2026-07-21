# Übergabeprotokoll

Was hier steht, steht **nicht** im Git-Log: Betriebswissen, Messfallen,
gescheiterte Ansätze und offene Punkte. Die Umsetzungshistorie ist
bewusst nicht enthalten — dafür ist `git log` da.

Stand: 2026-07-21, Commit `dea0857`.

---

## 1. Betriebswissen

**Was wirkt wann:**

| Geändert | Wirksam durch |
|---|---|
| `index.html` | **Harter Reload** (Strg+Shift+R) — kein Cache-Control! |
| `film_producer.py` | Film aus/an (neuer Prozess per `spawn`) |
| `server.py`, `nbody_kernel.py` | Backend-Prozess muss beendet werden |

Der Server läuft socket-aktiviert mit `--idle-exit 120`: Er beendet sich
erst **120 s nach der letzten getrennten Verbindung**. Ein Reload reicht
nicht. Sauber: `systemctl --user stop solar-cuda.service` — der Socket
lauscht weiter und startet ihn beim nächsten Verbindungsaufbau neu.

**Prozesse korrekt zählen** (`pgrep -af "server.py"` matcht die eigene
Shell-Zeile mit!):
```bash
pgrep -c -f "SolarSystemSimulator/venv/bin/python server.py"
```

**Läuft der geänderte Code wirklich?** Startzeit des Prozesses gegen die
Dateizeit prüfen:
```bash
ps -eo pid,lstart,cmd | grep "venv/bin/python server.py"
stat -c '%y %n' backend/server.py
```

**Logs:** `journalctl --user -u solar-cuda.service -f`
(Zeitanteile nur mit `--diag` in der Unit.)

**GPU-Nummerierung:** `nvidia-smi` zählt nach PCI-Bus, CUDA nach
Rechenleistung. Die drei V100 sind für CUDA 0–2, die beiden RTX 8000
sind 3 und 4. Kein Fehler.

**PCIe:** Alle fünf Karten hängen mit **×4** an M.2-zu-Oculink- bzw.
USB4-Adaptern (Mini-PC). Hardwarebedingt, nicht änderbar.

**ruff** liegt nicht im Projekt-venv:
`/home/mp/Projekte/AIfred-Intelligence/venv/bin/ruff check backend/`

---

## 2. Tests und Messwerkzeuge

Standalone-Skripte, kein pytest:

```bash
cd backend
../venv/bin/python test_film_golden.py      # Ende-zu-Ende Kollisionskette
../venv/bin/python test_kill_ort.py         # Kill-Abstand gegen Sternradius
../venv/bin/python test_bewegter_stern.py   # enge Begegnung, BEWEGTER Stern
../venv/bin/python test_subsamples.py       # Stützpunkte gegen NumPy-Referenz
../venv/bin/python test_film_protokoll.py   # Frame gegen das Worker-Schema
../venv/bin/python test_playhead_state.py   # Ring-Roundtrip, Sub-Überlauf
../venv/bin/python test_erkennung_streifen.py  # Streifen == eine Karte
../venv/bin/python test_kernel.py           # Kernel + Multi-GPU
../venv/bin/python bench_erkennung.py       # Umschaltschwelle der 2. Karte
../venv/bin/python bench_film.py -n 60000 --det-gpus 1
```

`test_film_protokoll.py` dekodiert ein echtes Frame nach demselben Schema
wie `zerlegeFilm` in `index.html` — dort laufen Server und Client am
ehesten auseinander, und zwar still: ein um vier Bytes verschobener
Offset liefert keine Fehlermeldung, sondern Positionen im Nichts.

**Blinder Fleck bis 21.07.:** Alle Tests hatten einen **ruhenden Stern im
Ursprung**. Ein Fehler in der Positions-Interpolation der massiven Körper
(die Feinschleife interpoliert sie linear über `dtH`) wäre nie
aufgefallen. `test_bewegter_stern.py` schließt die Lücke.

---

## 3. Messfallen

- **Die Kandidatenzahl ist die relevante Größe, nicht N.** Dieselben
  250k Asteroiden ergeben als Knödel 10⁹ und als Gürtel 10⁵ Paare pro
  Sample — Faktor 10⁴ in der Erkennungslast. Zwei Messungen mit gleicher
  Objektzahl sind darum **nicht** vergleichbar.
- `bench_film.py --szene knoedel` durchläuft diese Entwicklung: Der
  Gesamtwert am Ende ist irreführend, phasenweise nach Kandidatenzahl
  vergleichen.
- Ein einzelner `nvidia-smi`-Aufruf misst in einem Stop-and-Go-System
  zufällig eine Pause. Immer Zeitreihen nehmen.
- Vor jeder Messung prüfen, ob der laufende Prozess den geänderten Code
  geladen hat (siehe Abschnitt 1).
- Für A/B gegen einen früheren Stand: `git stash` → messen →
  `git stash pop`. Verlässlicher als der Vergleich mit dokumentierten
  Zahlen aus anderen Szenen.

**Diagnose-Vorgehen bei „die Szene sieht falsch aus":** Zuerst die
**Anfangsbedingungen** nachrechnen, dann erst die Pipeline verdächtigen.
Am 21.07. hat die Suche nach einem „Loch vor der Sonne" durch vier
Sackgassen geführt, weil in Interpolation, Streaming und Quantisierung
gesucht wurde. Ein Blick auf `v/v_esc` = 0,24 hätte in fünf Minuten
gezeigt, dass die injizierte Wolke gebunden ist und hineinfallen *muss*.
Das eigentliche Problem war ein ganz anderes (fehlende
Kollisionsanzeige).

---

## 4. Aktuelle Kennzahlen

Gemessen 21.07., 200 001 Körper, Raster 0,5 d, drei V100 (Physik) +
RTX 8000 (Erkennung):

| Szene | Produktion | Kernel | Wartet auf Erkennung | Bounce (GPU) | Host gesamt |
|---|---|---|---|---|---|
| Gürtel (2 Mio Kandidaten) | 46,8 d/s | 33 % | 59 % | 44 % | ~9 % |
| dichter Ring (49 Mio) | 12,6 d/s | 9 % | **88 %** | **84 %** | ~2 % |

**Die Erkennung ist der Taktgeber, und sie läuft vollständig auf der
GPU.** `bounceCPU` ist in beiden Fällen 0 % — es gibt viele Kandidaten,
aber wenige echte Treffer, und nur die kosten Host-Zeit. Auslagern lässt
sich nichts mehr; der einzige nennenswerte CPU-Posten ist das
Ring-Schreiben (9 % bei leichter Last, 2 % unter Last).

Produktionsrate über die Umbauten des 21.07.: 60,3 → 62,3 → 66,6 d/s
(60k, Gürtel, eine Erkennungskarte) — Unterschiede in der Streuung
zwischen Läufen.

**Nicht gemessen:** `build_frame` im Server-Prozess (Culling, Dichte-LOD
mit Bisektion und 512×512-`bincount`, Quantisierung, Sub-Kette) ist reine
numpy-Arbeit für jedes gestreamte Sample. Der Server ist ein eigener
Prozess, läuft also parallel zur GPU — ob er bei großem N selbst zum
Engpass wird, sagen die `[stream]`-Zeilen mit `--diag`.

---

## 5. Verworfen — nicht erneut versuchen

### 5.1 Bounce-Reichweite auf lokale Dispersion umstellen
`h = 2·v95·dt` richtet sich nach der **absoluten** Bahngeschwindigkeit,
obwohl für Kollisionen nur die **relative** zählt. Gemessen und
verworfen: das Maximum der Dispersion liegt 40- bis 100-mal über dem
95-Perzentil. Eine garantiert korrekte Reichweite wäre rund
**vierzigmal größer** als die heutige, nicht kleiner.

**Nebenbefund, wichtig:** Der bestehende Ansatz deckt damit nur die
langsamsten 95 % ab. Kollisionen sehr schneller Körper — Sonnennähe,
ausgeschleuderte Asteroiden — werden **bereits heute nicht gefunden**.
Bewusste Näherung, keine Regression, aber eine bekannte Genauigkeitsgrenze.

### 5.2 Multi-GPU-Schwelle senken
`test_kernel.py` legte nahe, eine Einzelkarte sei schneller als drei.
Im Betrieb umgestellt → **fünf- bis sechsmal langsamer**. Der Benchmark
misst den Kernel ohne die Erkennungs-Pipeline; sobald diese mitläuft,
kehrt sich das Bild um. Die Schwelle von 30k ist korrekt.

### 5.3 Client-seitige Bahnintegration
Der *Algorithmus* war gut (Blend-Fehler ~1e-6 gegen Ground-Truth), im
Browser aber zu teuer: Cache-Rebuild bei einer sonnennahen Wolke 87 ms
(5k Körper) bis 1,8 s (50k) → periodisches Einfrieren. Ersetzt durch
GPU-Stützpunkte, im Client ersatzlos gestrichen.

### 5.4 Dichte-Glättung im `_dichte_filter`
Box-Blur der Zellbelegung bringt nichts und schadet leicht (0,1924 gegen
0,1789 rel. Std bei homogenem Feld). Der Filter dämpft Poisson-
Fluktuation bereits über die lokale `rate`-Anpassung.

### 5.5 Schachbrettmuster = LOD-Gitter?
Widerlegt. Autokorrelation der Dichte im Log-Raum zeigt **keinen** Peak
bei der Zellperiode. Sichtbar sind großräumige diagonale Dichtebänder,
die auch **ohne** Ausdünnung da sind (echte Struktur).

### 5.6 Gröber streamen bei Zeitlupe
Wäre falsch: Man geht ja runter, um *mehr* Details zu sehen.

---

## 6. Offen

### 6.1 Ring-Schreiben ohne Host-Kopie
`schreibe_slot` kopiert `outs[i]` per `tobytes()` in den Shared-Memory-
Ring — 9 % der Producer-Zeit bei leichter Last. Der D2H könnte direkt in
den Ring gehen. Erwartbar gut 4 Tage/s im Gürtelfall, unter Last nichts.

### 6.2 Kill-Ort: letzte 2 Sternradien
Die Berührungsprüfung läuft seit `cdf5a43` je Substep in der Feinschleife
und trifft den Kontakt auf ~Radius/20. Der gemessene Median liegt bei
2,0 × Sternradius — der Rest steckt in der **linearen Interpolation der
Messung** (`test_kill_ort.py`), nicht mehr im Zeitpunkt. Im Client tragen
die Sub-Stützpunkte die Bahn achtmal feiner. Wer es genauer wissen will,
müsste den Test über `slot_sub` interpolieren lassen.

Nicht-heiße Körper durchlaufen die Feinschleife nicht; für sie gilt
weiter die Streckenprüfung im Producer (Kontaktzeitpunkt aus `tt`).

### 6.3 Datenlokalität GPU-zu-GPU
**Bewertet, trägt so nicht.** `step_batch` liefert bewusst ein
Host-Array, und das wird gebraucht — `bounce_deltas` rechnet auf der CPU,
`apply_merges` liest Host-Positionen. Ein P2P-Transfer ersetzte nur den
**H2D** zur Erkennungskarte, nicht den **D2H** von den Physikkarten.

Der billigere Hebel zuerst: **Pinned Host-Puffer** für `out` in
`step_batch`. Vorher messen — beim DOWNLOAD bringt pinned nach früherer
Messung nichts (3,21 gegen 3,24 GB/s, der ×4-Link ist ausgereizt).

### 6.4 Abrupter Verbindungsverlust im Film
Bricht die CUDA-Verbindung WÄHREND des Films unerwartet ab (Server-Crash,
`systemctl stop`), nullt der `onclose`-Pfad `filmQueue`, **ohne** vorher
`filmReconstructVelocities()` aufzurufen — die Körper landen im Worker
mit v≈0 und stürzen in die Sonne. Der geplante Engine-Wechsel ist sauber.
Fix wäre einzeilig, auf Nutzerwunsch nicht gebaut.

### 6.5 Rückspul-Ring in den VRAM
Der Sample-Ring liegt in `/dev/shm` (tmpfs = RAM). Die freien
Erkennungskarten (je 48 GB) hätten Platz für eine viel längere Historie.
**Haken:** Der Ring entkoppelt zwei Prozesse — der Server liest ihn heute
in µs direkt aus dem RAM. Im VRAM bräuchte es CUDA-IPC und pro
`build_frame` ein D2H. Vor dem Bau messen.

### 6.6 nginx (braucht sudo)
- `sites-available/narnia` ist seit Mai **nicht synchron** mit
  `sites-enabled/narnia`. Aktiv ist `sites-enabled` (kein Symlink!).
- `/solar-system/` hat **kein `Cache-Control`** — daher nach jeder
  `index.html`-Änderung ein harter Reload nötig.

### 6.7 Socket-Unit auf `0.0.0.0:8765`
Testrest der Direktverbindung; das Backend ist damit ohne Auth im LAN
erreichbar. Auf Wunsch des Nutzers vorerst so belassen.
`?cudaws=ws://host:8765` im Client hängt den nginx-Proxy ab (Diagnose)
und kann bleiben.

### 6.8 Ereignis-Anzeige: bekannte Grenzen
- Der Server liefert höchstens **1024 Ereignisse pro Frame**, der Client
  zeigt höchstens **32 Blitze pro Bild**. In dichten Szenen baut sich ein
  Rückstand auf; der Zähler springt schubweise.
- Läuft der Rückstand über den Ereignisring (65536), springt der Server
  vor (`ev_from = max(sent_ev, ev_total - ev_cap + 8)`) und Ereignisse
  gehen **still verloren**. Der Badge zählt dann dauerhaft zu niedrig.
