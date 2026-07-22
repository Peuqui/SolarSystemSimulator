# Übergabeprotokoll

Was hier steht, steht **nicht** im Git-Log: Betriebswissen, Messfallen,
gescheiterte Ansätze und offene Punkte. Die Umsetzungshistorie ist
bewusst nicht enthalten — dafür ist `git log` da.

Stand: 2026-07-21, Commit `39ffb50`.

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

**Die Karten sind nicht identisch — und das Datenblatt verrät es nicht.**
Eine der drei V100 (CUDA-Index 1, PCI 07:00.0) ist eine **SXM2-Karte im
PCIe-Umbau**, erkennbar am abweichenden VBIOS `88.00.7E.00.03` gegen
`88.00.48.00.02` der beiden anderen.

Gemessen (`miss_gewicht` in `selfgrav_kernel.py`, mehrfach und in beiden
Reihenfolgen gegengeprüft):

| | f32 | f64 |
|---|---|---|
| CUDA 0 (V100) | 4.759 | 1.490 |
| CUDA 1 (V100, SXM2-Umbau) | 4.739 | 1.486 |
| CUDA 2 (V100) | 4.840 | **1.655** |
| CUDA 3 (RTX 8000) | 5.294 | 155 |
| CUDA 4 (RTX 8000) | 4.174 | 154 |

Der Umbau kostet **nichts** — CUDA 1 liegt gleichauf mit CUDA 0. Auffällig
ist stattdessen **CUDA 2: in f64 elf Prozent schneller** als die beiden
anderen V100, in f32 nur zwei. Und die beiden RTX 8000 unterscheiden sich
in f32 um **21 %**, obwohl baugleich.

`f64_score` gibt allen drei V100 exakt denselben Wert und sieht davon
nichts. Wo Lasten verteilt werden, gehört darum gemessen — siehe 6.12.

**Hängt noch GPU-Speicher fest?**
```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
pkill -f 'SolarSystemSimulator.*multiprocessing'   # nur im Notfall
```
Nicht `nvidia-smi … | xargs kill` — auf der Maschine läuft auch
HaemoTrace mit multiprocessing. Seit `39ffb50` beendet sich der Producer
selbst, wenn sein Server verschwindet (Elternwächter, ~0,5 s), das sollte
also nicht mehr nötig sein.

**Injizieren tut nichts?** Bei 64 massiven Körpern ist Schluss — das ist
`M_MAX` im CUDA-Kernel, wo sie im Shared Memory liegen. Der Server nennt
die Zahl beim Handshake, der Client sperrt ab da und meldet es sichtbar.
Die Sperre gilt auch in den CPU-Engines, sobald ein Backend geantwortet
hat: sonst baut man im Worker eine Szene, die sich später nicht mehr auf
die GPU laden lässt. Asteroiden und Wolken sind nicht betroffen — sie
sind Testteilchen und zählen nicht gegen die Grenze.

**Zwei CUDA-Kernel, eine Verbund-Mechanik.** `nbody_kernel.py` (wenige
Massen, enge Begegnungen, f64) und `selfgrav_kernel.py` (zehntausende
selbstgravitierende Massen, geglättete Kraft) bleiben getrennt — sie
beantworten zwei verschiedene physikalische Fragen. Wie sie ihre Karten
koppeln, steht dagegen **einmal** in `gpu_verbund.py`: System-Barrier
(atomics-frei, siehe dort) und der Blick auf gemappten Host-Speicher.
Die Barrier nimmt einen `unsigned int*` statt eines Struct-Zeigers,
damit jedes Modul seinen eigenen Austauschbereich mitbringen kann.

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
../venv/bin/python test_waechter.py         # Runaway-Kills als kind 2
../venv/bin/python test_waisen.py           # GPU frei nach hartem Serverstod
../venv/bin/python test_producer_tod.py     # Producer-Absturz wird gemeldet
../venv/bin/python test_selfgrav.py         # Kernel A (selbstgravitierend)
../venv/bin/python test_lastregler.py       # Streifen-Lastregler (ohne GPU)
../venv/bin/python bench_erkennung.py       # Umschaltschwelle der 2. Karte
../venv/bin/python bench_film.py -n 60000 --det-gpus 1
```

`test_waisen.py` startet sich selbst als Server-Ersatz und schießt ihn mit
`SIGKILL` ab — er darf also ruhig „Prozess getötet" ins Log schreiben, das
ist der Testgegenstand.

`test_selfgrav.py` misst Kartenleistung und vergleicht Läufe bitgenau —
er verträgt darum **keine fremde GPU-Last**. Läuft nebenher noch eine
Film-Session (etwa aus einem offenen Browser-Tab), schlagen die
Kalibrier- und Determinismus-Fälle scheinbar grundlos fehl. Vorher
`pgrep -c -f "SolarSystemSimulator.*multiprocessing"` prüfen.

`test_producer_tod.py` deckt eine Klasse von Fehlern ab, die sich sonst
als „der Film startet einfach nicht" zeigt: Der Producer ist ein eigener
Prozess, sein Traceback landet nur im Server-Log, und `stream()` schleift
auf `running_val`, das allein der Server beim `stop()` zurücksetzt. Ohne
die `is_alive()`-Prüfung wartet der Stream auf ein `head_val`, das nie
mehr wächst — stumm, in 30-ms-Schritten, für immer.

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

- **Eine Regelgröße muss auf das reagieren, was man stellt.** Die Zeit je
  Erkennungskarte schien das ideale Lastmaß — sie enthält Arbeit *und*
  Kartenleistung. Sie taugt trotzdem nicht: Ein schmalerer Streifen
  behält seinen vollen Halo und wird darum nicht proportional schneller.
  Ein Regler darauf schiebt endlos weiter (siehe 6.12). Vor jedem
  Regelkreis prüfen, ob die Regelgröße auf die Stellgröße überhaupt
  proportional antwortet.
- **Die Kandidatenzahl ist die relevante Größe, nicht N.** Dieselben
  250k Asteroiden ergeben als Knödel 10⁹ und als Gürtel 10⁵ Paare pro
  Sample — Faktor 10⁴ in der Erkennungslast. Zwei Messungen mit gleicher
  Objektzahl sind darum **nicht** vergleichbar.
- `bench_film.py --szene knoedel` durchläuft diese Entwicklung: Der
  Gesamtwert am Ende ist irreführend, phasenweise nach Kandidatenzahl
  vergleichen.
- Ein einzelner `nvidia-smi`-Aufruf misst in einem Stop-and-Go-System
  zufällig eine Pause. Immer Zeitreihen nehmen.
- **Eine ruhende GPU misst zu langsam.** Sie steht in einer
  Idle-Taktstufe und braucht Sekundenbruchteile auf vollen Takt. In einer
  Messreihe über fünf Karten lag deshalb die *jeweils zuerst gemessene*
  10–20 % unter allen folgenden — die erste Fassung von `miss_gewicht`
  (`selfgrav_kernel.py`) bildete damit die Messreihenfolge ab statt der
  Leistung und hätte einer baugleichen Karte ein um ein Fünftel kleineres
  Segment gegeben. Daraus wurde fälschlich geschlossen, die V100 seien
  untereinander 20 % verschieden. Gegenmittel: mehrere Runden, und die
  **beste** zählt — Störungen (kalte Karte, fremde Last) wirken nur nach
  unten, das Maximum ist der robuste Schätzer, nicht der Mittelwert.
- **`utilization.gpu` ist kein Auslastungsmaß.** Es misst den Zeitanteil,
  in dem *mindestens ein Kernel resident* war — nicht, wie viele
  Rechenwerke arbeiten. Gemessen: drei V100 bei 100 % Util und **42 W von
  250 W**, bei vollem SM-Takt. Die Karten waren wach und hatten nichts zu
  tun. Wer Auslastung wissen will, nimmt die Leistungsaufnahme oder
  rechnet die Paare pro Sekunde gegen die FLOP-Zahl.
- **Im Browser nie im Hauptthread messen.** `advance()` rechnet über
  Objekt-Properties (`bodies[i].x`), der Worker über typed arrays.
  Gemessen an 900 selbstgravitierenden Körpern: 1.754 ms gegen ~19 ms pro
  Bild — **Faktor 70**, ohne dass sich an der Physik etwas ändert. Eine
  Messung im CPU-Modus sagt nichts über den Betrieb aus.
- **Headless-Chrome hat keine GPU.** WebGL fällt auf Software zurück und
  verschmiert 150k halbtransparente Punkte zu Flächen, die im echten
  Browser nicht existieren. Am 21.07. wurde daraus fälschlich ein
  „Rendering-Fehler bei den Galaxien" abgeleitet. Bei Optik-Befunden aus
  headless: erst im echten Browser gegenprüfen.
- **Der Browser-Cache verschluckt CSS-Änderungen.** `/solar-system/` hat
  kein `Cache-Control` (siehe 6.6). Ein Screenshot nach `navigate` zeigt
  womöglich den alten Stand; mit `?v=<zufall>` an der URL umgehen.
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

### 5.7 Berührungsradius senken, damit weniger Testteilchen verschluckt werden
Bringt fast nichts. Der Einfangquerschnitt wird von **Gravitational
Focusing** bestimmt, und das skaliert mit **√R**, nicht mit R: Bei den
Galaxien (10⁸ M☉) ist die Fluchtgeschwindigkeit am Rand dreimal so hoch
wie die Relativgeschwindigkeit, der effektive Radius also 1.561 AE statt
der eingestellten 500.

| R_GAL | effektiv | Halbwertszeit der Wolke |
|---|---|---|
| 500 AE | 1.561 AE | 61 Jahre |
| 100 AE | 669 AE | 142 Jahre |
| 10 AE | 209 AE | 453 Jahre |
| 1 AE | 66 AE | 1.435 Jahre |

Selbst punktförmige Massen fressen die Wolke noch auf, nur langsamer —
und die Galaxienverschmelzung wäre dabei tot. Wer den Schwund wirklich
loswerden will, muss Tracer×Masse-Kollisionen abschalten, nicht am
Radius drehen.

### 5.8 „Über M_MAX hinaus überschreibt der Kernel Shared Memory"
Widerlegt. `load_state` prüft und wirft (`nbody_kernel.py`), nachgemessen
mit 64 → OK und 65 → `ValueError`. Der Kernel ist an dieser Stelle sicher;
gefährlich war nur, **wo** der Fehler landete: im Film-Pfad stirbt der
Producer damit im Kindprozess, und das blieb bis dahin unbemerkt (siehe
`test_producer_tod.py` in Abschnitt 2).

### 5.9 Filamente aus masselosen Tracern
Prinzipiell unmöglich, unabhängig von Anfangsbedingungen, Expansion oder
Teilchenzahl. Testteilchen ziehen sich nicht gegenseitig an; das kosmische
Netz entsteht aber genau durch Selbstgravitation — eine Überdichte muss
sich *selbst* zusammenziehen. Ohne das bleibt das Feld so glatt, wie es
gestartet ist.

Am 21.07. wurde erst die fehlende Expansion als Ursache vermutet und ein
Hubble-Fluss eingebaut (Szenario „Galaxienhaufen (Expansion)", marginal
gebunden bei Ω ≈ 1). Das ist physikalisch richtig und ändert am Ergebnis
trotzdem nichts, weil die zweite, wichtigere Zutat weiter fehlt. Weg
dorthin: `TODO.md`.

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

### 6.9 Shared-Memory-Ringe überleben einen harten Serverstod
Der Elternwächter (`39ffb50`) gibt die **GPU** frei, aber das `unlink()`
der Ringpuffer steckt in `FilmSession.stop()` — stirbt der Server per
SIGKILL, bleiben sie in `/dev/shm` liegen. Gefunden am 21.07.: drei
Leichen vom 19.07., zusammen 287 MB RAM.

Meist selbstheilend, weil Pythons `resource_tracker` beim Herunterfahren
aufräumt (daher die „leaked shared_memory objects"-Warnungen in den
Testläufen). Nur wenn auch der stirbt, bleiben sie. Prüfen mit
`ls -la /dev/shm/`, löschen wenn `lsof` null Handles zeigt. Fix wäre, den
Producer beim Verwaisen zusätzlich freigeben zu lassen.

### 6.10 Schwarze-Loch-Schwelle bei kosmologischen Massen — erledigt
`BH_MASS_THRESHOLD` steht auf **230 M☉** — sinnvoll für Sternreste, aber
die Körper der kosmologischen Szenarien liegen vier bis fünf Größen-
ordnungen darüber. **Jede** Verschmelzung wurde dort zwangsläufig zum
Schwarzen Loch.

Gelöst über ein Szenario-Flag `kosmologisch` (wie `precise` und
`nurBrowser`), nicht über eine szenarioabhängige Schwelle: Es hängt nicht
an der Masse, sondern daran, was ein Körper **darstellt** — eine Galaxie
bzw. ein Massen-Sample des Dichtefelds ist kein kompaktes Objekt, egal
wie schwer. Dort verschmelzen sie unter Volumenerhaltung.

**Nebenbefund, weiter offen:** Der Film-Pfad promoviert überhaupt nie.
Ein Merge-Ereignis setzt im Client nur `mass` und `radius` — `realR`,
`isBlackHole` und Farbe bleiben, wie sie waren. Live und Film laufen hier
also auseinander, unabhängig vom Szenario.

### 6.13 „kein Film aktiv" beim Filmstart
Beim Start einer Film-Session schickt der Client ein `MSG_FILM_SUB`,
bevor der Server die Session angelegt hat — der antwortet dann mit
„kein Film aktiv". Sichtbar in der Browser-Konsole, mehrfach je Start.

Folgenlos, aber nicht harmlos: Der Fehlerpfad des Clients schaltet bei
Backend-Fehlern auf den WebWorker zurück. Bei den selbstgravitierenden
Szenarien riss das die ganze Sitzung mit — die Szene landete im Worker
bei 1,5 FPS, obwohl er sie gar nicht rechnen kann. Dort wird der Fehler
deshalb jetzt ignoriert (`clientRechnetSelbst()`), was die Wurzel nicht
beseitigt.

Sauber wäre, den ersten `filmSub` erst nach der Bestätigung des Servers
zu senden — oder `MSG_FILM_SUB` ohne aktive Session still zu verwerfen,
statt einen Fehler zu melden, der wie ein Defekt aussieht.

### 6.12 Lastverteilung — Erkennung geregelt, Physik offen
**Erledigt: die Erkennung.** `_streifengrenzen` schnitt die Szene in
Streifen mit gleich vielen Asteroiden. Gleiche Körperzahl ist aber nicht
gleiche Arbeit — die Paarzahl wächst mit der Dichte **quadratisch**.
Gemessen an 120k Asteroiden, halb im Knödel: 81,8 Mio Kandidaten auf der
einen Karte, 42,2 Mio auf der anderen; die schnellere wartete 44 von
96 ms. Bei gleichmäßigen Szenen und beim Gürtel stimmte es dagegen auf
1 % genau.

`_lastanteile` führt die Anteile jetzt aus der gemessenen Kandidatenzahl
nach. Gemessen im geschlossenen Regelkreis: **1,27×** beim Knödel,
unverändert bei den anderen (0,98× / 1,02× = Rauschen).

**Nicht die Zeit als Regelgröße nehmen** — das war der erste Versuch und
er lief weg: Ein schmaler Streifen behält seinen vollen **Halo**, wird
also nie proportional schneller, der Regler schiebt weiter und landet
bei 0,93/0,07 auf einer nachweislich ausgeglichenen Szene. Der Gürtel
wurde damit *langsamer* als ohne Regelung. `test_lastregler.py` Fall C
hält das fest.

Der Preis der Kandidatenzahl: Sie gleicht **unterschiedlich schnelle
Karten nicht aus** (zwei RTX 8000 liegen in f32 21 % auseinander). Das
ist der kleinere Effekt gegen Faktor 1,9 und bräuchte ein Zeitmaß, das
den Halo herausrechnet.

**Offen: die Physik.** `load_state` in `nbody_kernel.py` gewichtet die
Asteroiden-Shards mit `f64_score`, und der gibt allen drei V100
identische Werte, obwohl CUDA 2 gemessen 11 % schneller ist (Abschnitt 1).
Rechnerischer Gewinn einer gemessenen Gewichtung: rund **4 %** des
Kernel-Anteils — und der Kernel ist nicht der Engpass (Abschnitt 4), also
unterm Strich unter 1 %. Auf gemischter Hardware wäre es viel.
`miss_gewicht` in `selfgrav_kernel.py` wäre dafür brauchbar.

Ebenfalls offen: `pick_detect_devices` sortiert nach `f64_score`, obwohl
die Erkennung **f32** rechnet. Auf dieser Maschine folgenlos (die Physik
belegt die V100, es bleiben ohnehin nur die RTX), auf anderer nicht.

### 6.11 Bahnspuren schwarzer Löcher — erledigt
Die Spur lief in `b.color`, und die ist beim Schwarzen Loch Fast-Schwarz
auf schwarzem Grund. Jetzt zeichnet sie in `BH_TRAIL_COLOR` (`#ff8c28`,
das Orange des Akkretions-Halos); der Körper selbst bleibt `BH_COLOR`.
Beide Farben sind Konstanten statt dreifach eingestreuter Literale.
