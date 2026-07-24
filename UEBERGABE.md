# Übergabeprotokoll

Was hier steht, steht **nicht** im Git-Log: Betriebswissen, Messfallen,
gescheiterte Ansätze und offene Punkte. Die Umsetzungshistorie ist
bewusst nicht enthalten — dafür ist `git log` da.

Stand: 2026-07-23, Particle-Mesh im Film-Streaming (Branch
`feat/particle-mesh`).

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

**Die LOD-Auswahl ist eine Stichprobe, keine Dichteregelung.** Das
dichteabhängige Ausdünnen (`_dichte_filter`, Bisektion über ein
512×512-`bincount`) ist raus: Weil es die Auswahl an der jeweils
aktuellen Dichte festmachte, wechselten von Sample zu Sample ANDERE
Objekte in die Auswahl — sichtbar als Flackern. Jetzt entscheidet eine
globale Hash-Stichprobe `budget/N` über `_index_hash`, also für dieselbe
Objekt-ID immer gleich. Determinismus schlägt hier gleichmässige Dichte;
die verworfenen Glättungsversuche stehen in 5.4 und 5.5.

**Ein `with cp.cuda.Device(d)` belegt beim VERLASSEN eine zweite Karte.**
CuPy merkt sich beim Betreten das vorherige Device und stellt es beim
Verlassen per `cudaSetDevice` wieder her — und `cudaSetDevice` legt den
Primary Context sofort an, auch wenn dort nie gerechnet wird. Gemessen:
**308 MiB auf CUDA 0**, sobald der PM-Kernel auf CUDA 2 seinen ersten
`load_state` beendet hatte, und das je Producer, also je Browser-Tab.
Der Kernel-Code war dabei durchweg korrekt; das Leck ist die
Rückstellung. Gegenmittel ist `setze_default_karte` in
`film_producer.py`: einmal `cp.cuda.Device(dev).use()` je Thread, bevor
der erste Device-Block verlassen wird — auch als `initializer` der
ThreadPools, denn das aktuelle Device ist **thread-lokal**.

Wer das nachmisst: `nvidia-smi --query-compute-apps=pid,gpu_bus_id,
used_memory` nach der eigenen PID filtern, nicht die Gesamtbelegung
ansehen. Und die Marke muss **hinter** dem `with`-Block stehen — innerhalb
sieht alles sauber aus, genau daran ist die erste Suche gescheitert.

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

**Ein Renderpfad, nicht zwei.** Punkte zeichnet ausschliesslich WebGL —
Canvas2D macht nur noch Text, Ringe und Verlaeufe. Vorher sprangen beide
ein: GL uebersprang die massiven Koerper, also zeichnete Canvas2D sie
nach, und bei 44k Koerpern kostete das zweistellige Millisekunden je
Bild. Wer einen Koerpertyp neu einfuehrt, muss ihn im GL-Batch anmelden
(`_punktwolke` / `_glPunktmodus`) — er wird sonst schlicht nicht
gezeichnet, und es gibt keinen zweiten Pfad mehr, der das auffaengt.

**Die Netze sind massstabsagnostisch.** Gitterringe und Koordinatenkreuz
ziehen ihre Stufen aus einer gemeinsamen 1-2-5-Leiter (`NETZ_LEITER`) von
1e-4 bis 1e6 AE; welche Stufe man sieht, entscheidet allein der
Pixelabstand beim Zeichnen. Vorher standen dort feste Listen (Ringe
0,1-32 AE, Achsen bis 5000 AE) — im kosmischen Netz mit 60.000 AE
Ausdehnung war beides unsichtbar. Ab 1e4 AE beschriftet `netzLabel` in
Lichtjahren, weil sechsstellige AE-Zahlen niemand liest.

**Gesperrt heisst ohne Tooltip.** `bedienelementSperren` ist die einzige
Stelle, die Bedienelemente sperrt: `disabled`, Ausgrauen und das
Abnehmen des Tooltips gehoeren zusammen (er erklaert, was ein Regler
stellt — gesperrt stellt er nichts). Beim Freigeben kommt er aus
`data-i18n-title` zurueck. Die Logik lag vorher an vier Stellen getrennt.
Wichtig dort: Sitzt das Element IM `<label>`, darf nur das Label gedimmt
werden — Opacity multipliziert sich sonst (0,4 x 0,4 = 0,16).

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
../venv/bin/python test_pm.py               # Particle-Mesh gegen all-pairs
../venv/bin/python test_pm_pipeline.py      # PM im Producer-Pfad
../venv/bin/python test_lod_auswahl.py      # Stichprobe deterministisch
../venv/bin/python test_tracer_split.py     # Tracer server-seitig, anonym
```

**Ein Producer-Prozess braucht `if __name__ == "__main__":`.** Die Sitzung
startet ihr Kind per `spawn`, und das importiert das Hauptmodul erneut —
ein Skript, das die `FilmSession` auf Modulebene baut, stirbt mit
„attempt to start a new process before the current process has finished
its bootstrapping phase". Betrifft jedes eigene Mess-Skript.

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

**Diagnose am laufenden Betrieb** (Film-Pfad):

- **Zeitanteile:** `FilmSession.DIAG` bzw. `--diag` in der Unit →
  `[film-diag]` (prod-/play-Rate in Sim-Tagen/s, Vorlauf, Ringfüllung,
  `drossel=`) und `[stream]` (`block=puffer|head|-`, Vorrat gegen Ziel,
  `sps`, `step`, Kosten je Sample). Beide Zeilen zusammen sagen, ob
  Producer, Server oder Client bremst. Als Drop-in einschaltbar, ohne die
  Unit zu ändern:
  `~/.config/systemd/user/solar-cuda.service.d/diag.conf` mit leerem
  `ExecStart=` plus der Zeile inklusive `--diag`.
- **Stack eines hängenden Producers ohne root** (`ptrace_scope=1` blockt
  gdb, py-spy fehlt): `faulthandler.register(signal.SIGUSR1,
  all_threads=True)` in `producer_main` einbauen, dann `kill -USR1 <pid>`
  → Stack ins Journal. Feuert NICHT bei blockierendem C-Call — dann
  `/proc/<pid>/wchan` lesen.
- **Live-Reproduktion im echten Browser:** `DISPLAY=:0
  XAUTHORITY=~/.Xauthority chrome --remote-debugging-port=9222
  http://127.0.0.1/solar-system/?v=dbg`, dann per chrome-devtools den
  log-Slider `film-speed` setzen (Wert 3 = 1000 Tage/s) und
  `filmPlayheadDays` mitschreiben.

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

**Kernel A/B (selbstgravitierend), 22.07.:** 44.212 Koerper vollstaendig
gestreamt bei 18,2 FPS und 36,1 Sim-Tagen/s, nach 7,76 Sim-Jahren klar
ausgepraegte Knoten. Die GPU faellt dabei regelmaessig auf 0 % — der
Producer drosselt bei 70 % Ringfuellung, das ist kein Defekt.

**Fernzugriff (Klinik ueber nginx), 22.07.:** 3,45 MB/s Ende zu Ende bei
66 ms RTT. Das traegt 9,7 Samples/s von 20 gewuenschten — siehe 6.14. Bei
Messungen aus der Ferne ist das die erste Groesse, die man pruefen muss,
bevor man Physik oder Rendering verdaechtigt.

**Particle-Mesh (Kernel B):** 1 Mio Massen + 500k Tracer, Gitter 1024²,
K=8 Substeps je Batch, eine V100 — gemessen über `step_batch`, also genau
den Anteil, der im Producer `step=93 %` belegt:

| Stand | ms/Batch | Sim-Tage/s |
|---|---|---|
| 23.07., Ausgangslage | 133 | 604 |
| Gitter je Batch statt je Substep | 128 | 626 |
| CIC fusioniert + Faltung in f32 | 89 | 903 |
| pinned Download-Puffer | 63 | 1267 |
| nur Positionen transferieren (kein vx\|vy) | **44** | **1820** |

Insgesamt **3,0×** (604 → 1820). Der letzte Schritt (kein vx\|vy) betrifft
nur `pm_kernel.py` und den Producer — Kernel A (`selfgrav_kernel.py`,
all-pairs) bleibt bit-identisch. Siehe 6.17.

**Wo die Zeit im Substep steckt** (1M Massen, 1024², vor dem Umbau):

| | ms |
|---|---|
| `gather` Massen (1M) | 2,37 |
| `_cic_deposit` | 1,72 |
| `gather` Tracer (500k) | 1,30 |
| alle drei FFTs zusammen | 1,51 |
| `grid_fuer` (4 synchrone min/max) | 0,13 |

**Die FFT war nie der Engpass** — CIC-Deposit und -Gather waren es, und
zwar nicht als Rechenlast, sondern als Speicherverkehr: viele kleine
cupy-Kernel mit temporären Arrays zu je 8 MB, und für `ax`/`ay` wurden
Zellindex und Gewichte zweimal berechnet. Jetzt je ein ElementwiseKernel
(Deposit mit `atomicAdd`, Gather für beide Komponenten in einem Zug), und
die Faltung läuft in f32 (`kraft_f32`; der Zustand bleibt f64).

**Der Engpass war danach der D2H-Transfer** (47 von 63 ms, Batch 152 MB) —
und der trug zu 40 % Geschwindigkeiten, die niemand liest. Weggelassen:
Batch 152 → 92 MB, 44 ms/Batch, 1820 d/s (Details in 6.17).

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

### 5.10 Zeldovich-Anfangsbedingungen fuer mehr Filamente
Nachgemessen am 22.07. auf einer V100, N=20.000, 3.000 Schritte
(eta ~ 0,13), vier Seeds je Variante, ausgewertet bei 40 Sim-Jahren:

| Anfangsbedingung | F/K |
|---|---|
| neun Keime (das bestehende Szenario) | 2,39 ± 0,53 |
| Zeldovich, Moden n=-2, A=0,12 | 2,72 ± 0,45 |
| Zeldovich, Moden n=-2, A=0,25 | 2,03 ± 0,34 |

**Die Streuung zwischen Seeds ist groesser als der Unterschied zwischen
den Verfahren.** Ein Umbau des Szenarios auf ein gausssches Zufallsfeld
mit Potenzspektrum lohnt nicht — das bestehende liefert dieselbe
Netzstruktur. Der Haupthebel ist die LAUFZEIT: F/K steigt von ~1,0 bei
15 Jahren auf 2,8-3,3 bei 60 Jahren (t_dyn liegt bei ~31 Jahren).

**Messfalle, die dabei fast zu einem falschen Szenario gefuehrt haette:**
Der erste Anlauf mass den FLAECHENanteil filamentartiger Zellen, mit
einer Schwelle relativ zur Streuung des Feldes. Die Nullprobe entlarvte
ihn:

| Kontrollfeld | "Filament" laut erster Metrik |
|---|---|
| Poisson, voellig strukturlos | **37,6 %** |
| ein echter Balken | **6,2 %** |

Genau verkehrt herum — gemessen wurde Schrotrauschen. Und weil die
FFT-Variante aus einem zufaellig ausgeduennten Gitter startet, also
rauschiger ist, gewann sie scheinbar deutlich (18 % gegen 7 %). Erst die
zweite Metrik taugt: MASSENanteil statt Flaechenanteil, Glaettung ueber
mehr als den Teilchenabstand, nur ueberdichte Gebiete (delta > 1), und
die Klassifikation ueber das VERHAELTNIS der Hesse-Eigenwerte
(vs = 0,25). Geeicht liefert sie Poisson 0,0 %, einen Balken F/K = 25,5,
Knoten mit Bruecken 1,1 und reine Knoten 0,7.

**Wer hier weitermisst: zuerst die Nullprobe, dann die Physik.** Eine
Strukturmetrik, die an einem Zufallsfeld nicht null ergibt, misst sich
selbst. Und einzelne Seeds sagen nichts — die Streuung gehoert
mitgemessen. Skripte im Scratchpad der Sitzung (`metrik2.py`,
`streuung.py`).

---

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

**Kernel B widerspricht dem nicht.** Dort laufen Tracer ZUSAETZLICH zu
selbstgravitierenden Massen: Die Struktur entsteht aus den Massen, die
Tracer zeichnen sie nur nach und kosten dabei N*T statt N^2. Was hier
verworfen ist, bleibt verworfen — Tracer ALLEIN bilden nichts.

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

### 6.3 Datenlokalität GPU-zu-GPU — erledigt, mit klarem Ergebnis
**GPU-zu-GPU direkt ist auf dieser Maschine unbrauchbar.** Nachgemessen
am 24.07. mit einem 8-MB-Kraftfeld zwischen zwei Karten:

| Weg | Zeit | Durchsatz |
|---|---|---|
| cross-device Kopie | 52,0 ms | 0,15 GB/s |
| dieselbe, nach `deviceEnablePeerAccess` | 52,0 ms | 0,15 GB/s |
| **über pinned Host (D2H + H2D)** | **5,0 ms** | **1,56 GB/s** |

`deviceCanAccessPeer` meldet `True`, und Peer-Access lässt sich auch
aktivieren — es ändert nur nichts. Über M.2-zu-Oculink- bzw.
USB4-Adapter (Abschnitt 1) ist der Umweg über den Host **zehnmal
schneller** als der direkte Weg. Wer hier Datenlokalität plant, plant
über den Host.

**Pinned Host-Puffer dagegen trägt sehr wohl** — die frühere Notiz
(„beim Download bringt pinned nichts, 3,21 gegen 3,24 GB/s") gilt für den
Download nicht: an 152 MB je Batch gemessen 70,4 ms pageable gegen
47,3 ms pinned (2,12 gegen 3,15 GB/s). Der Unterschied ist der
Staging-Puffer, durch den der Treiber pageable Speicher schleusen muss.
Eingebaut in `NBodyPM._host_puffer`; der Puffer wird wiederverwendet und
gilt darum nur bis zum nächsten `step_batch`.

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

### 6.15 OFFEN: alle Punkte kollabieren auf die x-Achse
Symptom: Nach laengerem Zoomen und Schwenken liegen samtliche Punkte auf
einer waagerechten Linie bei y = 0, quer durchs Bild. Gitter, Schwerpunkt
und Beschriftungen bleiben korrekt, die Zeitleiste laeuft weiter.

Beobachtet am 22.07. bei 44.212 Koerpern, Full-HD-Fenster, nach mehreren
schnellen Kamerabewegungen.

**Was ausgeschlossen ist:**
* *Kamera.* Reset und starker Zoom aendern nichts.
* *Fenstergroesse.* `resize` fuehrt W/H korrekt nach, das Fenster war
  regulaer gross.
* *Degenerierte Referenz-Box.* Der Server klemmt `spanx/spany` auf
  mindestens 1e-6, und der Client meldet das Sichtfenster im 1-Hz-Takt
  neu — ein flacher Ausschnitt wuerde sich binnen einer Sekunde
  korrigieren.
* *Protokoll v6.* `test_film_protokoll.py` Fall i) faehrt genau diesen
  Ablauf durch (Auswahl wechselt mitten im Strom, weil die Kamera
  springt): Das Flag "Liste wie vorher" steht ausschliesslich dann,
  wenn die Auswahl wirklich gleich ist, und der Frame geht in jedem
  Fall exakt auf.

**Was hilft:** ein Tab-Reload. Ein Reset der Kamera nicht.

Damit deutet es auf einen CLIENT-Zustand, der sich ueber die Laufzeit
aufbaut und den nur ein Neuladen raeumt — nicht auf die Daten. Wer das
weiterverfolgt: Zuerst pruefen, ob die y-Werte schon in `filmApply`
falsch ankommen oder erst im GL-Batch verlorengehen; das trennt
Dekodierung von Darstellung. Ein Blick in den Ring unter `/dev/shm`
klaert zusaetzlich, ob der Producer ueberhaupt korrekte y liefert.

### 6.14 Ruckeln aus der Ferne ist eine BANDBREITEN-, keine Rechenfrage
Gemeldet als „Puffer voll, GPU idle, Animation trotzdem ruckelig" — also
weder Physik noch Ring. Nachgemessen (22.07., Zugriff aus der Klinik über
nginx):

| Strecke | Durchsatz |
|---|---|
| Server → nginx (Loopback) | 3,42 MB/s |
| nginx → Gegenstelle | 3,45 MB/s |

Gleich groß, also **staut nginx nicht** — die TCP-Backpressure schlägt
bis zum Server durch. Die Außenverbindung meldete `cwnd:213` bei
`rtt:66,2 ms`, das sind rund 4,5 MB/s Fenstergrenze; `retrans` von 68 bis
142 deckelt sie zusätzlich.

Dagegen der Bedarf: **8 Byte je Punkt und Sample** (4 B Index + 2×2 B
quantisierte Koordinaten, `build_frame`). Bei 44.212 gestreamten Körpern
sind das 354 KB je Sample, bei den gewünschten 20 Samples/s also
**7,1 MB/s — doppelt so viel, wie die Leitung trägt.** Es kommen 9,7
Samples/s an, und die mit dem Jitter der Leitung. Genau das sieht man.

**Der Hebel ist der Index: 4 der 8 Byte.** Nachgemessen an einer
geclusterten Punktwolke mit 44.212 Punkten (Skript im Scratchpad):

| Verfahren | Größe | bei 3,45 MB/s | Tempo |
|---|---|---|---|
| roh (heute) | 345,4 KB | 9,8 Samples/s | — |
| zlib-1 | 233,0 KB (67 %) | 14,5 Samples/s | 73 MB/s |
| zlib-6 | 232,9 KB | 14,5 Samples/s | 24 MB/s |
| Index weglassen | 172,7 KB (50 %) | 19,5 Samples/s | — |
| Delta-Index u16 + zlib-1 | 188,5 KB | 17,9 Samples/s | 87 MB/s |

**Kompression ist keine Alternative zum Indexsparen, sondern greift auf
dieselbe Ursache.** Die Koordinaten sind praktisch inkompressibel — 172,7
auf 172,4 KB, ganze 0,2 %. Der gesamte zlib-Gewinn stammt aus den
sortierten u32-Indizes. Nimmt man die heraus, bringt zlib danach nichts
mehr.

**Umgesetzt als Protokoll v6** (`FILM_PROTO_VERSION = 6`): Bit 31 der
Sample-Länge heißt „Indexliste wie im vorigen Sample", die Liste fehlt
dann. Nachher gemessen, gleiche Strecke: 3,26 MB/s bei nun 177 KB je
Sample, also **≈18,4 statt 9,7 Samples/s**. Die Leitung bleibt der
Deckel — sichtbar an 609 KB Send-Q und einer von 66 auf 94–108 ms
gestiegenen RTT (Bufferbloat). Unkritisch, weil der Client 5 s
vorpuffert.

Der Bezug läuft über FRAME-Grenzen, nicht frameweise wie der Sub-Block:
Remote passt oft nur EIN Sample in einen Frame (Budget 512 KB gegen 354
KB Kosten), frameweise verankert hätte nie gegriffen.

**Die Versionsnummer steht zwangsläufig doppelt** — als Konstante in
`server.py` und als Literal im `FILM_START` des Clients; über die
Sprachgrenze gibt es keine SSOT. Beim Umstieg auf v6 wurde nur der
Server gezogen, woraufhin der Server JEDEN Filmstart ablehnte und der
Browser „Film-Protokollversion veraltet" meldete — durch keinen Reload
zu beheben, weil es kein Cache-Problem war. `test_film_protokoll.py`
Fall h) vergleicht die beiden Stellen jetzt gegeneinander.

Damit ist auch die alte Begründung für `compression=None` einzuordnen:
zlib war der Durchsatz-Deckel, **solange die Leitung schnell war**. Bei
3,45 MB/s schafft zlib-1 mit 73 MB/s das 21-fache — remote bremst es
nichts mehr. Wer es wieder einschaltet, muss aber bedenken, dass
permessage-deflate beim Handshake ausgehandelt wird und den LOKALEN Fall
wieder ausbremsen würde; unterscheidbar wären die Fälle nur am
`X-Forwarded-For` des Proxys.

**Messfalle dabei:** Eine erste Messung zeigte den Stream in BEIDE
Richtungen stillstehend — kein Byte über sechs Sekunden. Das war kein
Fehler, sondern ein **Browser-Tab im Hintergrund**: Der Client meldet
seinen Playhead aus dem rAF-Loop (1 Hz), und den friert der Browser in
verdeckten Tabs ein. Ohne Playhead sendet der Server nicht mehr
(`sent_t - ph > target`), und der Producer drosselt daraufhin auf 70 %
Ringfüllung — die GPU fällt auf 0 %. Alles korrekt, aber von außen
ununterscheidbar von einem Defekt. Wer den Stream vermisst, muss das
Fenster sichtbar halten.

### 6.13 „kein Film aktiv" — erledigt, Ursache war eine DOPPELTE Verbindung
Im Betrieb lief in der Browser-Konsole endlos:

    CUDA-Backend-Fehler: kein Film aktiv

Die Wiedergabe stand, die GPU war untätig, und ein Reload half nur
kurz. Zwei Fehldiagnosen führten daran vorbei — ein Wettrennen beim
Start (plausibel, aber falsch) und Netzwerk/Client-Leistung.

**Die Ursache:** Der Server hält `film` LOKAL in `handle(ws)`, also eine
Session **pro Verbindung**. Beim Laden entstand ein Wettlauf: Die
Auto-Erkennung baut eine Verbindung auf, und während die noch läuft,
ruft die Szenario-Logik `applyEngine('cuda')` — dort ist `cudaSocket`
noch null, also wurde ein zweites Mal verbunden. `FILM_START` ging über
die eine Verbindung, `FILM_SUB` über die andere, und die kannte keine
Session.

Erkennbar war es an einem **doppelten „CUDA-Backend verbunden"** in der
Konsole. Genau darauf zu achten wäre der kürzeste Weg gewesen.

Behoben in `initCudaBackend`: Es baut nur noch eine Verbindung zur Zeit
auf (`_cudaVerbindungLaeuft`, freigegeben bei `hallo` oder `zu`).

**Als Netz bleibt** ein Selbstheilungspfad: Meldet der Server „kein Film
aktiv", startet der Client die Sitzung genau einmal neu; scheitert auch
das, erscheint eine klickbare Meldung, die es erneut versucht. Innerhalb
einer Verbindung ist ein Desync nicht mehr konstruierbar — `film` wird
nur bei `FILM_STOP` geleert, und das schickt der Client selbst.

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

### 6.16 IDEE: Kernel C — viele Massen, die verschmelzen (Kugelsternhaufen)
Es gibt ein physikalisches Regime, das **keiner** der beiden Kerne
bedient:

- **Alter Kernel** (`nbody_kernel.py`): hat Kollisionen/Merges, aber
  höchstens `M_MAX = 64` Massen. Grund ist NICHT der Shared Memory (M_MAX
  belegt nur ~2,6 KB von 64–96 KB, Platz für ~1.500), sondern dass er
  Masse×Masse **seriell auf Thread 0** rechnet (O(M²)).
- **Selfgrav-Kernel** (`selfgrav_kernel.py`, Kernel A): viele Massen
  (50.000+, gekachelt/parallel), aber **keine** Kollisionen — das
  Plummer-Softening hält die Teilchen auf Abstand, Merges sind für
  Massen-Samples eines Dichtefelds der falsche Begriff.

Gewünschtes Regime dazwischen: **zehntausende Sterne, die einander
anziehen UND verschmelzen** — ein Kugelsternhaufen, in dem echte
Kollisionen passieren.

**Wichtig, damit niemand am falschen Ende anfängt:** Es ist NICHT damit
getan, die serielle Masse×Masse des alten Kernels zu parallelisieren. Bei
M ≤ 64 ist die serielle Schleife 3.136 Paare = Mikrosekunden, also 0,04 %
der Arbeit — die Tracer dominieren und laufen längst parallel. Das
Parallelisieren allein bringt **null** Speedup und würde nur den
Selfgrav-Kernel duplizieren.

Der echte Aufwand ist die **Kollisionserkennung parallel über zehntausende
Massen** — nicht seriell, nicht O(M²), sondern über ein **Raumgitter**
(räumliches Hashing: nur Nachbarzellen prüfen). Das ist ein eigener
Kernel C:
- Kräfte: Tiling wie Kernel A (parallel, gekacheltes VRAM).
- Kollisionen: Raumgitter-Broadphase + Merge-Anwendung.
- Softening dann klein oder aus (sonst kollidiert nichts).
- f64-Zustand, f32-Kraft (wie in Kernel A gemessen, UEBERGABE-Kennzahlen).

Gehört konzeptionell neben die Kernel-Roadmap in `TODO.md` (Tiling A,
Kernel B, Barnes-Hut/PM). Reihenfolge offen — erst wenn ein Szenario das
wirklich braucht.

### 6.17 PM-Durchsatz: 3,0× geholt, der Rest wäre CUDA-Graphs
Stand: 604 → 1820 d/s (Abschnitt 4). Der letzte Schritt hat den D2H-Engpass
angegangen — **`vx|vy` wurden übertragen und weggeworfen.**

Der PM-Kernel liefert jetzt nur noch Positionen `[x|y | tx|ty]` statt
`[x|y|vx|vy | tx|ty]`; die Geschwindigkeiten bleiben f64 auf der GPU (die
Leapfrog-Integration nutzt sie weiter), gehen aber nicht mehr über den
×4-Link. Der Ring trägt ohnehin nur Positionen (`schreibe_slot` liest
`out_i[0:2*n]`), und die Engine-Übergabe holt den v-Zustand separat via
`export_f64`. Batch 152 → 92 MB, 63 → 44 ms, kein Genauigkeitsverlust.

**Kernel A blieb bit-identisch** — auf Wunsch nur `pm_kernel.py` angefasst.
Der Producer liest das Layout am Klassenattribut `NBodyPM.nur_positionen`
(`getattr(sim, "nur_positionen", False)`), Kernel A liefert weiter
`[x|y|vx|vy|tx|ty]` und der Producer verwirft dessen v beim `voll`-Umbau.
Der Tracer-Block liegt darum je Kernel an anderer Stelle (2·nm bzw. 4·nm).

**Was bleibt: CUDA-Graphs.** Die reine Rechnung liegt bei ~16 ms je Batch,
der Rest ist Transfer und Launch-Overhead. Die Launch-Zahl je Substep ist
durch die Kernel-Fusion schon von ~zwanzig auf eine Handvoll gefallen;
einen Substep als Graph aufzeichnen und abspielen spart den verbliebenen
Python-/Launch-Overhead. Erst messen, ob es den Aufwand trägt — 1820 d/s
sind bei 1000 d/s Wunschtempo bereits über der Wiedergabe-Grenze.

### 6.21 Tracer auf eine zweite Karte — VERWORFEN, gemessen
Erst als lohnend eingeschätzt (Break-Even `T = grid_n²`, über das
Ausgabe-Volumen gerechnet), dann **vor dem Bau gemessen — und verworfen.**
Der Denkfehler der ersten Rechnung: Sie stellte den Feldtransfer gegen die
Tracer-*Ausgabe*. Der eigentliche Nutzen des Splits wäre aber, den
Tracer-*Gather* aus dem kritischen Pfad zu nehmen — und der ist seit der
CIC-Fusion (6.17, Punkt 2) fast nichts mehr.

Gemessen (Massen-Karte V100, Tracer-Karte RTX, pinned Host):

| | grid 1024 | grid 2048 |
|---|---|---|
| Feldtransfer je Substep | 5,0 ms | 20,0 ms |
| Tracer-Gather 1 Mio | 0,15 ms | 0,55 ms |
| Tracer-Gather 5 Mio | 0,67 ms | 2,70 ms |

Der Feldtransfer kostet **das 7- bis 40-fache** des Gathers, den er
einspart. Sequenziell ist der Split klar schädlich; selbst mit Pipelining
(Streams/Events/Doppelpuffer) wäre der Gewinn nur die paar ms Gather,
während die Massen-Karte den Feldtransfer (bei grid 2048: 160 MB je Batch)
oben drauf bekäme. Kein Regime, in dem es sich trägt.

**Lehre:** Der fusionierte CIC-Gather (0,7 ms bei 5 Mio) hat dem Split die
Grundlage entzogen. Der frühere Break-Even galt für den alten,
Fancy-Index-Gather. Wer das später aufgreift, misst zuerst wieder Gather
gegen Feldtransfer, bevor er baut. Die Slider-Erweiterung auf 5 Mio
(Massen und Tracer) bleibt davon unberührt — 5 Mio laufen auf einer Karte.

### 6.18 Tracer-Kreis nach dem Handover
Beim Neustart der Sitzung (Inject/Engine-Übergabe) würfelt der Producer
die Tracer auf die t=0-Scheibe zurück (`wuerfle_tracer(seed=0)`, harte
Kante bei 72.000 AE), während die Massen ihren Zustand behalten — sichtbar
als scharfer Kreis mit Halo. Fix: den Tracer-Zustand server-intern über
den Handover mitführen statt neu zu würfeln. Der Client merkt davon
nichts, die Tracer bleiben dort anonym.

### 6.19 Massen und Tracer in einer Farbe — erledigt
Getrennt: Massen `_MASS_HEX` (`#ffd3a0`, sehr hellorange), Tracer
`_TRACER_RGB` (`#a8c8ff`, hellblau), beide benannte Konstanten (SSOT).

**Nebenwirkung, offen (siehe 6.23):** Die Farbtrennung hat einen
bestehenden Tracer-Rendering-Kompromiss sichtbar gemacht — beim Zoom
frieren die anonymen Tracer ein, die orange Massen wandern davon: ein
oranger Halo am Verteilungsrand. Vorher (beide hellblau) unsichtbar.

**Zweite offene Nebenwirkung:** Bei sehr vielen Tracern (5 Mio gegen 500k
Massen) übertönen die blauen Tracer die orange Massen im additiven
Blending — die Massen sind nur am dünnen Rand sichtbar. Wer die Massen
heraustreten lassen will: sattere Farbe und/oder grössere Punkte,
und/oder sie NACH den Tracern zeichnen (nicht additiv untergehen).

### 6.20 Wiedergabe bei hohem Tempo — gemessen, gedämpft
Gemeldet als „stockt bei 1000 Tage/s, GPUs arbeiten gar nicht".
Nachgemessen mit `--diag` (1,5 Mio Objekte, Raster 10 d):

    prod=430 d/s   play=403…921 d/s   drossel=0%   step=93%   GPU 55%
    vorrat=384…1771d (ziel 5000d)     block meist "-", zeitweise "head"

Drei Befunde, in dieser Reihenfolge zu lesen:

1. **Der Producer stand nicht.** `drossel=0 %`, `block=head` — der Server
   hatte alles gesendet, was da war. Der beobachtete Stillstand war ein
   Sessionwechsel: Der ALTE Producer stand mit `drossel=99 %` am 70-%-
   Deckel, während der neue erst hochlief. Jeder Dreh am Weichzeichnungs-
   Regler startet eine neue Sitzung (siehe 6.22).
2. **1000 Tage/s sind unerreichbar**, wenn 430 produziert werden. Die
   Wiedergabe läuft dann mit Produktionstempo — kein Fehler, sondern die
   Grenze aus 6.17.
3. **Die Unruhe kam aus dem Regelkreis.** Der Vorrat schwankt um Faktor
   4,6 (die Stream-Dichte springt: `sps` 0,5 → 1,7 → 20, `step` 200 → 58
   → 5, während sich die Bandbreitenschätzung einschwingt), und
   `follow = min(rate, Vorrat/FILM_PUFFER_S)` übersetzt das 1:1 in
   Tempo-Schwankung.

Behoben über `filmFolgeRate` (eine Stelle für beide Zweige): Der Vorrat
geht mit der Zeitkonstante `FILM_VORRAT_TAU_S` (2 s) **symmetrisch** in
die Folge-Rate ein.

**Die Interpolations-Reserve muss dabei zeitlich gedeckelt sein** — das
ist der Teil, an dem eine erste Fassung den Film zum Stehen brachte, und
der Fehler ist lehrreich genug für eigene Zeilen:

`kante` liegt bewusst ein Sample vor der Spitze (Catmull-Rom braucht ein
Voraus-Sample). Die Grösse dieser Reserve ist aber der SAMPLE-ABSTAND, und
den bestimmt der Server aus seiner Bandbreitenschätzung. Bei 1000 Tagen/s
und noch kalter Schätzung streamt er jedes 200. Raster — **2000 Sim-Tage
je Sample**. Die Reserve, gedacht als 50 ms, wurde damit zu 2 Sekunden
Playback: Der Playhead stand bei 535 Tagen HINTER seiner eigenen Kante,
Vorrat 0, Folge-Rate 0. Und weil der Playhead stand, stand auch der
Producer (Ring 70 %, `drossel=99 %`, GPU 0 %), es kam kein drittes Sample,
und nichts konnte den Zustand mehr auflösen.

**Der naheliegende Fix funktioniert nicht:** „die Reserve gilt nur,
solange sie vor dem Playhead liegt" schaltet nie um, weil die Folge-Rate
proportional zum Vorrat ist — der Playhead nähert sich der Kante
asymptotisch und erreicht sie nie. Nur ein Deckel wirkt:
`reserve = min(Sample-Abstand, FILM_RESERVE_S · rate)`.

Nachgestellt (Producer 900 d/s, Wunsch 1000 d/s, Skript im Scratchpad):

| Reserve | Sample-Abstand 2000 d | Sample-Abstand 50 d |
|---|---|---|
| ungedeckelt | **steht in 100 % der Frames** | 900 d/s, ruhig |
| Deckel 0,1 s | 900 d/s | 900 d/s, ruhig |

Dieselbe Nachstellung hat auch die Glättung entschieden: asymmetrisch
(Anstieg träge, Abfall sofort) ist bei grobem Streaming **doppelt so
unruhig** wie symmetrisch (Schwankung 112 gegen 57 Tage/s) und bei feinem
nicht besser. Gegen das Anlaufen an der Kante schützt die harte Klemme auf
`kante`, nicht die Glättung.

**Wer an dieser Regelung etwas ändert, stellt es vorher nach.** Producer,
Server und Client bilden einen geschlossenen Kreis, in dem jeder auf die
Bewegung der anderen reagiert; im Browser sieht man nur das Ergebnis, und
zwei der drei Fehler dieses Kapitels waren im Code nicht zu sehen.

**Der Server durfte weiter springen, als Daten da waren.** Der eigentliche
Auslöser des Stillstands, im echten Browser gefunden (nicht in der
Simulation): Der Stream berechnet `step` aus der Bandbreitenschätzung — bei
1,5 Mio Objekten remote 200 Raster. Steht der Producer am 70-%-Ring-Deckel,
bleiben aber weniger Slots übrig als ein Schritt breit ist (gemessen
`avail=162` gegen `step=200`), und die alte Bedingung `if avail < step:
continue` sendete dann NIE. Der Stream wartete auf einen Kopf, der nicht
mehr wuchs, weil der Producer auf einen Playhead wartete, der nicht mehr
lief, weil der Client keine Samples bekam — ein dreifacher Stillstand ohne
Log. Fix in `server.py`: `schritt = min(step, avail)`, gesendet wird,
sobald ein einziges Sample bereitliegt.

**Und der Client hielt bei einer Szenenänderung an, ohne fortzufahren.**
Der sichtbarste Teil von „startet kurz, stoppt dann wieder": Ändert man
Massen- oder Tracer-ZAHL mitten im Betrieb, dumpt die alte Sitzung ihren
f64-Zustand, dessen Körperzahl nicht mehr zur neu gebauten Szene passt.
`applyDump` rief dann `setPaused(true)` und zeigte eine Störung — obwohl
unmittelbar darauf `filmStart()` eine neue Sitzung startete. Die Pause
blieb stehen, der Playhead fror ein, der Producer drosselte. Fix: Steht ein
Neustart an (`_filmRestartPending` oder `_filmStartWartet`), ist der alte
Dump gegenstandslos und wird kommentarlos verworfen; die harte Pause bleibt
dem echten Mismatch vorbehalten (Körperzahl ändert sich OHNE Neustart —
dann wäre stilles Weiterrechnen auf geschätzten Impulsen falsch).

Verifiziert im eigenen Debug-Chrome (1M Massen + 500k Tracer, volles LOD,
1000 Tage/s): Wiedergabe läuft mit 525–567 d/s (±4 %, vorher ±40 %),
`drossel=0 %`, und ein Reglerwechsel wie eine Massenzahl-Änderung mitten im
Lauf hält sie nicht mehr an.

**Warum der Deadlock „beim 2. Mal" wegblieb** — und wie man ihn lokal
reproduziert: Der Deadlock braucht `step > avail`, und `step` ist bei
kaltem Start gross (grobes Streaming, weil `_bw` noch auf dem 4-MB/s-
Startwert steht). Beim zweiten Filmstart ist die Schätzung warm, `step`
klein, die Falle schnappt nicht. Lokal (Loopback) tritt er NIE auf, weil
die Leitung nie der Flaschenhals ist — Chrome-Netzdrosselung erreicht den
Loopback-WebSocket nicht. Reproduzierbar nur mit einer server-seitigen
Sende-Bremse: nach `ws.send` ein `await asyncio.sleep(len(frame) /
(MBs*1024*1024))`. Bei 3 MB/s und dann PAUSE→Ring läuft auf 69 %,
`drossel=99 %`→FORTSETZEN hängt der alte Code 24 s (permanent), der neue
läuft an. Die Bremse ist Diagnose-Werkzeug, kein Dauercode.

Beim Lesen der Formel hilft: Die frühere Schreibweise
`rate · min(1, istVorratS/FILM_PUFFER_S)` mit `istVorratS = Vorrat/rate`
ist dieselbe Funktion — die Rate kürzt sich heraus. Der Playhead läuft
also mit „Vorrat je 1,5 s", **unabhängig vom eingestellten Tempo**.

### 6.22 Weichzeichnungs-Regler im PM — technisch gefixt, physikalisch begrenzt
**Zwei getrennte Ursachen, die zusammen den Eindruck „wirkt nichts" gaben.**

**1. Technisch (behoben):** `sg-eps` hatte nur einen `input`-Handler
(Anzeige), aber **keinen `change`-Handler** wie `sg-n`/`sg-t`. Der Regler
änderte den Wert, startete den Film aber NIE neu — der laufende Producer
behielt sein altes `softening_au`. Egal was am Kernel geschraubt wurde,
der Regler blieb folgenlos. Fix: `change` → `loadScenario` (index.html).

**2. Physikalisch (Grenze des Verfahrens):** Die PM-Untergrenze ist jetzt
`softening_zellen = 1,0` (war 1,5), der Regler steuert `softening_au`
darüber: `eps = max(1,0 · Zellweite, softening_au)`. Damit WIRKT er — aber
nur im gröberen Bereich. Die Messtabelle für den Dichtekontrast
(`softeningAU`-Kommentar, Kernel A) läuft von eps/Abstand 0,03 (Kontrast
7,2×) bis 0,5 (3,8×) — der interessante, filament-schärfende Teil. Im PM
ist eps ≥ 1 Zellweite ≈ **1,4 × Abstand**, also KOMPLETT oberhalb der
Tabelle, im flachen, weggeglätteten Ast. PM kann prinzipiell nicht feiner
als ~1 Zelle auflösen (dafür TreePM). Der Regler ändert also die Glättung
im Bereich 1–4 Zellen, aber der dramatische Struktur-Kontrast ist so nicht
erreichbar.

**Wer spürbaren Schärfe-Effekt im PM will**, muss an die GITTERAUFLÖSUNG
(grid_n) statt ans Softening — feiner rechnen. Haken: grid > √N ist
schrotrausch-dominiert (viele leere Zellen, siehe `NBodyPM.__init__`). Das
wäre ein eigenes Paket (adaptives/feineres Gitter, evtl. gekoppelt an den
Regler).

### 6.23 OFFEN: Tracer frieren beim Zoom ein (der „Halo") + Punktdichte
**Ein Bug, der zwei Symptome erzeugt** — vom Nutzer diagnostiziert.

Der Client interpoliert Massen und Tracer verschieden:
- **Massen tragen einen Index** (`sel_m`, server.py `build_frame`). Der
  Client ordnet sie über Samples per Index zu — robust gegen
  Auswahl-Änderung, sie laufen immer mit der aktuellen Zeit.
- **Tracer sind anonym** (kein Index, bewusst — Millionen ohne
  Index-Overhead). Der Client interpoliert sie NUR, wenn zwei Samples
  gleich viele haben (`s1.tqx.length === nt`, index.html ~6604). Bei
  Kamerabewegung ändert das Sichtfenster-Culling (`build_frame`, `in-box`)
  die Tracer-ANZAHL → Bedingung falsch → Tracer frieren auf dem letzten
  Sample ein, während die Massen weiterlaufen.

**Symptom 1 (Halo):** Massen wandern (orange), Tracer stehen (blau) →
oranger Rand-Halo. Nur bei/nach Kamerabewegung, seit der Farbtrennung
(6.19) sichtbar.
**Symptom 2 (Punktdichte):** Dieselbe Culling-Abhängigkeit lässt die
Tracer-Zahl im Bild beim Zoom springen — irritierend, obwohl `lod_budget`
(Regler auf Max) konstant ist.

**Fix-Plan (ein Umbau löst beide):** Die Tracer NICHT aufs Sichtfenster
cullen, sondern eine **feste Hash-Stichprobe** über alle Tracer (wie das
deterministische Massen-LOD, `_lod_auswahl`). Dann ist die Auswahl über
Samples stabil → die anonyme Interpolation greift → sie laufen mit.

**Haken (deshalb ein eigenes, sauber zu bauendes Paket):** Ohne
Sichtfenster-Cull brauchen die Tracer eine EIGENE Box zur
u16-Quantisierung (die Verteilungs-Ausdehnung statt des Sichtfensters), im
Sample mitübertragen — ein Protokoll-Umbau über Server UND Client mit
Versionssprung. Trade-off: u16-Auflösung der Tracer wird global (≈2 AE bei
135.000 AE Ausdehnung) statt sichtfenster-fein; beim tiefen Zoom auf
einzelne Tracer gröber. Für Deko-Tracer vertretbar, die Massen bleiben
sichtfenster-f32-präzise. Der Client cullt die Tracer dann beim Zeichnen
(`wxMin/wxMax`) statt der Server.

Nebenbei zu klären: Die Hubble-Expansion (jetzt per Regler, 6.24) treibt
Massen und Tracer gemeinsam auseinander — sie erzeugt KEINE Divergenz, ist
also nicht die Halo-Ursache, aber die physikalische Quelle der weiten
Verteilung (p99 = 135.000 AE nach 918 Sim-Jahren, Start 60.000).

### 6.24 Hubble-Expansion per Regler — erledigt
Neuer Slider „Expansion" (`sg-hubble`, unter der Weichzeichnung): skaliert
die marginal gebundene Rate (`strukturHubbleFaktor` × `√(2GM/rMax)/rMax`).
0 = Kollaps, 1 = marginal (bisheriges Verhalten), >1 = offen/dauerhaft
expandierend. Wirkt auf Massen (`buildStrukturbildung`) und Tracer
(`_tracerAuftrag.hubble`). Wie `sg-eps` löst eine Änderung einen
Sitzungsneustart aus (`change` → `loadScenario`).
