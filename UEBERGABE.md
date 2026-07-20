# Übergabeprotokoll — Stand 2026-07-20 (abends)

Fortschreibung nach der Umsetzung der Punkte 2.1, 2.3 und 2.4 des
Vormittags-Protokolls. Vorherige Commits: `4916dd9`, `462d9a7`.

---

## 1. Gemessene Kennzahlen

Szene: 267k Körper, Raster 0,5 Tage, drei Tesla V100 (Physik) + Quadro
RTX 8000 (Erkennung), Client über LAN.

| Posten | Anteil | Bemerkung |
|---|---|---|
| Bounce-Erkennung | **75–93 %** | der Engpass |
| Physik (`step_batch`, 3× V100) | 4–20 % | |
| Merge-Erkennung | 2–4 % | |
| Anwenden der Kollisionen | 1–3 % | |
| Ring-Schreiben (`tobytes`) | 1–5 % | |

**Produktionsrate:** 60k → 120 Sim-Tage/s · 117k → 16 · 267k → 3–6
(stark von der räumlichen Verteilung abhängig, nicht nur von N).

**PCIe:** Alle fünf Karten hängen mit **×4** an M.2-zu-Oculink- bzw.
USB4-Adaptern (Mini-PC). Hardwarebedingt, nicht änderbar.

**GPU-Nummerierung:** `nvidia-smi` zählt nach PCI-Bus, CUDA nach
Rechenleistung. Die drei V100 sind für CUDA 0–2, die beiden RTX 8000
sind 3 und 4. Kein Fehler.

---

## 2. Am 20.07. abends umgesetzt

### 2.1 Erkennung räumlich auf beide RTX 8000 — **erledigt, aber dynamisch**

Umgesetzt wie im Entwurf: `Erkennungskarte` (Klasse in
`film_producer.py`) hält eine GPU samt residenten Arrays und ist für
einen x-Streifen `[lo, hi)` zuständig. Ein Paar gehört ihr, wenn
`min(x_i, x_j)` darin liegt; um jedes eigene Paar sehen zu können,
prüft sie Körper bis `hi + halo`. Nach links braucht sie **keinen**
Halo — läge der Partner links von `lo`, wäre er selbst das Minimum.
Die Streifen sind dadurch lückenlos **und** überschneidungsfrei, es
gibt zwischen den Karten nichts abzustimmen.

`halo = 2h + 2·r_max`: Nachbarzellen werden bis Versatz 1 geprüft, ein
Paar liegt also höchstens 2h auseinander. `h` und die Streifengrenzen
(Quantile der lebenden Asteroiden) werden **einmal global** bestimmt und
an beide Karten gegeben — rechnete jede Karte ihr eigenes `h`, wären
die Halo-Breiten verschieden und die Besitz-Regel nicht mehr dicht.

**Der Befund, der den Entwurf korrigiert:** Der erwartete Faktor 2 tritt
nur bei sehr hoher Kollisionsdichte ein. Gemessen mit
`bench_erkennung.py` (250k Asteroiden, zwei RTX 8000):

| Kandidatenpaare/Sample | 1 Karte | 2 Karten | Gewinn |
|---|---|---|---|
| 3,4 Mio | 8,5 ms | 13,1 ms | 0,65× |
| 6,2 Mio | 12,0 ms | 14,4 ms | 0,83× |
| 11,4 Mio | 17,2 ms | 14,5 ms | 1,18× |
| 22,9 Mio | 31,5 ms | 20,3 ms | 1,55× |
| 91 Mio | 111,5 ms | 64,8 ms | 1,72× |
| 178 Mio | 212,7 ms | 128,9 ms | 1,65× |

Der Umschlagpunkt liegt bei **8–9 Mio Kandidatenpaaren pro Sample**.
Darunter kosten doppelter Batch-Upload und doppelte Kernel-Starts mehr,
als die halbierte GPU-Arbeit einbringt. Ein ausgebildeter Gürtel liegt
mit unter 10^6 klar darunter, frisch injizierte, sich durchdringende
Wolken mit 10^8–10^9 klar darüber.

Deshalb entscheidet `karten_wahl()` **pro Batch** neu (Schwellen
`DET_ZU_AB` = 15 Mio, `DET_AB_UNTER` = 6 Mio, Hysterese). Ein
Film-Neustart ist dafür **nicht** nötig: Streifen werden ohnehin je
Sample zugeteilt, eine ruhende Karte bekommt schlicht keinen Upload
mehr, ihre residenten Arrays bleiben liegen. Zuschalten kostet nur den
nächsten Batch-Upload.

Ende-zu-Ende an der Knödel-Szene (5 eng injizierte Wolken, 250k
Asteroiden) bestätigt: `detkarten=2/2` in der dichten Phase, automatisch
`1/2`, sobald sich der Gürtel gebildet hat.

Jenseits von ~10^10 Paaren fällt der Gewinn wieder auf 1,0×: dort wird
`h` so groß, dass der Halo einen erheblichen Teil des Nachbarstreifens
mit abdeckt. Nur ein Transient, nicht optimiert.

**Regressionsschutz:** `test_erkennung_streifen.py` prüft die Invariante,
an der der Anlauf vom Vormittag scheiterte — die Vereinigung der
Streifen-Treffer ist *exakt* die Trefferliste einer einzelnen Karte,
über gleichverteilte Wolke, dichten Klumpen, zwei getrennte Wolken und
ein Paar direkt auf der Grenze, jeweils mit 2, 3 und 5 Streifen.

### 2.3 Client-Worker — **erledigt**

Der WebSocket liegt jetzt in einem eigenen Worker (`NET_WORKER_SRC` in
`index.html`), der die v4-Frames vollständig zerlegt und die Nutzlast
**transferiert** statt kopiert (idx/qx/qy sind Views auf den empfangenen
Puffer).

Präzisierung der Diagnose des Vormittags: die Dekodierung selbst war nie
teuer — sie erzeugt fast nur Views. Teuer ist, dass der Main-Thread pro
Bild in `filmApply` + Rendern hängt und in dieser Zeit **kein**
`onmessage` läuft; der Socket wird nicht geleert, das Empfangsfenster
kollabiert. Der Worker leert ihn unabhängig davon weiter.

Der Rest des Clients spricht unverändert `send()`/`close()`/
`readyState` — über eine schmale Fassade auf den Worker. Der Live-CUDA-
Pfad (status 0) läuft mit, kostet eine `postMessage`-Etappe (~µs gegen
ms Netz-Roundtrip).

Im Browser durchgetestet: Verbindungsaufbau, Verfügbarkeits-Probe mit
sofortigem Trennen, Film-Stream (Samples, Ereignisse), f64-Dump zur
Engine-Übergabe (status 5), Film-Neustart danach. Keine Konsolenfehler.

### 2.6 Dichte-LOD des Streams — **neu, erledigt**

Befund aus dem Betrieb: Bei 300k–515k Körpern war der ursprüngliche
Asteroidengürtel „wie eine schüttere Glatze" praktisch verschwunden.

Ursache war die alte Ausdünnung `sel % stride == 0` — eine **feste Rate
über den Original-Index**, also überall derselbe Anteil. Der Gürtel hat
eine feste kleine Objektzahl (800 + 600), verteilt über riesige Ringe;
bei `stride 5` blieben ~280 Punkte übrig. Injizierte Wolken verlieren
denselben Anteil, wirken aber kompakt weiter dicht. Die Regel war also
**dichteblind**: sie löscht dünne Strukturen und lässt dichte unberührt
wirken.

Ersetzt durch `_lod_auswahl` / `_dichte_filter` mit **Rangfolge**:

1. Massive Körper (Sonne, Planeten, Rogues, Sterne, SL) — nie ausgedünnt
   (war schon vorher so: `| ~self._is_ast[sel]`)
2. Asteroiden des geladenen Systems
3. Nachträglich injizierte Wolken — bekommen den Rest

Innerhalb jeder Stufe gilt `behalten ~ anzahl^0,5` je Gitterzelle.
Nicht nivellierend: dichte Gebiete bleiben
sichtbar dichter, nur nicht mehr um den vollen Faktor. Gemessen
(`test_lod_dichte.py`, Gürtel 1400 gegen Wolke 300k):

| Budget | Gürtel | Wolke | Dichteverhältnis |
|---|---|---|---|
| 120k | 100 % | 34,5 % | 73,9× statt roh 214× |
| 60k | 100 % | 16,5 % | 35,3× |
| 20k | 100 % | 6,0 % | 12,8× |

Die Auswahl läuft über einen **Hash des Original-Index** gegen eine
stufenlose Rate, nicht über einen ganzzahligen Stride. Ein Stride je Zelle
kann nur die Raten 1, ½, ⅓ … treffen; zwei Nachbarzellen landen dann auf ⅓
und ¼ und unterscheiden sich um ein Drittel Dichte — mit harter Kante
entlang der Zellgrenze, sichtbar als **rechteckige Blockartefakte**. Der
Hash beseitigt zugleich ein zweites Raster: „jeder n-te Index" legt in
gleichmäßig erzeugten Wolken selbst eines an, weil die Körper in
Erzeugungsreihenfolge im Raum liegen. Nebeneffekt: Das Budget wird voll
ausgenutzt (119.931 statt 104.878 von 120.000 — die ganzzahlige Stufung
verschenkte 13 %). Deterministisch, also über Samples stabil.

Das Gitter ist 512×512, nicht 128×128: Die Auto-Box umspannt alle Körper
inklusive weit hinausgeschleuderter, der sichtbare Ausschnitt ist davon
oft nur ein Bruchteil — grobe Zellen wären dort entsprechend groß.

Das Budget ist damit ein **Zielwert**, keine harte Schranke: die Auswahl
trifft die Zellvorgabe im Erwartungswert und streut um ~√Budget (0,7 % bei
20.000). Ein exakter Deckel bräuchte eine Teilsortierung — Aufwand ohne
Wirkung, da das Budget selbst aus einer Bandbreitenschätzung stammt.

**Neuer Regler „Punkte im Bild"** — logarithmisch wie Abspieltempo und
Sample-Raster (linker Anschlag Auto, dann 10.000 … 5 Mio, rechter Anschlag
„Alle" = gar keine Ausdünnung).
Bewusst **nur einer**: die Bildrate hängt an der Summe der Punkte, zwei
Budgets ließen unspielbare Kombinationen zu, ohne dass ein einzelner
Regler verdächtig aussieht — derselbe Fehler wurde bei dt/Verlangsamung
schon einmal korrigiert. Wird über `MSG_FILM_SUB` mitgeschickt und wirkt
daher **ohne Filmneustart**.

**Protokoll:** FILM_START hat ein viertes u8-Array `injiziert`,
Version 4→ Bits sind jetzt `2` (`FILM_PROTO_VERSION` in server.py).

**Nebenbei repariert:** `getState`/`restoreState` sicherten `isAsteroid`
gar nicht — nach Export/Import wären alle Asteroiden massive Körper
geworden (der Producer hätte sie als solche integriert, `M_MAX`!).

**Bekannte Grenzen bei der Ereignis-Anzeige** (nicht angefasst):

- Ein Blitz wird nur gezeigt, wenn beide Partner gestreamt sind
  (`_filmInView`), und es sind höchstens 8 pro Frame. Das Dichte-LOD
  entschärft das dort, wo es auffällt (dünne Gebiete sind jetzt
  vollständig gestreamt), beseitigt es aber nicht.
- Der Server liefert höchstens **1024 Ereignisse pro Frame**. In dichten
  Szenen baut sich ein Rückstand auf; die Ereignisse treffen nach ihrer
  Zeit ein und werden in einem Frame angewandt — der Zähler springt
  schubweise statt zu laufen.
- Läuft der Rückstand über den Ereignisring (65536), springt der Server
  vor (`ev_from = max(sent_ev, ev_total - ev_cap + 8)`) und Ereignisse
  gehen **still verloren**. Der Badge zählt dann dauerhaft zu niedrig
  gegenüber der tatsächlich gerechneten Physik.

### 2.4 Aufräumen — **erledigt**

- Die Dispersions-Diagnose in `analyze_bounce` (`cp.unique`, zwei
  `bincount`, zwei `percentile`, Nachbar-Suche je Sample) ist **weg**.
  Sie gehörte zur verworfenen Idee 3.1 und war damit toter Code, der
  jedes Sample Rechenzeit kostete.
- Die verbliebene Zeitanteils-Diagnose (Producer und `[stream]`) läuft
  nur noch mit `--diag`. Sie ist für die Engpass-Suche gebaut und kostet
  jetzt praktisch nichts mehr, hat im Alltag aber nichts zu suchen.
- `analyze_bounce` und `analyze_merge` waren in `if True:`-Blöcke
  gehüllt (Restgerüst) — entfernt.
- `pick_detect_device` → `pick_detect_devices` (liefert eine Liste).

---

## 3. Noch offen

### 3.1 Datenlokalität GPU-zu-GPU (war 2.2)
**Bewertet, nicht umgesetzt — der Entwurf trägt so nicht.**

Der Gedanke war, den Host-Umweg zu sparen. Das geht so nicht auf:
`step_batch` liefert bewusst ein Host-Array, und das wird auch
gebraucht — `bounce_deltas` rechnet auf der CPU, `apply_merges` liest
Host-Positionen. Ein P2P-Transfer würde also nur den **H2D** zur
Erkennungskarte ersetzen, nicht den **D2H** von den Physikkarten. Dafür
müsste zusätzlich der Shard-Scatter in die Originalreihenfolge auf die
GPU wandern.

Der billigere Hebel zuerst: **Pinned (page-locked) Host-Puffer** für
`out` in `step_batch`. Verdoppelt den PCIe-Durchsatz in beide
Richtungen ohne Architekturänderung. Vorher messen — mit zwei aktiven
Erkennungskarten wird pro Batch doppelt hochgeladen, der Anteil ist
also gestiegen.

### 3.1b Interpolation weiter verbessern (Ideen, nicht umgesetzt)
Der Client interpoliert die Film-Positionen jetzt mit **Catmull-Rom**
(glatte Kurve durch vier Sample-Punkte) statt linearer Sehne — behebt das
Pumpen in Sonnennähe, gemessen 29× genauer auf einer Kreisbahn. Zwei
Hebel bleiben für den Extremfall (Perihel, wo ein Körper fast eine halbe
Umrundung pro Raster macht — dort sind die Samples selbst zu grob):

- **Geschwindigkeit mitstreamen** (Hermite mit echten Tangenten statt aus
  Nachbar-Samples geschätzten). Verdoppelt die Bandbreite pro Punkt
  (16 statt 8 Byte); nur nötig, wenn Catmull-Rom sichtbar nicht reicht.
- **Adaptiv feineres Raster für sonnennahe/schnelle Körper** — analog zur
  „enge Begegnungen präzise"-Mechanik des Hybrid-Backends (astAdaptive im
  Worker). Dort wird der Substep an nahen schweren Körpern gedrückt; das
  Muster ließe sich für das Film-Sample-Raster übernehmen.

### 3.1c Bekannte Grenze: abrupter Verbindungsverlust im Film
Bricht die CUDA-Verbindung WÄHREND des Films unerwartet ab (Server-Crash,
Netzwerk, `systemctl stop`), nullt der `onclose`-Pfad `filmQueue`, **ohne**
vorher `filmReconstructVelocities()` aufzurufen — die Körper landen im
Worker mit v≈0 und stürzen in die Sonne. Der GEPLANTE Engine-Wechsel ist
sauber (Dump + Rekonstruktion über `filmStop`). Fix wäre einzeilig
(Rekonstruktion im `onclose` vor dem Nullen), auf Nutzerwunsch vorerst
nicht gebaut — im Normalbetrieb tritt der Fall nicht auf.

### 3.2 nginx (braucht sudo)
- `sites-available/narnia` ist seit Mai **nicht synchron** mit
  `sites-enabled/narnia`. Aktiv ist `sites-enabled` (kein Symlink!).
  Zwei Wahrheiten für dieselbe Config.
- `/solar-system/` hat **kein `Cache-Control`** — anders als
  `/haemotrace/` und `/ai-atc/`. Daher nach jeder `index.html`-Änderung
  ein harter Reload nötig.

### 3.3 Socket-Unit auf `0.0.0.0:8765`
Testrest der Direktverbindung; das Backend ist damit ohne Auth im LAN
erreichbar. Original gesichert, auf Wunsch des Nutzers vorerst so
belassen. `?cudaws=ws://host:8765` im Client hängt den nginx-Proxy ab
(Diagnose) und kann bleiben.

---

## 4. Verworfen — nicht erneut versuchen

### 4.1 Bounce-Reichweite auf lokale Dispersion umstellen
`h = 2·v95·dt` richtet sich nach der **absoluten** Bahngeschwindigkeit,
obwohl für Kollisionen nur die **relative** zählt. Gemessen und
verworfen: das Maximum der Dispersion liegt 40- bis 100-mal über dem
95-Perzentil. Eine garantiert korrekte Reichweite wäre rund
**vierzigmal größer** als die heutige, nicht kleiner.

**Nebenbefund, wichtig:** Der bestehende Ansatz deckt damit nur die
langsamsten 95 % ab. Kollisionen sehr schneller Körper — Sonnennähe,
ausgeschleuderte Asteroiden — werden **bereits heute nicht gefunden**.
Bewusste Näherung, kein Regressionsfehler, aber eine bekannte
Genauigkeitsgrenze.

### 4.2 Multi-GPU-Schwelle senken
`test_kernel.py` legte nahe, eine Einzelkarte sei schneller als drei.
Im Betrieb umgestellt → fünf- bis sechsmal langsamer. Zurückgenommen;
eine saubere Wiederholung ergab 244 gegen 198 Tage/s zugunsten des
Verbunds. Die Schwelle von 30k ist korrekt, Warnung steht im Code.

---

## 5. Betriebswissen

**Was wirkt wann:**

| Geändert | Wirksam durch |
|---|---|
| `index.html` | **Harter Reload** (Strg+Shift+R) — kein Cache-Control! |
| `film_producer.py` | Film aus/an (neuer Prozess per `spawn`) |
| `server.py` | Backend-Prozess muss beendet werden |

Der Server läuft socket-aktiviert mit `--idle-exit 120`: Er beendet sich
erst **120 s nach der letzten getrennten Verbindung**. Ein Reload reicht
nicht. Sauber: `systemctl --user stop solar-cuda.service`.

**Prozesse korrekt zählen** (`pgrep -af "server.py"` matcht die eigene
Shell-Zeile mit!):
```bash
pgrep -c -f "SolarSystemSimulator/venv/bin/python server.py"
```

**Logs:** `journalctl --user -u solar-cuda.service -f`
(Zeitanteile nur mit `--diag` in der Unit.)

**Tests und Messwerkzeuge** — Standalone-Skripte, kein pytest:
```bash
cd backend
../venv/bin/python test_film_golden.py        # Ende-zu-Ende Kollisionskette
../venv/bin/python test_erkennung_streifen.py # Streifen == eine Karte
../venv/bin/python test_kernel.py             # Kernel + Multi-GPU
../venv/bin/python bench_erkennung.py         # Umschaltschwelle der 2. Karte
../venv/bin/python bench_film.py -n 250000 --szene knoedel --det-gpus 1 2
```

**ruff** liegt nicht im Projekt-venv:
`/home/mp/Projekte/AIfred-Intelligence/venv/bin/ruff check backend/`

**Messfallen:**
- Ein einzelner `nvidia-smi`-Aufruf misst in einem Stop-and-Go-System
  zufällig eine Pause. Immer Zeitreihen nehmen.
- **Die Kandidatenzahl ist die relevante Größe, nicht N.** Dieselben
  250k Asteroiden ergeben als Knödel 10^9 und als Gürtel 10^5 Paare pro
  Sample — Faktor 10^4 in der Erkennungslast. Zwei Messungen mit
  gleicher Objektzahl sind darum **nicht** vergleichbar.
- `bench_film.py --szene knoedel` durchläuft diese Entwicklung: Der
  Gesamtwert am Ende ist irreführend, phasenweise nach Kandidatenzahl
  vergleichen.
- Vor jeder Messung prüfen, ob der laufende Prozess den geänderten Code
  geladen hat (Startzeit gegen `stat -c '%y'` der Datei).

---

## 6. Am 20.07. vormittags behoben (zur Einordnung)

- **Live-Deadlock** über `_filmHeadRate` (Kante stand → Rate 0 →
  gegenseitiges Warten).
- **Handover-Race:** `filmStart()` legte neue `filmRefs` an, während der
  f64-Dump der alten Session unterwegs war. Start wird jetzt aufgeschoben
  und vom Dump ausgelöst.
- **Neustart-Endlosschleife:** `_filmSentN` gegen `bodies.length`
  verglichen statt gegen die gefilterte Länge.
- **Playhead-Klemmung** im Server (`[tail, head]`).
- **Interpolation:** Körper nur im älteren Sample galten als aktuell →
  statische Flecken.
- **Bandbreitenschätzung:** `_bw` maß die Dauer von `await ws.send`
  (Socket-Puffer!). Jetzt über ein Fenster.
- **Frame-Kosten:** Drosselung rechnete mit `sample_bytes` statt der
  tatsächlichen Framegröße.
- **Leichen-Filter:** Tote Asteroiden werden beim Filmstart aussortiert.
- **Ringpuffer** per `--ring-gib` konfigurierbar (Default 8 GiB).
