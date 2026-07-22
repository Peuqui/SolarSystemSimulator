# TODO

## Selbstgravitierende Teilchen auf die GPU (kosmisches Netz)

**Ziel:** Filamente, Knoten und Voids, die aus der Simulation *entstehen*
statt nachgezeichnet zu werden — mit zehntausenden Massen statt 56.

### Warum es heute nicht geht

Zwei Dinge stehen im Weg, und die Konstante ist nur das kleinere.

**1. Die Testteilchen sind masselos.** Der Kernel berechnet ihre
Beschleunigung ausschließlich gegen die massiven Körper
(`nbody_kernel.py`, Schleife `for kk < M`). Ein Asteroid spürt Galaxien,
niemals einen anderen Asteroiden. Verdichtung setzt aber genau das voraus:
Eine Ansammlung muss sich *selbst* anziehen, um enger zu werden. Ohne das
folgen die Teilchen nur dem Feld weniger Punktmassen und bleiben so glatt
verteilt, wie sie gestartet sind. Das kosmische Netz entsteht in der Natur
durch anisotropen Kollaps selbstgravitierender Überdichten: erst zu
Flächen, dann zu Linien, dann zu Knoten.

**2. Der Kernel ist für „wenige Massen" gebaut.** Er lädt sie mit
`threadIdx.x < M` in den Shared Memory (Block = 256) und rechnet
massiv×massiv anschließend in `if (tid == 0)` — also O(M²) **seriell auf
einem einzigen GPU-Thread**. Bei 500 Körpern wären das 250.000 Paare auf
einem Thread, weit langsamer als der WebWorker. `M_MAX = 64` ließe sich
erhöhen, dieser Flaschenhals nicht.

### Der bestehende Kernel bleibt

Er ist für sein Regime richtig: wenige Körper, enge Begegnungen, Bahnen
auf der Messerschneide, f64. Softening und feste Zeitschritte wären dort
falsch — sie würden Lagrange-Punkte und Choreografien verfälschen.

Die neuen Kernel treten **daneben**, nicht an seine Stelle. Das ist keine
Doppelung, sondern die passende Antwort auf zwei verschiedene physikalische
Fragen.

### Zwei neue Kernel

**Kernel A — Masse gegen Masse.** All-pairs mit Tiling (Nyland/Harris):

    Jeder Thread besitzt einen Zielkörper i.
    Für jede Kachel von p = blockDim.x Quellkörpern:
        Block lädt die p Quellen kooperativ nach Shared Memory (x, y, m)
        __syncthreads()
        Jeder Thread summiert die Kräfte dieser p Quellen auf seinen Körper
        __syncthreads()

Shared Memory: 256 × 24 B = 6 KB, **unabhängig von N**. Damit fallen weg:
`M_MAX`, das Laden per `threadIdx.x < M`, die Gegenkraft-Buchhaltung mit
`backX`/`backY` und `atomicAdd`, der serielle Thread-0-Block.

Newtons drittes Gesetz wird bewusst **nicht** ausgenutzt: Es spart die
halbe Rechnung, erzwingt aber Synchronisation zwischen Threads und macht
das Ergebnis von der Ausführungsreihenfolge abhängig. Doppelt rechnen ist
auf der GPU billiger als sich abstimmen — und bleibt deterministisch, was
für den Multi-GPU-Abgleich zählt.

**Kernel B — Tracer gegen Massen.** Jeder Thread besitzt einen Tracer,
dieselben Massen-Kacheln als Quellen. Keine Rückwirkung, keine
Synchronisation, kein `atomicAdd` — ein Tracer schreibt nur seine eigene
Beschleunigung. Der einfachste denkbare GPU-Kernel, skaliert linear.

Die Testteilchen bekommen **keine** Masse. Sie bleiben Tracer; sie rechnen
künftig nur gegen zehntausende Quellen statt gegen 56.

### Was es kostet

Paare je Kraftauswertung = **(N_Massen + N_Tracer) × N_Massen**. Die Massen
gehen quadratisch ein und multiplizieren zusätzlich die Tracer.

Budget bei f32 auf allen fünf Karten (2× RTX 8000 + 3× V100, 79 TFLOP/s
Peak, 60 % Auslastung, 20 FLOP je Paar, 144 Kraftauswertungen/s):
**16,5 Mrd Paare**.

| Massen | Tracer | Paare | Auslastung |
|---|---|---|---|
| 5.000 | 300.000 | 1,5 Mrd | 9 % |
| 10.000 | 300.000 | 3,1 Mrd | 19 % |
| 30.000 | 300.000 | 9,9 Mrd | 60 % |
| 50.000 | 300.000 | 17,5 Mrd | 106 % — zu viel |
| 30.000 | 600.000 | 18,9 Mrd | 115 % — zu viel |

Heute: 56 × 150.000 = 8,4 Mio Paare. Der Sprung ist Faktor ~1.000, aber
die Hardware trägt ihn — sie hat bisher fast nichts zu tun (V100s zeigen
100 % Auslastung bei 42 W von 250 W, weil `utilization.gpu` nur misst, ob
ein Kernel resident ist, nicht ob die SMs arbeiten).

### Hardware-Aufteilung

Getrennte Kernel erlauben getrennte Zuordnung, und die passt zur
vorhandenen Hardware auffällig gut:

- **Massen in f64 auf die drei V100** (7,8 TFLOP/s f64 je Karte).
- **Tracer in f32 auf die beiden RTX 8000** — in f64 liefern sie nur 1/32
  ihrer Leistung (~0,5 TFLOP/s, praktisch nutzlos), in f32 je ~16 TFLOP/s.

Jede Karte tut das, worin sie stark ist, statt dass zwei von fünf
brachliegen. Zusätzlich möglich: gröbere Zeitschritte für die Tracer, sie
beeinflussen ja nichts.

### Softening macht den neuen Pfad einfacher

Selbstgravitierende Simulationen brauchen **Plummer-Softening** — ohne
dominiert Zweikörper-Streuung das Ergebnis. Faustregel ε ≈ mittlerer
Teilchenabstand / 10; bei 30.000 Körpern auf 100.000 AE also rund 100 AE.

Damit gibt es keine engen Begegnungen mehr, und die halbe Komplexität des
heutigen Kernels entfällt: Klassifikation in „heiße" Asteroiden, private
Feinschleife mit eigenem adaptivem dt, Berührungsprüfung im Substep,
hierarchische Zeitschritte. Ein fester Zeitschritt genügt. Der neue Pfad
wird nicht nur schneller, sondern deutlich weniger Code.

f32 reicht dabei: Bei 100.000 AE Ausdehnung löst f32 auf 0,006 AE auf —
zehntausendmal feiner als das Softening.

### Etappen

1. ~~**Kernel A allein**~~ — **fertig und angebunden.**
   `selfgrav_kernel.py` mit `test_selfgrav.py`, im Film-Producer über
   das Softening im Protokoll wählbar, Szenario „Strukturbildung
   (selbstgravitierend)" mit Reglern für Teilchenzahl und
   Weichzeichnung. Was dabei anders kam als geplant, steht unten unter
   „Was die Messung korrigiert hat".
2. ~~**Kernel B**~~ — **fertig.** Masselose Tracer laufen in derselben
   Kernel-Schleife wie die Massen (`tx/ty/tvx/tvy` in der Signatur),
   nach dem Massen-Update im selben `grid.sync()`-Takt. Sie spüren das
   Feld, wirken aber nicht zurück — darum kosten sie N·T statt N².
   Regler „Tracer" im Szenario, Voreinstellung 30.000.
3. Kollisionen nachrüsten — über ein Raumgitter, nicht alle gegen alle.
   Oder weglassen: Bei Softening verschmelzen Teilchen ohnehin nicht mehr
   sinnvoll.
4. **Film-Stream: der Index kostet die Hälfte.** Nachgemessen (UEBERGABE
   6.14): 8 Byte je Punkt und Sample, davon 4 Byte reiner Index. Bei
   44.212 Körpern sind das 354 KB je Sample; die Wunschrate von 20/s
   ergäbe 7,1 MB/s. Eine Fernverbindung (RTT 66 ms) trug gemessen
   3,45 MB/s — also 9,7 Samples/s, sichtbar als Ruckeln bei völlig
   unbeschäftigter GPU.
   Die LOD-Auswahl ändert sich zwischen zwei Samples kaum. Sendet man
   die Indexliste nur bei Änderung und sonst ein Flag „unverändert",
   halbiert sich die Nutzlast und 20 Samples/s passen durch. Erfordert
   `FILM_PROTO_VERSION = 6` (Server und `zerlegeFilm` im Client).

Zwischenschritt, der schon läuft: das Szenario „Strukturbildung
(selbstgravitierend)" mit 500 Körpern im WebWorker.

### Was die Messung an diesem Konzept korrigiert hat

**Die Hardware-Aufteilung oben (f64-Massen auf V100, f32-Tracer auf RTX)
gilt so nicht für Kernel A.** Gemessen an N=400 über 8.000 Schritte,
Energieerhaltung gegen eine f64-NumPy-Referenz (4,6·10⁻⁵):

| | dE/E |
|---|---|
| alles f32 | 1,3·10⁻¹ |
| f64-Zustand, **f32-Kraft** | 7,9·10⁻⁵ |
| alles f64 | 7,4·10⁻⁵ |

Entscheidend ist der **Zustand**, nicht die Kraft: `v += a·dt/2` addiert
bei v ≈ 1.680 AU/a Inkremente von ~0,2, und die f32-Auflösung liegt dort
schon bei 1·10⁻⁴. Die Kraftsumme dagegen verträgt f32 mühelos — das
Softening glättet sie ohnehin.

Damit rechnen **alle** Karten dasselbe (f64-Zustand, f32-Kraft), und die
RTX 8000 sind vollwertig statt mit 1/32 unbrauchbar: 655 gegen 1.092
Schritte/s einer V100. Über fünf Karten ist das Faktor 4,2 gegenüber f64
auf drei V100.

**Die Kartenwahl misst, statt zu schätzen** (`miss_gewicht`). Eine
Datenblatt-Metrik lag nachweislich falsch herum, und selbst baugleiche
Karten weichen um 11–21 % voneinander ab (siehe UEBERGABE Abschnitt 1).

**Gemessene Skalierung** (fünf Karten, f32-Kraft):

| N | 1 Karte | 5 Karten | Sim-Tage/s |
|---|---|---|---|
| 20.000 | 993 | 825 | 1.657 |
| 50.000 | 147 | 311 | 625 |
| 100.000 | 41 | 91 | 183 |
| 200.000 | 10 | 32 | 64 |

Unter ~30.000 Körpern lohnt der Verbund nicht — die Barrier frisst den
Gewinn. Der Producer nutzt dafür dieselbe Schwelle `MULTI_GPU_AB` wie
der alte Kernel.

**Was die Auslegung wirklich kostet.** Die Tabelle oben hält das
Softening fest. Physikalisch muss es mit dem Teilchenabstand schrumpfen
(~1/√N), und weil der Zeitschritt an ε hängt (η = v·dt/ε ≤ 0,12,
gemessen — bei 0,25 bricht die Energieerhaltung um Faktor 40 ein),
kosten mehr Teilchen doppelt: **N^2,5**. Gemessen mit ε = 0,1 × Abstand:

| N | ε (AU) | dt (Tage) | Sim-Jahre pro Minute |
|---|---|---|---|
| 10.000 | 100,0 | 1,63 | 371 |
| 50.000 | 44,7 | 0,73 | 37,5 |
| 100.000 | 31,6 | 0,52 | 8,7 |
| 200.000 | 22,4 | 0,37 | 2,1 |

Deutliche Klumpen brauchen rund 120 Sim-Jahre. Der praktisch brauchbare
Bereich zum Zuschauen ist damit **25.000–100.000**, nicht 200.000 — und
genau deshalb sind Teilchenzahl und Weichzeichnung Regler: Wer ε groß
lässt, bekommt Tempo statt Schärfe.

## Kernel-Grenze M_MAX — erledigt, aber weiterhin eng

Die Grenze wird jetzt durchgesetzt: Der Server nennt `M_MAX` beim
Handshake, der Client lehnt den Rogue darüber hinaus mit sichtbarer
Meldung ab, und Berstfragmente gelten als Asteroiden (vorher wuchs die
Zahl der massiven Körper bei jedem Zerbersten um 3).

Was bleibt: **64 ist wenig.** Für die selbstgravitierenden Szenarien oben
sind zehntausende Massen das Ziel. Kernel A hebt die Grenze auf — sie
existiert nur, weil die Massen im Shared Memory liegen und
massiv×massiv seriell auf Thread 0 läuft. Bis dahin ist 64 hart.
