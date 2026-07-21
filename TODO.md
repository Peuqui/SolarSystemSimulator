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

1. **Kernel A allein**, f32, festes Softening, fester Zeitschritt, keine
   Kollisionen. Ein neues Szenario mit 10.000 Massen und **ohne** Tracer.
   Das ist schon das Zwanzigfache dessen, was heute im Worker läuft — und
   erst danach weiß man, ob das Netz wirklich entsteht.
2. **Kernel B** dazu, 200.000–300.000 Tracer.
3. Kollisionen nachrüsten — über ein Raumgitter, nicht alle gegen alle.
   Oder weglassen: Bei Softening verschmelzen Teilchen ohnehin nicht mehr
   sinnvoll.
4. Film-Stream prüfen: 30.000 Körper × 8 B sind 240 KB pro Sample. Das
   Protokoll trägt das, das LOD-System sowieso.

Zwischenschritt, der schon läuft: das Szenario „Strukturbildung
(selbstgravitierend)" mit 500 Körpern im WebWorker.

## Kernel-Grenze M_MAX wird nicht geprüft

`M_MAX = 64` begrenzt die massiven Körper, weil sie im Shared Memory des
Kernels liegen (`s_mx[M_MAX]` und Nachbarn). Eine Prüfung, ob eine Szene
diese Grenze überschreitet, gibt es weder im Client noch in `server.py`.

Erreichbar ist das im normalen Betrieb: Es genügt, genügend Rogues zu
injizieren. Was der Kernel dann tut, ist ungeprüft — im besten Fall rechnet
er die überzähligen Körper nicht mit, im schlechteren schreibt er über die
Shared-Memory-Arrays hinaus.

Zu klären: Wo wird die Zahl der massiven Körper festgestellt, und was soll
beim Überschreiten passieren — Ablehnen mit Meldung (wie bei den
f64-empfindlichen Szenarien, die nicht auf die GPU dürfen) oder gleich der
Tiling-Kernel von oben, womit die Grenze ganz entfiele.
