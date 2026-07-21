# TODO

## Selbstgravitierende Teilchen (kosmisches Netz)

**Ziel:** Filamente, Knoten und Voids, die aus der Simulation *entstehen*
statt nachgezeichnet zu werden.

**Warum es heute nicht geht.** Die Testteilchen der Galaxienhaufen-Szenarien
sind masselos und ziehen sich gegenseitig nicht an. Der Kernel berechnet ihre
Beschleunigung ausschließlich gegen die massiven Körper
(`backend/nbody_kernel.py`, Schleife `for kk < M`), und `M` ist auf
`M_MAX = 64` gedeckelt — die massiven Körper liegen im Shared Memory.

Ein Asteroid spürt also Galaxien, aber niemals einen anderen Asteroiden.
Verdichtung setzt aber genau das voraus: Eine Ansammlung muss sich *selbst*
anziehen, um enger zu werden. Ohne das folgen die Teilchen nur dem
vorgegebenen Feld weniger Punktmassen und bleiben so glatt verteilt, wie sie
gestartet sind. Das kosmische Netz entsteht in der Natur durch anisotropen
Kollaps selbstgravitierender Überdichten: erst zu Flächen, dann zu Linien,
dann zu Knoten.

**Was zu bauen wäre.** Ein zweiter Rechenpfad „alle gegen alle" mit Tiling
(Standardtechnik für N-Body auf GPUs: Blockweise in Shared Memory laden,
statt eine feste Obergrenze anzunehmen). Protokoll und Film-Pfad bleiben
unberührt, es geht nur um die Kraftberechnung.

**Aufwandsschätzung** (drei V100, f64, 48 Schritte/s für flüssige Wiedergabe):

| Teilchen | Paare/Schritt | Bedarf | Ergebnis |
|---|---|---|---|
| 150.000 | 2,25 · 10¹⁰ | ~20 TFLOP/s — an der Hardwaregrenze | 5–25 Tage/s statt 242 |
| 20.000 | 4 · 10⁸ | mit Reserve machbar | flüssig, für 2D-Filamente ausreichend |

Empfehlung: mit 20.000 anfangen. Für sichtbare Struktur in zwei Dimensionen
reicht das, und die Wiedergabe bleibt schnell.

**Zwischenschritt, der ohne neuen Kernel geht:** ein Szenario aus wenigen
*massiven* Körpern statt vieler masseloser — die ziehen sich im vorhandenen
Code bereits gegenseitig an. Zeigt Verdichtung im Kleinen, bleibt aber bei
CUDA an `M_MAX = 64` gebunden.

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
f64-empfindlichen Szenarien, die nicht auf die GPU dürfen) oder Kernel mit
Tiling, womit die Grenze ganz entfiele.
