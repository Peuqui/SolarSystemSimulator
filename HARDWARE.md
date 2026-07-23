# Hardware & Findings

Deutsch | [English](HARDWARE.en.md)

Der Referenz-Rechner, auf dem das CUDA-Backend und der Film-Modus
entwickelt und gemessen wurden — ein ungewöhnliches Fünf-GPU-Setup an einem
**Mini-PC**, sowie die Erkenntnisse aus dem Betrieb. Die Zahlen im README
und in den Kommentaren beziehen sich auf diese Maschine.

## Die Konfiguration

| Komponente | |
|---|---|
| **Rechner** | Mini-PC (AMD Ryzen 7 7840HS, 8 Kerne / 16 Threads, Zen 4) |
| **RAM** | 32 GB (30,7 GB nutzbar) |
| **OS** | Ubuntu 24.04 LTS, Kernel 7.0 |
| **Display** | AMD Radeon 780M (integriert) — treibt den Monitor |
| **Rechen-GPUs** | 3 × Tesla V100-PCIE (32 GB, Boost 1380 MHz) |
| | 2 × Quadro RTX 8000 (48 GB, Boost 1620 MHz) |
| **Anbindung** | alle über **M.2-zu-Oculink- bzw. USB4-Adapter**, jeweils **PCIe Gen3 ×4** |

Fünf ausgewachsene Rechenzentrums-/Workstation-GPUs an einem handflächen-
großen Mini-PC, dessen eigene Grafik eine integrierte 780M ist. Die Karten
hängen nicht in Slots, sondern an externen Adaptern — das ist der
entscheidende Punkt, an dem sich die Architektur des Backends ausrichtet.

## Der ungewöhnliche Teil: eGPUs über ×4-Links

Ein normaler Server gibt jeder Karte PCIe ×16. Hier hat **jede Karte nur
×4** (Gen3 ≈ 4 GB/s pro Richtung), weil M.2/Oculink/USB4 nicht mehr Bahnen
führen. Und es gibt **kein NVLink** — die Karten können nicht direkt
miteinander reden.

Das klingt nach einem Krüppel-Setup, ist aber für die passende Last kein
Problem — vorausgesetzt, der Code respektiert es. Genau darauf ist das
Backend ausgelegt:

- **Kein Peer-to-Peer, keine NVLink-Annahmen.** Die Substep-Synchronisation
  der Karten läuft **atomics-frei über PCIe-gemappten Host-Speicher** — jede
  Karte sieht dieselben Bytes über PCIe, kein direkter GPU-zu-GPU-Transfer.
- **Minimaler Bus-Verkehr.** Über den ×4-Link geht pro Substep nur eine
  winzige Barriere-Struktur, nicht der Körper-Zustand. Der D2H-Download der
  Ergebnisse ist der eigentliche Bus-Posten und liegt gemessen bei
  ~3,2 GB/s — der ×4-Link ist damit ausgereizt, aber es reicht.

## Findings

### 1. Echte parallele Auslastung — anders als LLM-Inferenz
Auf demselben Rechner lasten LLMs (ohne Tensorparallelität) immer nur
**eine** Karte gleichzeitig aus: Autoregressive Generierung plus
Pipeline-Split heißt Staffellauf — Karte 0 rechnet ihre Layer, reicht die
Aktivierung weiter, die anderen warten. N-Body ist das Gegenteil:
**peinlich parallel.** Die Körper werden auf alle Karten verteilt, jede
rechnet ihre Scheibe gleichzeitig, ein Sync pro Substep. Erstmals laufen
**alle fünf Karten zugleich am Anschlag** (100 %, 160–170 W) — nicht
seriell nacheinander.

### 2. f64/f32-Split: jede Karte tut, worin sie stark ist
Die RTX 8000 (Turing) sind in **f64** grottig (1/32 der f32-Rate) — auf dem
Papier für f64-Physik nutzlos. Aber der selbstgravitierende Kernel misst,
dass der **Zustand** f64 braucht, die **Kraftsumme** aber f32 verträgt (das
Softening glättet sie ohnehin). Damit rechnen die schwachen f64-Karten die
Kräfte in f32 voll mit — und statt zwei brachliegender Karten ziehen alle
fünf. Über fünf Karten ist das Faktor ~4,2 gegenüber f64 auf drei V100.

### 3. „100 % Auslastung" heißt fast nichts
`nvidia-smi` zeigt `utilization.gpu`, und das misst nur, ob **irgendein
Kernel resident** war — nicht, ob die Rechenwerke arbeiten. Gemessen: drei
V100 bei **100 % Util und 42 W** (von 250 W), bei vollem Takt. Die Karten
waren wach und hatten nichts zu tun. Wer echte Auslastung wissen will, nimmt
die **Leistungsaufnahme** (unter Last 160–170 W) oder rechnet Paare pro
Sekunde gegen die FLOP-Zahl. Ein leichtes Szenario (56 Massen × 150k Tracer)
lastet die Karten nominell zu 100 % aus, verbrennt aber Idle-Watt.

### 4. Speicher ist nie der Engpass — die Rechenzeit ist es
Bei 750.000 Körpern belegt die Physik **< 1,1 GB von 32–48 GB** VRAM. Die
Wand ist die O(N²)-Produktionsrate, nicht der Platz. Deshalb ist der nächste
Optimierungshebel ein **schnellerer Algorithmus** (Particle-Mesh / TreePM,
O(N log N)), nicht mehr Hardware — die Karten könnten weit mehr Körper
tragen, wenn der Kernel cleverer wäre.

### 5. Film-Bandbreite ist selbstbegrenzend
Der Film-Stream über LAN kostet `angezeigte Punkte × Bytes × Samples/s`.
Gemessen: 20.832 Punkte → 6,4 MB/s, 120.000 Punkte → 62 MB/s. Aber je
schwerer die Szene, desto **langsamer die Produktion** — bei 750k Körpern
nur ~9 Samples/s, sodass die Bandbreite trotz der vielen Punkte moderat
bleibt (~40 MB/s). Deshalb läuft selbst das Extremszenario flüssig über ein
Gigabit-Heimnetz; der Loopback-Vorteil eines lokalen Zugriffs greift erst
jenseits von ~180.000 gleichzeitig gestreamten Punkten.

## Gemessene Leistung (selbstgravitierender Kernel, fünf Karten)

| Massen | Tempo | Anmerkung |
|---|---|---|
| 50.000 | ~37 Sim-Jahre / Minute | |
| 100.000 | ~9 | |
| 200.000 | ~2 | Kosten ~N^2,5 (Softening schrumpft mit dem Abstand → feinerer Zeitschritt) |
| 750.000+ | flüssig abspielbar | 250k Massen + 500k Tracer, fünf Karten bei 100 % / 160–170 W |

Der Aufwand wächst mit **N^2,5**, nicht nur N²: Mehr Teilchen bedeuten
gleiche Gesamtmasse, aber feinere Auflösung — und weil das Softening mit dem
Teilchenabstand schrumpft, wird zusätzlich der Zeitschritt feiner. Mehr
Körper heißt also *schärfer aufgelöst*, nicht *mehr Materie*.
