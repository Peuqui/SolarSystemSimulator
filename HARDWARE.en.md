# Hardware & Findings

[Deutsch](HARDWARE.md) | English

The reference machine on which the CUDA backend and film mode were
developed and benchmarked — an unusual five-GPU setup on a **mini PC** —
plus what we learned running it. The numbers in the README and code comments
refer to this machine.

## The configuration

| Component | |
|---|---|
| **Machine** | Mini PC (AMD Ryzen 7 7840HS, 8 cores / 16 threads, Zen 4) |
| **RAM** | 32 GB (30.7 GB usable) |
| **OS** | Ubuntu 24.04 LTS, kernel 7.0 |
| **Display** | AMD Radeon 780M (integrated) — drives the monitor |
| **Compute GPUs** | 3 × Tesla V100-PCIE (32 GB, 1380 MHz boost) |
| | 2 × Quadro RTX 8000 (48 GB, 1620 MHz boost) |
| **Attachment** | all via **M.2-to-Oculink / USB4 adapters**, each **PCIe Gen3 ×4** |

Five full datacenter/workstation GPUs on a palm-sized mini PC whose own
graphics is an integrated 780M. The cards don't sit in slots — they hang off
external adapters, and that's the key constraint the backend architecture is
built around.

## The unusual part: eGPUs over ×4 links

A normal server gives each card PCIe ×16. Here **every card has only ×4**
(Gen3 ≈ 4 GB/s per direction), because M.2/Oculink/USB4 carry no more lanes.
And there is **no NVLink** — the cards cannot talk to each other directly.

That sounds crippled, but for the right workload it isn't a problem —
provided the code respects it. That's exactly what the backend is designed
for:

- **No peer-to-peer, no NVLink assumptions.** Per-substep synchronization
  between cards runs **atomics-free over PCIe-mapped host memory** — every
  card sees the same bytes over PCIe, no direct GPU-to-GPU transfer.
- **Minimal bus traffic.** Only a tiny barrier structure crosses the ×4 link
  per substep, not the body state. The D2H download of results is the real
  bus cost, measured at ~3.2 GB/s — the ×4 link is maxed out, but it's
  enough.

## Findings

### 1. Genuine parallel utilization — unlike LLM inference
On the same machine, LLMs (without tensor parallelism) only ever load **one**
card at a time: autoregressive generation plus pipeline split is a relay —
card 0 computes its layers, hands the activation on, the others wait. N-body
is the opposite: **embarrassingly parallel.** Bodies are split across all
cards, each computes its slice simultaneously, one sync per substep. For the
first time **all five cards run flat out at once** (100 %, 160–170 W), not
serially one after another.

### 2. f64/f32 split: each card does what it's good at
The RTX 8000 (Turing) are terrible at **f64** (1/32 of the f32 rate) — on
paper useless for f64 physics. But the self-gravitating kernel measures that
the **state** needs f64 while the **force sum** tolerates f32 (softening
smooths it anyway). So the weak-f64 cards compute the forces in f32 at full
speed — and instead of two idle cards, all five pull. Across five cards
that's a factor of ~4.2 over f64 on three V100s.

### 3. "100 % utilization" means almost nothing
`nvidia-smi` shows `utilization.gpu`, which only measures whether **some
kernel was resident** — not whether the execution units are working.
Measured: three V100s at **100 % util and 42 W** (of 250 W), at full clock.
The cards were awake with nothing to do. To know real load, use **power
draw** (160–170 W under load) or count pairs per second against the FLOP
budget. A light scenario (56 masses × 150k tracers) shows 100 % util while
burning idle watts.

### 4. Memory is never the bottleneck — compute time is
At 750,000 bodies the physics uses **< 1.1 GB of 32–48 GB** VRAM. The wall is
the O(N²) production rate, not space. That's why the next optimization lever
is a **faster algorithm** (Particle-Mesh / TreePM, O(N log N)), not more
hardware — the cards could carry far more bodies if the kernel were smarter.

### 5. Film bandwidth is self-limiting
The film stream over LAN costs `displayed points × bytes × samples/s`.
Measured: 20,832 points → 6.4 MB/s, 120,000 points → 62 MB/s. But the heavier
the scene, the **slower the production** — at 750k bodies only ~9 samples/s,
so bandwidth stays moderate despite the many points (~40 MB/s). That's why
even the extreme scenario plays smoothly over a gigabit home network; the
loopback advantage of local access only matters beyond ~180,000
simultaneously streamed points.

## Measured performance (self-gravitating kernel, five cards)

| Masses | Speed | Note |
|---|---|---|
| 50,000 | ~37 sim-years / minute | |
| 100,000 | ~9 | |
| 200,000 | ~2 | cost ~N^2.5 (softening shrinks with spacing → finer timestep) |
| 750,000+ | plays smoothly | 250k masses + 500k tracers, five cards at 100 % / 160–170 W |

Cost grows as **N^2.5**, not just N²: more particles means the same total
mass but finer resolution — and because the softening shrinks with the
particle spacing, the timestep gets finer too. More bodies means *sharper
resolution*, not *more matter*.
