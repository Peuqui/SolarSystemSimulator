"""Gemeinsame Grundlagen der Multi-GPU-Kernel.

`nbody_kernel.py` (wenige Massen, enge Begegnungen, f64) und
`selfgrav_kernel.py` (zehntausende selbstgravitierende Massen, geglaettete
Kraft) sind zwei Antworten auf zwei verschiedene physikalische Fragen und
bleiben getrennt. Wie sie ihre Karten koppeln, ist aber dieselbe Frage —
und die wurde zweimal beantwortet, zeilenidentisch.

Hier steht die Antwort einmal:

* **Die System-Barrier** zwischen allen beteiligten Karten. Sie nimmt
  bewusst einen `unsigned int*` statt eines Struct-Zeigers: Jedes Modul
  bringt seinen eigenen Austauschbereich mit (der alte Kernel schiebt
  Gegenkraefte und Minima darueber, der neue Positionssegmente), aber
  gebraucht wird davon nur das Feld der Rundenzaehler.

* **Der Blick auf gemappten Host-Speicher** als Device-Pointer.

ATOMICS-FREI, und das ist keine Stilfrage: System-Atomics auf gemapptem
Host-Speicher sind ueber PCIe nicht unterstuetzt (nur mit
`hostNativeAtomicSupport`, also NVLink-Host-Kopplung). Der erste Wurf des
alten Kernels hing genau deshalb in der Barrier. Jede Karte schreibt
darum ausschliesslich in ihre eigenen Slots und liest die der anderen:
reine Loads und Stores, die jede PCIe-Plattform beherrscht.
"""
from __future__ import annotations

import cupy as cp

G_MAX = 8       # max. Karten im Verbund (bestimmt das GSync-Layout)
PAD = 16        # eigene Cacheline pro Karte gegen False Sharing

# CUDA-Quelltext, den beide Kernel voranstellen. Er definiert G_MAX und
# PAD selbst, damit der C-Code nicht von der Reihenfolge der Python-seitig
# zusammengesetzten Fragmente abhaengt.
BARRIER_SRC = r"""
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define G_MAX 8
#define PAD 16   // eigene Cacheline pro Karte gegen False Sharing

// Systemweite Barrier zwischen allen Karten des Verbunds (Sense ueber
// monoton wachsende Rundenzaehler). Ein Thread pro Karte macht den
// PCIe-Handshake, der Rest haengt im grid.sync. Bei nGpus == 1
// degeneriert sie zum reinen grid.sync.
//
// `rounds` zeigt auf G_MAX * PAD unsigned int im GEMAPPTEN Host-Speicher.
// Bewusst kein Struct-Zeiger: Was die Karten sonst noch austauschen,
// unterscheidet sich je Kernel, die Barrier braucht davon aber nur die
// Rundenzaehler.
__device__ void sys_barrier(unsigned int* rounds, const int gpuId,
                            const int nGpus, unsigned int* barRound,
                            cg::grid_group& grid, const int tid)
{
    grid.sync();
    if (nGpus > 1 && tid == 0) {
        // Alle vorherigen Writes dieser Karte muessen VOR dem
        // Rundenzaehler systemweit sichtbar sein — er kuendigt sie an.
        __threadfence_system();
        const unsigned int r = ++(*barRound);
        ((volatile unsigned int*)rounds)[gpuId * PAD] = r;
        for (int g = 0; g < nGpus; g++) {
            volatile unsigned int* p =
                &((volatile unsigned int*)rounds)[g * PAD];
            while (*p < r) { __nanosleep(256); }
        }
    }
    grid.sync();
}
"""


def device_view_of_host(host_ptr: int, nbytes: int, device: int,
                        dtype=cp.float64):
    """Gemappten Host-Speicher als CuPy-Array im Kontext von `device`.

    Unter UVA (64-bit, alle modernen Karten) ist der Host-Pointer eines
    `cudaHostAlloc(Portable|Mapped)`-Bereichs direkt als Device-Pointer
    gueltig — `cudaHostGetDevicePointer` waere ein No-op und fehlt in
    CuPy ohnehin."""
    with cp.cuda.Device(device):
        mem = cp.cuda.UnownedMemory(host_ptr, nbytes, owner=None)
        return cp.ndarray((nbytes // cp.dtype(dtype).itemsize,), dtype,
                          memptr=cp.cuda.MemoryPointer(mem, 0))
