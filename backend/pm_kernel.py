"""Particle-Mesh-Kraftberechnung (isolierte Randbedingung, 2D-Ebene).

Ersetzt die O(N²)-all-pairs-Summe des selbstgravitierenden Kernels durch
eine Faltung ueber ein Gitter — O(N log N) statt O(N²). Erster Schritt der
PM/TreePM-Roadmap (siehe TODO.md); hier NUR die Kraft, gegen all-pairs in
`test_pm.py` validiert, bevor sie in `film_producer.py` eingebunden wird.

WICHTIG — 3D-Gravitation in der 2D-Ebene:
Der Simulator ist 2D (x, y), aber die Kraft faellt mit 1/r² (der all-pairs-
Kernel rechnet a = G·m·(r_j−r_i)/(|Δ|²+ε²)^1,5). Die Faltungs-Green-Funktion
ist deshalb die geglaettete 3D-Kraft K(Δ) = −G·Δ/(|Δ|²+ε²)^1,5, NICHT der 2D-
ln(r)-Kern. So stimmt PM grossraeumig exakt mit all-pairs ueberein.

ISOLIERTE Randbedingung (Hockney/James, kein periodisches Universum):
Das Rechengitter wird auf 2N×2N genullt (Zero-Padding), die Masse liegt nur
im N×N-Quadranten, und der Green-Kern wird so ueber die 2N-Periode gelegt,
dass die umgewickelten Kopien in den Nullbereich fallen. Nur der [0:N,0:N]-
Ausschnitt der Faltung ist gueltig. Eine naive periodische FFT wuerde die
periodischen Bilder umwickeln und am Rand falsche Kraefte liefern.

Faltung DIREKT mit der Kraft-Green-Funktion (zwei Komponenten K_x, K_y)
statt ueber ein Potential mit anschliessender Differenziation: reproduziert
die geglaettete Kraft treuer und vermeidet den zusaetzlichen Gitter-Fehler
der Ableitung.
"""
from __future__ import annotations

import cupy as cp

from nbody_kernel import G_AU


def build_force_kernels(grid_n: int, h: float, eps2: float,
                        G: float = G_AU) -> tuple:
    """FFT der beiden Kraft-Green-Funktionen auf dem 2N×2N-Padding-Gitter.

    Einmal vorab bauen, dann in jeder Kraftauswertung wiederverwenden — die
    Kerne haengen nur an Gittergroesse, Zellweite und Softening, nicht am
    Zustand. Rueckgabe: (FKx, FKy) als rfft2-Spektren (2N, N+1).
    """
    n_pad = 2 * grid_n
    # off[m] bildet den FFT-Index m auf den ECHTEN Versatz ab: 0..N-1 bleibt,
    # N..2N-1 wird negativ (−N..−1). So liegt Versatz 0 auf Index 0, und die
    # negativen Versaetze fuellen die obere Haelfte — genau die Anordnung,
    # die die zirkulare FFT-Faltung mit dem Zero-Padding braucht.
    off = cp.arange(n_pad, dtype=cp.float64)
    off = cp.where(off < grid_n, off, off - n_pad)
    dx = (off * h)[:, None]           # (2N, 1) — Versatz entlang Achse 0 (x)
    dy = (off * h)[None, :]           # (1, 2N) — Versatz entlang Achse 1 (y)
    r2 = dx * dx + dy * dy + eps2
    inv_r3 = 1.0 / (r2 * cp.sqrt(r2))  # 1/(|Δ|²+ε²)^1,5
    # K(Δ) = −G·Δ/(...)^1,5. Das Minus macht die Faltung (ρ ⊛ K)[p] direkt
    # zur Beschleunigung a(p) = Σ_j m_j·G·(r_j−p)/... (nachgerechnet: der
    # Versatz Δ = p − q = Feld − Quelle, also Δ_x = x_p − x_q, und
    # −G·Δ_x/... = G·(x_q − x_p)/... = Zug zur Quelle). Bei Δ=0 ist Δ_x=0 →
    # K_x=0: keine Selbstkraft.
    kx = (-G) * dx * inv_r3
    ky = (-G) * dy * inv_r3
    return cp.fft.rfft2(kx), cp.fft.rfft2(ky)


def _cic_deposit(gx, gy, gm, grid_n, x0, y0, h):
    """Cloud-in-Cell: jede Masse auf die 4 umliegenden Gitterpunkte
    verteilt. Rueckgabe: Massengitter (N, N) — Summe = Gesamtmasse."""
    fx = (gx - x0) / h
    fy = (gy - y0) / h
    i = cp.floor(fx).astype(cp.int32)
    j = cp.floor(fy).astype(cp.int32)
    # In [0, N-2] halten, damit i+1 / j+1 gueltig bleiben. Bei ausreichendem
    # Rand (siehe grid_fuer) greift die Klemmung nicht.
    i = cp.clip(i, 0, grid_n - 2)
    j = cp.clip(j, 0, grid_n - 2)
    tx = fx - i
    ty = fy - j
    # Vier Ecken in EINEM bincount — Indizes und Gewichte konkateniert.
    idx = cp.concatenate([
        i * grid_n + j,
        (i + 1) * grid_n + j,
        i * grid_n + (j + 1),
        (i + 1) * grid_n + (j + 1),
    ])
    w = cp.concatenate([
        gm * (1 - tx) * (1 - ty),
        gm * tx * (1 - ty),
        gm * (1 - tx) * ty,
        gm * tx * ty,
    ])
    rho = cp.bincount(idx, weights=w, minlength=grid_n * grid_n)
    return rho.reshape(grid_n, grid_n)


def _cic_gather(feld, gx, gy, grid_n, x0, y0, h):
    """Cloud-in-Cell rueckwaerts: Feldwert an der Teilchenposition aus den 4
    umliegenden Gitterpunkten interpolieren — DASSELBE Schema wie beim
    Deposit, sonst entstuende eine Selbstkraft."""
    fx = (gx - x0) / h
    fy = (gy - y0) / h
    i = cp.clip(cp.floor(fx).astype(cp.int32), 0, grid_n - 2)
    j = cp.clip(cp.floor(fy).astype(cp.int32), 0, grid_n - 2)
    tx = fx - i
    ty = fy - j
    f = feld.ravel()
    return (f[i * grid_n + j] * (1 - tx) * (1 - ty)
            + f[(i + 1) * grid_n + j] * tx * (1 - ty)
            + f[i * grid_n + (j + 1)] * (1 - tx) * ty
            + f[(i + 1) * grid_n + (j + 1)] * tx * ty)


def grid_fuer(gx, gy, grid_n: int, rand_zellen: float = 3.0):
    """Gitter-Geometrie aus der Punktwolke: Ursprung (x0, y0) und Zellweite
    h, sodass alle Punkte mit `rand_zellen` Zellen Rand hineinpassen.

    Gibt (x0, y0, h) zurueck. Fuer die Integration im Betrieb waere das
    Gitter fest verdrahtet; fuers Kraft-Testen leitet es sich aus der
    aktuellen Wolke ab."""
    xmin = float(cp.min(gx)); xmax = float(cp.max(gx))
    ymin = float(cp.min(gy)); ymax = float(cp.max(gy))
    spanne = max(xmax - xmin, ymax - ymin, 1e-30)
    # Rand auf beiden Seiten -> nutzbare Zellen = N - 2*rand
    h = spanne / (grid_n - 2 * rand_zellen)
    x0 = xmin - rand_zellen * h
    y0 = ymin - rand_zellen * h
    return x0, y0, h


def pm_accelerations(gx, gy, gm, grid_n, x0, y0, h, eps2,
                     G: float = G_AU, kernels=None):
    """Beschleunigung (ax, ay) je Teilchen via Particle-Mesh.

    gx, gy, gm: CuPy-Arrays (Positionen, Massen). kernels: optional das
    Ergebnis von build_force_kernels (spart den Neubau)."""
    if kernels is None:
        kernels = build_force_kernels(grid_n, h, eps2, G)
    fkx, fky = kernels
    n_pad = 2 * grid_n

    rho = _cic_deposit(gx, gy, gm, grid_n, x0, y0, h)   # (N, N)
    rho_pad = cp.zeros((n_pad, n_pad), cp.float64)
    rho_pad[:grid_n, :grid_n] = rho

    fr = cp.fft.rfft2(rho_pad)
    ax_grid = cp.fft.irfft2(fr * fkx, s=(n_pad, n_pad))[:grid_n, :grid_n]
    ay_grid = cp.fft.irfft2(fr * fky, s=(n_pad, n_pad))[:grid_n, :grid_n]

    ax = _cic_gather(ax_grid, gx, gy, grid_n, x0, y0, h)
    ay = _cic_gather(ay_grid, gx, gy, grid_n, x0, y0, h)
    return ax, ay
