"""
Core PIC (capitoli 1–5): griglia di Yee 3D, macroparticelle,
deposito di carica (CIC) e di corrente charge-conserving con
spezzettamento della traiettoria in sotto-spostamenti interni alla cella.

Contenuti:
- Strutture dati per particelle e griglia di Yee (staggered).
- Funzione di forma 1D triangolare (CIC) e deposito ρ su centri cella.
- Spezzettamento della traiettoria entro cella (“elementary motions”).
- Deposito corrente su facce (Jx, Jy, Jz) con termini aα e bαβ (conservazione carica).
- Verifica numerica della continuità discreta.

NON implementati qui (capitoli successivi):
- Pusher (Boris, ecc.)
- Aggiornamento campi (Yee/NDF)

Unità: usa un sistema coerente (cgs o SI). Nel codice non vincoliamo le unità.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Generator
import numpy as np

# ---------------------------------------------------------------
# Costanti (opzionali, solo segnaposto)
c = 2.99792458e8  # m/s se usi SI
FOUR_PI = 4.0 * np.pi


# ---------------------------------------------------------------
# Funzioni di forma (CIC) — triangolare 1D centrata al centro cella

def S_cic_1d(x_rel: float, h: float) -> float:
    """
    Funzione di forma 1D triangolare (CIC) centrata nel centro cella.
    x_rel: distanza dal centro cella lungo un asse
    h    : passo di griglia lungo l’asse (Δx, Δy, Δz)
    Ritorna un peso in [0, 1].
    """
    a = 1.0 - 2.0 * abs(x_rel) / h
    return a if a > 0.0 else 0.0


def cic_weights_cell_centers(x: float, y: float, z: float,
                             i: int, j: int, k: int,
                             dx: float, dy: float, dz: float) -> Tuple[float, float, float]:
    """
    Restituisce (ax, ay, az) = pesi 1D CIC rispetto al centro della cella (i,j,k),
    valutati alla posizione (x,y,z).
    """
    xc = (i + 0.5) * dx
    yc = (j + 0.5) * dy
    zc = (k + 0.5) * dz
    ax = S_cic_1d(x - xc, dx)
    ay = S_cic_1d(y - yc, dy)
    az = S_cic_1d(z - zc, dz)
    return ax, ay, az


# ---------------------------------------------------------------
# Strutture dati

@dataclass
class Particles:
    """
    Macroparticelle:
    x, v    : array (Np, 3) posizioni e velocità
    x_prev  : posizioni all’inizio del passo (n)
    q, m    : array (Np,)
    alive   : mask boolean (Np,)
    """
    x: np.ndarray
    v: np.ndarray
    q: np.ndarray
    m: np.ndarray
    x_prev: np.ndarray
    alive: np.ndarray

    @classmethod
    def zeros(cls, Np: int) -> "Particles":
        return cls(
            x=np.zeros((Np, 3), dtype=float),
            v=np.zeros((Np, 3), dtype=float),
            q=np.zeros(Np, dtype=float),
            m=np.ones(Np, dtype=float),
            x_prev=np.zeros((Np, 3), dtype=float),
            alive=np.ones(Np, dtype=bool),
        )

    @property
    def N(self) -> int:
        return self.x.shape[0]


@dataclass
class YeeGrid:
    """
    Griglia di Yee con sorgenti collocate a posizioni staggered:

    ρ  → centri cella:          (Nx, Ny, Nz)      (i+1/2, j+1/2, k+1/2)
    Jx → facce normali a x:     (Nx+1, Ny, Nz)    (i,     j+1/2, k+1/2)
    Jy → facce normali a y:     (Nx, Ny+1, Nz)    (i+1/2, j,     k+1/2)
    Jz → facce normali a z:     (Nx, Ny, Nz+1)    (i+1/2, j+1/2, k)

    Campi E/B allocati ma non aggiornati qui.
    """
    Nx: int
    Ny: int
    Nz: int
    dx: float
    dy: float
    dz: float
    dt: float

    Ex: np.ndarray
    Ey: np.ndarray
    Ez: np.ndarray
    Bx: np.ndarray
    By: np.ndarray
    Bz: np.ndarray

    rho: np.ndarray
    Jx: np.ndarray
    Jy: np.ndarray
    Jz: np.ndarray

    @classmethod
    def allocate(cls, Nx: int, Ny: int, Nz: int, dx: float, dy: float, dz: float, dt: float) -> "YeeGrid":
        # Campi (posizioni Yee tipiche; non usati qui)
        Ex = np.zeros((Nx,   Ny+1, Nz+1), dtype=float)
        Ey = np.zeros((Nx+1, Ny,   Nz+1), dtype=float)
        Ez = np.zeros((Nx+1, Ny+1, Nz  ), dtype=float)
        Bx = np.zeros((Nx+1, Ny,   Nz  ), dtype=float)
        By = np.zeros((Nx,   Ny+1, Nz  ), dtype=float)
        Bz = np.zeros((Nx,   Ny,   Nz+1), dtype=float)
        # Sorgenti
        rho = np.zeros((Nx, Ny, Nz), dtype=float)
        Jx  = np.zeros((Nx+1, Ny, Nz), dtype=float)
        Jy  = np.zeros((Nx, Ny+1, Nz), dtype=float)
        Jz  = np.zeros((Nx, Ny, Nz+1), dtype=float)
        return cls(Nx, Ny, Nz, dx, dy, dz, dt, Ex, Ey, Ez, Bx, By, Bz, rho, Jx, Jy, Jz)

    def clear_sources(self) -> None:
        self.rho.fill(0.0)
        self.Jx.fill(0.0)
        self.Jy.fill(0.0)
        self.Jz.fill(0.0)

    # Bound-check per indici
    def in_cell_indices(self, i: int, j: int, k: int) -> bool:
        return (0 <= i < self.Nx) and (0 <= j < self.Ny) and (0 <= k < self.Nz)

    def face_x_exists(self, i: int, j: int, k: int) -> bool:
        return (0 <= i <= self.Nx) and (0 <= j < self.Ny) and (0 <= k < self.Nz)

    def face_y_exists(self, i: int, j: int, k: int) -> bool:
        return (0 <= i < self.Nx) and (0 <= j <= self.Ny) and (0 <= k < self.Nz)

    def face_z_exists(self, i: int, j: int, k: int) -> bool:
        return (0 <= i < self.Nx) and (0 <= j < self.Ny) and (0 <= k <= self.Nz)


# ---------------------------------------------------------------
# Deposito carica ρ (CIC) sui centri cella

def deposit_charge_CIC(grid: YeeGrid, parts: Particles) -> None:
    """
    Deposita la carica sui centri cella con schema CIC 3D:
    ρ_{i+1/2,j+1/2,k+1/2} += (q/Vc) * Sx * Sy * Sz
    """
    Vc = grid.dx * grid.dy * grid.dz
    for p in range(parts.N):
        if not parts.alive[p]:
            continue
        q = parts.q[p]
        x, y, z = parts.x[p]
        i = int(np.floor(x / grid.dx))
        j = int(np.floor(y / grid.dy))
        k = int(np.floor(z / grid.dz))
        # max 8 centri cella circostanti
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    ii, jj, kk = i + di, j + dj, k + dk
                    if 0 <= ii < grid.Nx and 0 <= jj < grid.Ny and 0 <= kk < grid.Nz:
                        xc = (ii + 0.5) * grid.dx
                        yc = (jj + 0.5) * grid.dy
                        zc = (kk + 0.5) * grid.dz
                        wx = S_cic_1d(x - xc, grid.dx)
                        wy = S_cic_1d(y - yc, grid.dy)
                        wz = S_cic_1d(z - zc, grid.dz)
                        w = wx * wy * wz
                        if w > 0.0:
                            grid.rho[ii, jj, kk] += (q / Vc) * w


# ---------------------------------------------------------------
# Spezzettamento traiettoria entro la cella

def segment_within_cell_limits(grid: YeeGrid, i: int, j: int, k: int) -> Tuple[float, float, float, float, float, float]:
    """Restituisce i confini fisici della cella (i,j,k)."""
    x0 = i * grid.dx
    y0 = j * grid.dy
    z0 = k * grid.dz
    return x0, x0 + grid.dx, y0, y0 + grid.dy, z0, z0 + grid.dz


def split_segment_into_cell_substeps(grid: YeeGrid,
                                     x0: np.ndarray,
                                     x1: np.ndarray) -> Generator[Tuple[int, int, int, np.ndarray, np.ndarray], None, None]:
    """
    Divide lo spostamento da x0 a x1 in sottosegmenti, ciascuno interamente
    contenuto in una singola cella (i,j,k).
    Ritorna (i, j, k, start, dseg).
    """
    p0 = np.array(x0, dtype=float)
    p1 = np.array(x1, dtype=float)

    # Cella corrente dalla posizione iniziale
    i = int(np.floor(p0[0] / grid.dx))
    j = int(np.floor(p0[1] / grid.dy))
    k = int(np.floor(p0[2] / grid.dz))

    if not grid.in_cell_indices(max(min(i, grid.Nx - 1), 0),
                                max(min(j, grid.Ny - 1), 0),
                                max(min(k, grid.Nz - 1), 0)):
        return

    d = p1 - p0
    remaining = d.copy()
    cur = p0.copy()

    while True:
        xL, xR, yL, yR, zL, zR = segment_within_cell_limits(grid, i, j, k)
        lam = 1.0
        hit_axis = None

        # Trova la frazione lam fino alla prima faccia colpita
        for axis, (pos, dpos, L, R) in enumerate(((cur[0], remaining[0], xL, xR),
                                                  (cur[1], remaining[1], yL, yR),
                                                  (cur[2], remaining[2], zL, zR))):
            if abs(dpos) < 1e-300:
                continue
            if dpos > 0.0:
                lam_axis = (R - pos) / dpos
            else:
                lam_axis = (L - pos) / dpos
            if lam_axis < lam:
                lam = lam_axis
                hit_axis = axis

        dseg = remaining * lam
        start = cur.copy()
        yield (i, j, k, start, dseg)

        cur = cur + dseg
        remaining = p1 - cur
        if np.linalg.norm(remaining, ord=np.inf) < 1e-300:
            break

        # Avanza alla cella adiacente lungo l’asse colpito
        if hit_axis == 0:
            i += 1 if d[0] > 0 else -1
        elif hit_axis == 1:
            j += 1 if d[1] > 0 else -1
        elif hit_axis == 2:
            k += 1 if d[2] > 0 else -1
        else:
            break

        if not grid.in_cell_indices(max(min(i, grid.Nx - 1), 0),
                                    max(min(j, grid.Ny - 1), 0),
                                    max(min(k, grid.Nz - 1), 0)):
            break


# ---------------------------------------------------------------
# Deposito di corrente charge-conserving (cap. 5)

def deposit_segment_current(grid: YeeGrid,
                            i: int, j: int, k: int,
                            start: np.ndarray, dseg: np.ndarray,
                            q: float) -> None:
    """
    Deposita la corrente generata da un sottosegmento interamente interno
    alla cella (i,j,k). Correnti su facce Jx, Jy, Jz (staggered).
    Convenzione: J accumula la fluenz integrata sul dt (carica attraversata).
    """
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    Vc = dx * dy * dz

    # Pesi CIC valutati al punto iniziale del sottosegmento (rispetto al centro cella)
    ax, ay, az = cic_weights_cell_centers(start[0], start[1], start[2], i, j, k, dx, dy, dz)

    # Spostamento normalizzato (per i termini misti bαβ)
    ux = dseg[0] / dx
    uy = dseg[1] / dy
    uz = dseg[2] / dz
    bxy = (1.0 / 12.0) * ux * uy
    byz = (1.0 / 12.0) * uy * uz
    bzx = (1.0 / 12.0) * uz * ux

    Wrho = q / Vc  # carica -> densità

    # ---- Jx su facce (i, j+1/2, k+1/2) attorno alla cella (i,j,k)
    val = dseg[0] * Wrho
    if grid.face_x_exists(i, j, k):
        grid.Jx[i, j, k] += val * (ay * az + byz)
    if grid.face_x_exists(i, j-1, k):
        grid.Jx[i, j-1, k] += val * ((1.0 - ay) * az - byz)
    if grid.face_x_exists(i, j, k-1):
        grid.Jx[i, j, k-1] += val * (ay * (1.0 - az) - byz)
    if grid.face_x_exists(i, j-1, k-1):
        grid.Jx[i, j-1, k-1] += val * ((1.0 - ay) * (1.0 - az) + byz)

    # ---- Jy su facce (i+1/2, j, k+1/2)
    val = dseg[1] * Wrho
    if grid.face_y_exists(i, j, k):
        grid.Jy[i, j, k] += val * (az * ax + bzx)
    if grid.face_y_exists(i, j, k-1):
        grid.Jy[i, j, k-1] += val * ((1.0 - az) * ax - bzx)
    if grid.face_y_exists(i-1, j, k):
        grid.Jy[i-1, j, k] += val * (az * (1.0 - ax) - bzx)
    if grid.face_y_exists(i-1, j, k-1):
        grid.Jy[i-1, j, k-1] += val * ((1.0 - az) * (1.0 - ax) + bzx)

    # ---- Jz su facce (i+1/2, j+1/2, k)
    val = dseg[2] * Wrho
    if grid.face_z_exists(i, j, k):
        grid.Jz[i, j, k] += val * (ax * ay + bxy)
    if grid.face_z_exists(i, j-1, k):
        grid.Jz[i, j-1, k] += val * (ax * (1.0 - ay) - bxy)
    if grid.face_z_exists(i-1, j, k):
        grid.Jz[i-1, j, k] += val * ((1.0 - ax) * ay - bxy)
    if grid.face_z_exists(i-1, j-1, k):
        grid.Jz[i-1, j-1, k] += val * ((1.0 - ax) * (1.0 - ay) + bxy)


def deposit_current_charge_conserving(grid: YeeGrid, parts: Particles) -> None:
    """
    Deposita le correnti Jx, Jy, Jz per tutte le particelle sul passo dt
    rispettando la continuità discreta (spezzettamento entro cella).
    """
    for p in range(parts.N):
        if not parts.alive[p]:
            continue
        x0 = parts.x_prev[p]
        x1 = parts.x[p]
        q = parts.q[p]
        for (i, j, k, start, dseg) in split_segment_into_cell_substeps(grid, x0, x1):
            if dseg.dot(dseg) == 0.0:
                continue
            if not grid.in_cell_indices(i, j, k):
                continue
            deposit_segment_current(grid, i, j, k, start, dseg, q)


# ---------------------------------------------------------------
# Verifica della continuità discreta

def continuity_residual(grid: YeeGrid, rho_prev: np.ndarray, rho_curr: np.ndarray, dt: float) -> float:
    """
    Calcola || ∂ρ/∂t + ∇·J ||_∞ sui centri cella.
    Nota: J è fluenz (integrata su dt), quindi usiamo J/dt come corrente media.
    """
    # Divergenza a centri cella via differenze finite compatibili con Yee
    divJ = np.zeros_like(grid.rho)
    divJ += (grid.Jx[1:, :, :] - grid.Jx[:-1, :, :]) / (grid.dx * dt)
    divJ += (grid.Jy[:, 1:, :] - grid.Jy[:, :-1, :]) / (grid.dy * dt)
    divJ += (grid.Jz[:, :, 1:] - grid.Jz[:, :, :-1]) / (grid.dz * dt)

    drho_dt = (rho_curr - rho_prev) / dt
    res = drho_dt + divJ
    return float(np.max(np.abs(res)))


# ---------------------------------------------------------------
# Stub aggiornamento campi (capitoli successivi)

def update_fields_ampere_faraday_stub(grid: YeeGrid) -> None:
    """Segnaposto: aggiornamento E/B (da implementare nei capitoli successivi)."""
    pass


# ---------------------------------------------------------------
# Esempio d’uso minimale
if __name__ == "__main__":
    # Griglia 8×8×8, passo unitario, dt = 0.1
    grid = YeeGrid.allocate(8, 8, 8, 1.0, 1.0, 1.0, 0.1)

    # Una particella che si muove leggermente in diagonale
    parts = Particles.zeros(1)
    parts.q[:] = 1.0
    parts.m[:] = 1.0
    parts.x_prev[0] = np.array([3.3, 2.7, 4.1])
    parts.x[0]      = np.array([3.8, 3.05, 4.6])

    # ρ^{n+1} con posizioni "finali"
    grid.clear_sources()
    deposit_charge_CIC(grid, parts)
    rho_n1 = grid.rho.copy()

    # ρ^{n} con posizioni iniziali
    grid.clear_sources()
    saved = parts.x.copy()
    parts.x[:] = parts.x_prev
    deposit_charge_CIC(grid, parts)
    rho_n = grid.rho.copy()
    parts.x[:] = saved  # ripristina

    # Deposito correnti (fluenz sul dt)
    grid.clear_sources()
    deposit_current_charge_conserving(grid, parts)

    # Verifica continuità
    res = continuity_residual(grid, rho_n, rho_n1, grid.dt)
    print("Residuo continuità (norma infinito):", res)
