import numpy as np
from typing import Callable, Tuple

# Physical constants
c = 299_792_458.0               # speed of light [m/s]
epsilon_0 = 8.8541878128e-12    # permittivity of free space [F/m]
mu_0 = 4e-7 * np.pi             # permeability of free space [H/m]
e  = 1.602176634e-19            # carica unitaria
mp = 1.67262192369e-27          # massa protone
me = 9.1093837015e-31           # massa elettrone
def gamma_from_v(v: np.ndarray) -> float:
    v2 = float(np.dot(v, v))
    if v2 >= c*c:
        v2 = 0.999999999999 * c * c
    return 1.0 / np.sqrt(1.0 - v2 / (c*c))

# ---------- Numerical operators ----------
def grad_potential_central(x: np.ndarray,
                           pos: np.ndarray,
                           potential: Callable[[np.ndarray, float], float],
                           t: float,
                           h: float = 1e-6) -> np.ndarray:
    """E = -∇φ at point x, where potential(x_rel, t) expects x_rel = x - pos."""
    F = np.zeros(3, dtype=float)
    r0 = x - pos
    for k in range(3):
        e = np.zeros(3); e[k] = 1.0
        phi_p = potential(r0 + h*e, t)
        phi_m = potential(r0 - h*e, t)
        F[k] = -(phi_p - phi_m) / (2.0 * h)
    return F

def curl_central(x: np.ndarray,
                 pos: np.ndarray,
                 A_func: Callable[[np.ndarray, float], np.ndarray],
                 t: float,
                 h: float = 1e-6) -> np.ndarray:
    """(∇×A) at point x. A_func(r_rel, t) expects r_rel = x - pos."""
    r0 = x - pos
    dA = np.zeros((3, 3), dtype=float)
    for i in range(3):
        e = np.zeros(3); e[i] = 1.0
        A_p = np.asarray(A_func(r0 + h*e, t), dtype=float)
        A_m = np.asarray(A_func(r0 - h*e, t), dtype=float)
        dA[i, :] = (A_p - A_m) / (2.0*h)
    curl = np.zeros(3, dtype=float)
    curl[0] = dA[1, 2] - dA[2, 1]
    curl[1] = dA[2, 0] - dA[0, 2]
    curl[2] = dA[0, 1] - dA[1, 0]
    return curl

# ---------- Particle-particle fields (non-relativistic / quasi-static) ----------
def coulomb_E(q: float, r_vec: np.ndarray, softening: float = 0.0) -> np.ndarray:
    """Electric field of a point charge q at displacement r_vec.
    Uses Plummer softening: r -> sqrt(r^2 + a^2) to avoid singularities.
    """
    r2 = float(np.dot(r_vec, r_vec)) + softening*softening
    r = np.sqrt(r2)
    if r == 0.0:
        return np.zeros(3)
    return (q / (4.0 * np.pi * epsilon_0)) * (r_vec / (r2 * r))

def motional_B(q: float, v_source: np.ndarray, r_vec: np.ndarray, softening: float = 0.0) -> np.ndarray:
    """Magnetic field from a moving point charge (low-β approximation):
    B = μ0/(4π) * q (v × r̂) / r^2, with softening like above.
    Valid when |v| << c and for modest separations (no retardation).
    """
    r2 = float(np.dot(r_vec, r_vec)) + softening*softening
    r = np.sqrt(r2)
    if r == 0.0:
        return np.zeros(3)
    r_hat = r_vec / r
    return (mu_0 / (4.0 * np.pi)) * q * np.cross(v_source, r_hat) / r2

def fields_from_particles(particles, i_self: int,
                          soft_E: float = 0.0, soft_B: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Return (E,B) on particle i_self due to all other particles in the list."""
    E = np.zeros(3, dtype=float)
    B = np.zeros(3, dtype=float)
    xi = particles[i_self].pos
    for j, pj in enumerate(particles):
        if j == i_self:
            continue
        r_vec = xi - pj.pos
        E += coulomb_E(pj.q, r_vec, softening=soft_E)
        B += motional_B(pj.q, pj.vel, r_vec, softening=soft_B)
    return E, B
