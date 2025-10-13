#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Relativistic Boris with RF via scalar potential Φ only (no direct E0).
E is ALWAYS computed as E = -∇Φ using central finite differences.
RF amplitude comes from a surface charge density σ(t) within the gap.

Modes:
- "rf_gap"  : Φ from ±σ(t) plates at ±d (uniform E inside gap).
- "dc_gap"  : same with constant σ.
- "point"   : softened Coulomb potential.
"""

import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

# Physical constants
c = 299_792_458.0
epsilon_0 = 8.8541878128e-12
epsilon_r = 1.0

# ---------- Helpers ----------
def gamma_from_v(v: np.ndarray) -> float:
    v2 = float(np.dot(v, v))
    if v2 >= c*c:
        v2 = 0.999999999999*c*c
    return 1.0/np.sqrt(1.0 - v2/(c*c))

def uniform_B(B0: float) -> np.ndarray:
    return np.array([0.0, 0.0, B0], dtype=float)

def grad_potential_central(f_phi, x: np.ndarray, t: float, h: float) -> np.ndarray:
    E = np.zeros(3, dtype=float)
    for k in range(3):
        e = np.zeros(3); e[k] = 1.0
        xp = x + h*e
        xm = x - h*e
        phi_p = f_phi(xp, t)
        phi_m = f_phi(xm, t)
        E[k] = -(phi_p - phi_m) / (2.0*h)
    return E

@dataclass
class RFPhase:
    phi: float = 0.0
    phi0: float = 0.0
    def half(self, omega: float, dt: float) -> float:
        return self.phi + 0.5*omega*dt
    def advance(self, omega: float, dt: float) -> None:
        self.phi += omega*dt
        self.phi = (self.phi + np.pi) % (2.0*np.pi) - np.pi

# ---------- Potentials Φ ----------
def phi_gap_rf(x: np.ndarray, t: float, n_hat: np.ndarray, d: float, sigma_t: float) -> float:
    ndotx = float(np.dot(n_hat, x))
    pref = -(sigma_t/(epsilon_0*epsilon_r))
    if abs(ndotx) <= d:
        return pref * ndotx
    else:
        return pref * d * np.sign(ndotx)

def phi_gap_dc(x: np.ndarray, t: float, n_hat: np.ndarray, d: float, sigma_const: float) -> float:
    return phi_gap_rf(x, t, n_hat, d, sigma_const)

def phi_point_soft(x: np.ndarray, t: float, q0: float, soft: float = 1e-9) -> float:
    r = float(np.linalg.norm(x))
    return q0 / (4.0*np.pi*epsilon_0*epsilon_r*np.sqrt(r*r + soft*soft))

# ---------- Boris ----------
def boris_push(x: np.ndarray, v: np.ndarray, q: float, m: float, dt: float,
               E_half: np.ndarray, B_half: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gamma = gamma_from_v(v)
    p = gamma * m * v
    p_minus = p + q * E_half * (0.5*dt)
    gamma_minus = np.sqrt(1.0 + np.dot(p_minus, p_minus)/(m*m*c*c))
    t_vec = (q * B_half / (gamma_minus * m)) * (0.5*dt)
    t2 = float(np.dot(t_vec, t_vec))
    s_vec = (2.0 * t_vec) / (1.0 + t2)
    p_prime = p_minus + np.cross(p_minus, t_vec)
    p_plus  = p_minus + np.cross(p_prime, s_vec)
    p_new = p_plus + q * E_half * (0.5*dt)
    gamma_new = np.sqrt(1.0 + np.dot(p_new, p_new)/(m*m*c*c))
    v_new = p_new / (gamma_new * m)
    x_new = x + v_new * dt
    return x_new, v_new

# ---------- Params & Sim ----------
@dataclass
class SimParams:
    dt: float = 1e-11
    steps: int = 30_000
    q: float = 1.602176634e-19
    m: float = 1.67262192369e-27
    B0: float = 1.0
    n_hat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    d: float = 5e-4
    mode: str = "rf_gap"           # 'rf_gap' | 'dc_gap' | 'point'
    B_in_gap: bool = True
    rf_phi0: float = 0.35*np.pi
    sigma0: float = 5e-7           # surface charge amplitude (C/m^2)
    q0_point: float = -1e-9
    x0: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float))
    v0: np.ndarray = field(default_factory=lambda: np.array([6.0e4, 0.0, 0.0], dtype=float))
    h_grad: float = 1e-6

def run_sim(params: SimParams):
    x = params.x0.astype(float).copy()
    v = params.v0.astype(float).copy()
    traj = np.zeros((params.steps, 3), dtype=float)
    rf = RFPhase(phi=0.0, phi0=params.rf_phi0)
    vels = np.zeros(params.steps)

    h = min(params.h_grad, 0.02*params.d)

    for i in range(params.steps):
        B = uniform_B(params.B0)
        gamma = gamma_from_v(v)
        omega = abs(params.q) * np.linalg.norm(B) / (gamma * params.m)

        if params.mode == "rf_gap":
            phi_half = rf.half(omega, params.dt) + rf.phi0
            sigma_t = params.sigma0 * np.sign(np.sin(phi_half))
            f_phi = lambda X, T: phi_gap_rf(X, T, params.n_hat, params.d, sigma_t)
        elif params.mode == "dc_gap":
            f_phi = lambda X, T: phi_gap_dc(X, T, params.n_hat, params.d, params.sigma0)
        elif params.mode == "point":
            f_phi = lambda X, T: phi_point_soft(X, T, params.q0_point)
        else:
            raise ValueError("Unknown mode")

        E_half = grad_potential_central(f_phi, x, t=0.0, h=h)

        if not params.B_in_gap and abs(float(np.dot(params.n_hat, x))) <= params.d:
            B_half = np.zeros(3)
        else:
            B_half = B

        x, v = boris_push(x, v, params.q, params.m, params.dt, E_half, B_half)
        traj[i] = x

        if params.mode == "rf_gap":
            rf.advance(omega, params.dt)

        vels[i] = np.linalg.norm(v)

        print(i, "\t", "|r| = ", np.linalg.norm(x), "\t", "|vel|/c = ", vels[i] / c, "\t \gamma = ", gamma)

    return traj, vels


def plot_trajectory(traj: np.ndarray, params: SimParams, title="Relativistic Boris – RF phase integrated"):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=1.0)

    # draw the two gap planes ±d
    n = params.n_hat / np.linalg.norm(params.n_hat)
    # build two vectors spanning the plane orthogonal to n
    if abs(n[0]) < 0.9:
        t1 = np.array([1.0, 0.0, 0.0])
    else:
        t1 = np.array([0.0, 1.0, 0.0])
    t1 -= np.dot(t1, n) * n
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)

    w = 4 * params.d
    h = 4 * params.d
    for sign in (-1.0, 1.0):
        center = sign * params.d * n
        corners = [
            center + (-w) * t1 + (-h) * t2,
            center + ( w) * t1 + (-h) * t2,
            center + ( w) * t1 + ( h) * t2,
            center + (-w) * t1 + ( h) * t2,
        ]
        poly = Poly3DCollection([corners], alpha=0.2, facecolor='r', edgecolor='k')
        ax.add_collection3d(poly)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(title)
    ax.set_box_aspect((1, 1, 0.3))

    plt.tight_layout()
    return fig, ax

def plot_data(params: SimParams, vels: np.ndarray, title="Velocity vs time"):
    # tempi vettorializzati
    t = np.arange(vels.size) * params.dt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, vels)             # oppure vels/c se vuoi normalizzare
    ax.set_xlabel("t (s)")
    ax.set_ylabel("|v| (m/s)")   # se normalizzi: "|v|/c"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax

# ----------------------
# Demo (you can comment this part out and import the module)
# ----------------------
if __name__ == "__main__":
    params = SimParams(
        dt=1.0e-12,
        steps=60_000,
        q=1.602176634e-19,
        m=1.67262192369e-27,
        B0=3.0,
        n_hat=np.array([1.0, 0.0, 0.0], dtype=float),
        d=5e-4,
        sigma0=1e-1,
        mode="rf_gap",
        B_in_gap=True,
        rf_phi0=0.35*np.pi,
        x0=np.array([0.0, 0.0, 0.0], dtype=float),
        v0=np.array([6.0e4, 0.0, 0.0], dtype=float),
    )

    traj, vels = run_sim(params)
    fig, ax = plot_trajectory(traj, params)
    fig1, ax1 = plot_data(params, vels)
    plt.show()
