import numpy as np
import phisics_sim.src.HystoryBuffer_class as hb
from typing import Callable, List, Tuple

Array = np.ndarray

class Simulation:
    dt: float
    steps: int
    Np: int

    def __init__(self, dt: float, steps: int, Np: int):
        self.dt = float(dt)
        self.steps = int(steps)
        self.Np = int(Np)
        # Output principali
        self.POS  = np.zeros((self.Np, self.steps, 3), dtype=float)   # (Np, steps, 3)
        self.vels = np.zeros((self.steps, self.Np, 3), dtype=float)   # (steps, Np, 3)

    # --------- helper vettoriale per campi esterni ----------
    @staticmethod
    def _accumulate_external_fields(X_half: Array, t_half: float,
                                    capacitors: List, dipoles: List, multipoles: List) -> Tuple[Array, Array]:
        N = X_half.shape[0]
        E_ext = np.zeros((N, 3), dtype=float)
        B_ext = np.zeros((N, 3), dtype=float)

        if capacitors:
            for c in capacitors:
                # somma E per tutti i punti
                for i in range(N):
                    E_ext[i] += c.electric_field(X_half[i], t_half)
        if dipoles:
            for d in dipoles:
                for i in range(N):
                    B_ext[i] += d.magnetic_field(X_half[i], t_half)
        if multipoles:
            for mp in multipoles:
                for i in range(N):
                    B_ext[i] += mp.magnetic_field(X_half[i], t_half)

        return E_ext, B_ext

    def sim(self, particles: list, capacitors: list, dipoles: list, multipoles: list,
            hist_len: int, R_max: float, integrator: Callable,
            PRINT_FORCES: bool, PRINT_EVERY: int = 1):

        # ------------------ Allocazioni e stato ------------------
        Np = len(particles)
        if (self.Np != Np) or (self.POS.shape != (Np, self.steps, 3)):
            self.Np = Np
            self.POS  = np.zeros((Np, self.steps, 3), dtype=float)
            self.vels = np.zeros((self.steps, Np, 3), dtype=float)

        # Stato in forma vettoriale (riduce overhead getattr/setattr in loop)
        pos = np.vstack([p.pos for p in particles]).astype(float)  # (Np,3)
        vel = np.vstack([p.vel for p in particles]).astype(float)  # (Np,3)
        q   = np.array([p.q   for p in particles], dtype=float)    # (Np,)
        m   = np.array([p.mass for p in particles], dtype=float)   # (Np,)

        # History buffer per LW
        hist = hb.HistoryBuffer(Np, hist_len=hist_len, dt=self.dt)
        hist.push(0.0, pos.copy(), vel.copy())

        barr = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
        i = 0

        # ------------------ Time loop ------------------
        for n in range(self.steps):
            t = n * self.dt

            # (1) salva posizioni correnti
            self.POS[:, n, :] = pos

            # (2) campi pp (LW) al tempo t
            E_pp, B_pp = hb.fields_LW_pairwise(t, pos, vel, q, hist, epsilon_r=1.0, R_max=R_max)

            # (3) campi esterni su x_half = x + 0.5 dt v
            X_half = pos + 0.5 * self.dt * vel
            t_half = t + 0.5 * self.dt
            E_ext, B_ext = self._accumulate_external_fields(X_half, t_half,
                                                            capacitors, dipoles, multipoles)

            # (4) totali
            E_tot = E_pp + E_ext
            B_tot = B_pp + B_ext

            # (5) integrazione Boris
            for i in range(Np):
                x_new, v_new = integrator.step(pos[i], vel[i], q[i], m[i], self.dt, E_tot[i], B_tot[i])
                pos[i], vel[i] = x_new, v_new

            # (6) salva vel e push history
            self.vels[n, :, :] = vel
            hist.push(t + self.dt, pos, vel)

        # (7) risincronizza gli oggetti Particle
        for i, p in enumerate(particles):
            p.pos = pos[i].copy()
            p.vel = vel[i].copy()
