# ----------- TEST QUADRUPOLI -------------

import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- tuoi moduli ---
import phisics_sim.src.physics as phy
import phisics_sim.src.particle_class as prt
import phisics_sim.src.quadrupole_magnet_class as qm
import phisics_sim.src.dipole_magnet_class as dm
import phisics_sim.src.simulation_class as sim
import phisics_sim.src.boris_push_class as Integrator
import phisics_sim.src.visuals as vsl


def create_particle(index: int, q: float, m: float,
                    pos0: np.ndarray, vel0: np.ndarray) -> prt.Particle:
    p = prt.Particle()
    p.generate(index, q, m, pos0.astype(float), vel0.astype(float))
    return p


if __name__ == "__main__":
    dt = 1.0e-9
    steps = 6000

    pos0 = np.array([0.0, 0.001, 0.001])
    vel0 = np.array([1.0e6, 0.0, 0.0])
    p = create_particle(0, phy.e, phy.mp, pos0, vel0)
    p_minus = create_particle(1, phy.e, phy.mp, pos0 + np.array([0.0, 1e-6, 0.0]), vel0)
    particles = [p]

    #ps.particle_grid_spawner(particles, 3, 3, 1e-8, 1e-6, phy.e, phy.mp, 1e6)

    Np = len(particles)

    R = 0.10
    L = 0.20
    pos = np.array([0.3, 0.0, 0.0])
    n_hat = np.array([1.0, 0.0, 0.0])
    roll = 0
    g = qm._gradient_from_focal(phy.e, phy.mp, vel0, L, 5.0)
    print(g)

    Q = qm.Quadrupole(R, L, pos, n_hat, g, roll, +1)
    Q1 = qm.Quadrupole(R, L, pos + np.array([0.3, 0.0, 0.0]), n_hat, g, roll, -1)

    D = dm.Dipole(0.001, R, L, pos, n_hat, roll)

    multipoles = [Q]

    x_quad = x_quad_span = (Q.pos[0] - 0.5*L, Q.pos[0] + 0.5*L)

    integ = Integrator.Boris_push()
    sim = sim.Simulation(dt, steps, Np)

    hist_len = 2
    R_max_ret = 0.0

    t0 = time.time()

    sim.sim(particles=particles, capacitors=[], dipoles=[], multipoles=multipoles, hist_len=hist_len,
            R_max=R_max_ret, integrator=integ, PRINT_FORCES=False, PRINT_EVERY=1)

    t1 = time.time()

    print("Tempo di simulazione: ", t1 - t0, "\t Tempo fisico: ", dt * steps, "\t", "delta = ",
          (t1 - t0) / (dt * steps))

    # ----------------- Calcolo vettorializzato delle norme e plot -----------------
    # vels: shape attesa (steps, Np, 3). Convertiamo e otteniamo (Np, steps).
    vels_arr = np.asarray(sim.vels)                             # (steps, Np, 3)
    if vels_arr.ndim != 3 or vels_arr.shape[2] != 3:
        raise RuntimeError(f"Shape inaspettata per sim.vels: {vels_arr.shape} (atteso: (steps, Np, 3))")
    vel_norm = np.linalg.norm(vels_arr, axis=2).T              # -> (Np, steps)

    # Etichette opzionali
    labels = [f"p{i}" for i in range(Np)]

    # Plot in scala lineare (puoi mettere yscale='log' se ti serve)
    vsl.plot_speed_norms(vel_norm, dt, labels=labels, show_c=False,
                     yscale="linear", title="Norme delle velocità — test quadrupolo")

    # ----------------- Plot delle proiezioni -----------------
    vsl.plot_proiezioni(
        sim.POS, multipoles=multipoles,
        title_suffix="— dipolo",
        Np_expected=Np, steps_expected=steps
    )

    vsl.plot_3d(POS=sim.POS, Np_expected=Np, steps_expected=steps)
