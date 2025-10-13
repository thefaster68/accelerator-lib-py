# ----------- TEST CONDENSATORE -------------

import time
import numpy as np


# --- tuoi moduli ---
import phisics_sim.src.physics as phy
import phisics_sim.src.particle_class as prt
import phisics_sim.src.dipole_class as dp
import phisics_sim.src.simulation_class as sim
import phisics_sim.src.boris_push_class as Integrator
import phisics_sim.src.visuals as vsl


def create_particle(index: int, q: float, m: float,
                    pos0: np.ndarray, vel0: np.ndarray) -> prt.Particle:
    p = prt.Particle()
    p.generate(index, q, m, pos0.astype(float), vel0.astype(float))
    return p


if __name__ == "__main__":
    dt = 1.0e-10
    steps = 6000

    pos0 = np.array([0.0, 0.0, 0.0])
    vel0 = np.array([1.0e0, 0.0, 0.0])
    p = create_particle(0, phy.e, phy.mp, pos0, vel0)
    p_minus = create_particle(1, phy.e, phy.mp, pos0 + np.array([0.0, 1e-6, 0.0]), vel0)
    particles = [p]

    Np = len(particles)

    pos = np.array([0.0, 1e-6, 0.0])
    m = np.array([1.0e-6, 1.0e-6, 0.0])

    d = dp.Dipole(pos, m)

    dipoles = [d]

    integ = Integrator.Boris_push()
    sim = sim.Simulation(dt, steps, Np)

    hist_len = steps
    R_max_ret = 1.0

    t0 = time.time()

    sim.sim(particles=particles, capacitors=[], dipoles=dipoles, multipoles=[], hist_len=hist_len,
            R_max=R_max_ret, integrator=integ, PRINT_FORCES=False, PRINT_EVERY=1)

    t1 = time.time()

    print("Tempo di simulazione: ", t1 - t0, "\t Tempo fisico: ", dt * steps, "\t", "delta = ",
          (t1 - t0) / (dt * steps))

    # ----------------- NUOVO: calcolo vettorializzato delle norme e plot -----------------
    # vels: shape attesa (steps, Np, 3). Convertiamo e otteniamo (Np, steps).
    vels_arr = np.asarray(sim.vels)                             # (steps, Np, 3)
    if vels_arr.ndim != 3 or vels_arr.shape[2] != 3:
        raise RuntimeError(f"Shape inaspettata per sim.vels: {vels_arr.shape} (atteso: (steps, Np, 3))")
    vel_norm = np.linalg.norm(vels_arr, axis=2).T              # -> (Np, steps)

    # Etichette opzionali
    labels = [f"p{i}" for i in range(Np)]

    # Plot in scala lineare (puoi mettere yscale='log' se ti serve)
    vsl.plot_speed_norms(vel_norm, dt, labels=labels, show_c=False,
                     yscale="linear", title="Norme delle velocità — test dipolo")

    # ----------------- Plot delle proiezioni (come prima) -----------------
    vsl.plot_proiezioni(
        sim.POS,
        title_suffix="— dipolo",
        Np_expected=Np, steps_expected=steps
    )

    vsl.plot_3d(POS=sim.POS, Np_expected=Np, steps_expected=steps)
