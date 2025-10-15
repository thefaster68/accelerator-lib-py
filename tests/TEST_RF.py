# ----------- TEST CONDENSATORE -------------

import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- tuoi moduli ---
import phisics_sim.src.physics as phy
import phisics_sim.src.particle_class as prt
import phisics_sim.src.simulation_class as sim
import phisics_sim.src.boris_push_class as Integrator
import phisics_sim.src.rf_cavity_class as rf
import phisics_sim.src.visuals as vsl


def create_particle(index: int, q: float, m: float,
                    pos0: np.ndarray, vel0: np.ndarray) -> prt.Particle:
    p = prt.Particle()
    p.generate(index, q, m, pos0.astype(float), vel0.astype(float))
    return p



# ====================== utility per v(x) ======================
def compute_vel_norm_matrix(POS: np.ndarray, dt: float, vels_arr: np.ndarray | None) -> np.ndarray:
    """
    Restituisce vel_norm con shape (Np, steps).
    - Se vels_arr è fornito ed ha shape (steps, Np, 3), usa quello.
    - Altrimenti stima v con differenze finite da POS.
    """
    if vels_arr is not None:
        vels_arr = np.asarray(vels_arr)
        if vels_arr.ndim == 3 and vels_arr.shape[2] == 3:
            return np.linalg.norm(vels_arr, axis=2).T  # (Np, steps)

    # fallback: differenze finite da POS (Np, steps, 3)
    assert POS.ndim == 3 and POS.shape[2] == 3, f"POS shape inattesa: {POS.shape}"
    Np, steps, _ = POS.shape
    V = np.empty((Np, steps, 3), dtype=float)
    if steps == 1:
        V[:] = 0.0
    else:
        V[:, 0, :]       = (POS[:, 1, :] - POS[:, 0, :]) / dt
        V[:, -1, :]      = (POS[:, -1, :] - POS[:, -2, :]) / dt
        V[:, 1:-1, :]    = (POS[:, 2:, :] - POS[:, :-2, :]) / (2.0 * dt)
    return np.linalg.norm(V, axis=2)  # (Np, steps)

# =====================================================================


if __name__ == "__main__":
    dt = 1.0e-8
    steps = 150000

    pos0 = np.array([0.0, 0.0, 0.0])
    vel0 = np.array([0.0, 0.0, 0.0])
    p = create_particle(0, phy.e, phy.mp, pos0, vel0)
    p_minus = create_particle(1, phy.e, phy.mp, pos0 + np.array([1e-1, 1e-4, 1e-4]), vel0)
    particles = [p, p_minus]

    Np = len(particles)

    E0 = 1.0
    d = 3.0
    R = 0.2
    axis = np.array([1, 0, 0])
    posix = np.array([0, 0, 0])
    f = 5e3
    phase = 20  # gradi
    delta = 0.05 * d
    rf1 = rf.RF_cavity(E0, d, R, axis, posix, 2*np.pi*f, np.deg2rad(phase), delta)

    capacitors = [rf1]

    integ = Integrator.Boris_push()
    sim = sim.Simulation(dt, steps, Np)

    hist_len = 2
    R_max_ret = 0.0

    t0 = time.time()

    sim.sim(particles=particles, capacitors=capacitors, dipoles=[], multipoles=[], hist_len=hist_len,
            R_max=R_max_ret, integrator=integ, PRINT_FORCES=False, PRINT_EVERY=1)

    t1 = time.time()


    print("Tempo di simulazione: ", t1-t0, "\t Tempo fisico: ", dt*steps, "\t", "delta = ", (t1-t0)/(dt*steps))

    # ----------------- Calcolo robusto delle norme e plot -----------------
    # Provo ad usare sim.vels se esiste (steps, Np, 3); se non esiste, uso differenze da POS.
    vels_arr = getattr(sim, "vels", None)  # può non esserci nella tua Simulation
    vel_norm = compute_vel_norm_matrix(sim.POS, dt, vels_arr)  # -> (Np, steps)

    labels = [f"p{i}" for i in range(Np)]

    # ||v||(t)
    vsl.plot_speed_norms(vel_norm, dt, labels=labels, show_c=False,
                     yscale="linear", title="Norme delle velocità — test rf cavity")

    # ----------------- ||v|| vs posizione -----------------
    x_cap_span = (posix[0] - 0.5*d, posix[0] + 0.5*d)
    vsl.plot_speed_vs_position(
        vel_norm, sim.POS, axis="x", labels=labels,
        x_cap=x_cap_span, x_quad=None,
        title=r"||v|| vs x — test rf cavity"
    )

    # ----------------- Plot delle proiezioni (come prima) -----------------
    vsl.plot_proiezioni(
        sim.POS,
        x_cap=x_cap_span,
        title_suffix="— rf cavity",
        Np_expected=Np, steps_expected=steps
    )
