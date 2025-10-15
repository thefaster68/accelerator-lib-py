# ----------- TEST ACCELERATORE LINEARE -------------
import time
import numpy as np

import phisics_sim.src.physics as phy
import phisics_sim.src.particle_class as prt
import phisics_sim.src.quadrupole_magnet_class as qm
import phisics_sim.src.capacitor_class as cap
import phisics_sim.src.simulation_class as sim
import phisics_sim.src.boris_push_class as Integrator
import phisics_sim.src.particle_spawners as ps
import phisics_sim.src.visuals as vsl


def create_particle(index: int, q: float, m: float,
                    pos0: np.ndarray, vel0: np.ndarray) -> prt.Particle:
    p = prt.Particle()
    p.generate(index, q, m, pos0.astype(float), vel0.astype(float))
    return p


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


if __name__ == "__main__":
    # --- parametri simulazione ---
    dt = 1.0e-9  # ok per v~1e6 m/s e k moderati
    steps = 1350

    # --- fascio ---
    particles = []
    ps.particle_grid_spawner(particles, 1, 1, 1e-5, 1e-5, phy.e, phy.mp, 1e6, jitter_rel=0)

    ps.particle_circle_spawner(particles, 8, 1e-3, np.zeros(3), np.array([1, 0, 0]), phy.e, phy.mp, 1e6)
    ps.particle_circle_spawner(particles, 8, 7.5e-4, np.zeros(3), np.array([1, 0, 0]), phy.e, phy.mp, 1e6)
    ps.particle_circle_spawner(particles, 8, 5e-4, np.zeros(3), np.array([1, 0, 0]), phy.e, phy.mp, 1e6)
    ps.particle_circle_spawner(particles, 8, 1e-4, np.zeros(3), np.array([1, 0, 0]), phy.e, phy.mp, 1e6)
    Np = len(particles)

    print("[info] N particelle: ", Np)
    print("[info] Energia iniziale (J): ", 0.5*phy.mp*np.power(1e6, 2))

    # --- sezione 1 ---
    # --- doppietto ---
    R = 0.10  # raggio foro (apertura)
    L = 0.20  # lunghezza di ciascun quadrupolo
    S = 0.10  # drift tra i due (edge-to-edge = S)
    pos = np.array([0.15, 0.0, 0.0])  # centro del primo quad
    n_hat = np.array([1.0, 0.0, 0.0])
    roll1 = 0.0
    roll2 = np.pi / 2  # ruoto il secondo di 90° (alternativa: stessa rotazione ma polarity=-1)
    f_target = 0.50 * 1.5  # 0.30 # m

    # -- calcolo gradienti magnetici --
    v01 = np.array([1.0e6, 0.0, 0.0])
    g1 = qm._gradient_from_focal(phy.e, phy.mp, v01, L, f_target)

    v02 = np.array([1e7, 0.0, 0.0])
    g2 = qm._gradient_from_focal(phy.e, phy.mp, v02, L, f_target)

    v03 = np.array([5e7, 0.0, 0.0])
    g3 = qm._gradient_from_focal(phy.e, phy.mp, v03, L, f_target) * 0.4 # * 0.8

    v04 = np.array([1e8, 0.0, 0.0])
    g4 = qm._gradient_from_focal(phy.e, phy.mp, v04, L, f_target) * 0.4 # * 0.8

    # --- lenti magnetiche ---

    QF1 = qm.Quadrupole(R, L, pos, n_hat, g1, roll1, polarity=+1)  # F in un piano, D nell'altro
    QD1 = qm.Quadrupole(R, L, pos + np.array([L + S, 0, 0]), n_hat, g1, roll2, polarity=+1)  # ruotato di 90°

    # --- condensatori ---
    off = 0.05
    sigma1 = phy.mp * phy.epsilon_0 / (2 * phy.e * 3) * (np.pow(1e7, 2) - np.pow(1e6, 2))
    d = 3.0
    l, h = 0.10, 0.10
    e_r = 1.0
    pos_cap = np.array([d/2 + off + L/2, 0, 0]) + pos + np.array([L + S, 0, 0])
    edge_delta = 0.05 * d

    C1 = cap.Capacitor(sigma1, d, l, h, n_hat, e_r, pos_cap, edge_delta)

    # --- sezione 2 ---
    pos2 = C1.pos + np.array([d/2 + off + L/2, 0, 0])

    QF2 = qm.Quadrupole(R, L, pos2, n_hat, g2, roll2, polarity=+1)  # F in un piano, D nell'altro
    QD2 = qm.Quadrupole(R, L, pos2 + np.array([L + S, 0, 0]), n_hat, g2, roll1, polarity=+1)  # ruotato di 90°

    pos_cap2 = QD2.pos + np.array([L / 2 + off + d / 2, 0, 0])
    sigma2 = phy.mp * phy.epsilon_0 / (2 * phy.e * d) * (np.pow(5e7, 2) - np.pow(1e7, 2))
    C2 = cap.Capacitor(sigma2, d, l, h, n_hat, e_r, pos_cap2, edge_delta)

    # --- sezione 3 ---
    pos3 = C2.pos + np.array([d/2 + off + L/2, 0, 0])

    QF3 = qm.Quadrupole(R, L, pos3, n_hat, g3, roll1, polarity=+1)  # F in un piano, D nell'altro
    QD3 = qm.Quadrupole(R, L, pos3 + np.array([L + S, 0, 0]), n_hat, g3, roll2, polarity=+1)  # ruotato di 90°

    pos_cap3 = QD3.pos + np.array([L / 2 + off + d / 2, 0, 0])
    sigma3 = phy.mp * phy.epsilon_0 / (2 * phy.e * d) * (np.pow(1e8, 2) - np.pow(5e7, 2))
    C3 = cap.Capacitor(sigma3, d, l, h, n_hat, e_r, pos_cap3, edge_delta)

    # --- sezione 3 ---
    pos4 = C3.pos + np.array([d / 2 + off + L / 2, 0, 0])

    QF4 = qm.Quadrupole(R, L, pos4, n_hat, g4, roll2, polarity=+1)  # F in un piano, D nell'altro
    QD4 = qm.Quadrupole(R, L, pos4 + np.array([L + S, 0, 0]), n_hat, g4, roll1, polarity=+1)  # ruotato di 90°

    pos_cap4 = QD4.pos + np.array([L / 2 + off + d / 2, 0, 0])
    sigma4 = phy.mp * phy.epsilon_0 / (2 * phy.e * d) * (np.pow(2e8, 2) - np.pow(1e8, 2))
    C4 = cap.Capacitor(sigma4, d, l, h, n_hat, e_r, pos_cap4, edge_delta)

    # --- liste componenti ---
    capacitors = [C1, C2, C3, C4]
    multipoles = [QF1, QD1, QF2, QD2, QF3, QD3, QF4, QD4]

    # --- print dati ---
    print("\n")
    print("[INFO] sigma1 calcolato = ", sigma1)
    print("[INFO] sigma2 calcolato = ", sigma2)
    print("[INFO] sigma3 calcolato = ", sigma3)
    print("[INFO] sigma4 calcolato = ", sigma4)
    print("\n")
    print("[INFO] G1 calcolato =", g1, "T/m")
    print("[INFO] G2 calcolato =", g2, "T/m")
    print("[INFO] G3 calcolato =", g3, "T/m")
    print("[INFO] G4 calcolato =", g4, "T/m")
    print("\n")

    # --- simulazione ---
    integ = Integrator.Boris_push()
    runner = sim.Simulation(dt, steps, Np)

    hist_len = steps    # = 2 se vuoi eliminare le interazioni PP
    R_max_ret = 1.0     # = 0.0 se vuoi eliminare le interazioni PP

    print("expected time: ", 220.75 * Np * Np / (Np + 448.70), " s")
    t0 = time.time()
    runner.sim(particles=particles, capacitors=capacitors, dipoles=[],
               multipoles=multipoles, hist_len=hist_len, R_max=R_max_ret,
               integrator=integ, PRINT_FORCES=False, PRINT_EVERY=0)
    t1 = time.time()

    print("elaspsed time: ", t1-t0, " s")

    #for i in range(Np): print(runner.POS[i][0], "\n")

    # --- plot ---
    vsl.plot_proiezioni(runner.POS, quad=multipoles, cap=capacitors,
                    title_suffix="— linac", Np_expected=Np, steps_expected=steps)
    vsl.plot_3d(runner.POS, quad=multipoles, cap=capacitors,
            title="Traiettorie 3D —  linac",
            Np_expected=Np, steps_expected=steps)

    vels_arr = getattr(sim, "vels", None)
    vel_norm = compute_vel_norm_matrix(runner.POS, dt, vels_arr)  # -> (Np, steps)

    vsl.plot_speed_norms(vel_norm, dt, labels=None, show_c=True,
                     yscale="linear", title="Norme delle velocità")

    # piano: P0 (posizione dello schermo) e normale n_hat
    P0 = np.array([15.0, 0.0, 0.0])  # schermo a x
    n_hat = np.array([1.0, 0.0, 0.0])  # normale verso +x

    H, extent = vsl.plane_crossings_density(
        runner.POS, P0, n_hat,
        nbins=(15, 15),
        half_extent=(2e-4, 2e-4),  # finestra visualizzata
        normalize=False  # True -> densità areale [counts/m^2]
    )
    vsl.plot_plane_density(H, extent, title="Schermo a x=15.0 m", cmap="magma", cbar_label="counts")
