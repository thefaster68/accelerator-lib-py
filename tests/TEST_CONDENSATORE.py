# ----------- TEST CONDENSATORE -------------

import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- tuoi moduli ---
import phisics_sim.src.physics as phy
import phisics_sim.src.particle_class as prt
import phisics_sim.src.capacitor_class as cap
import phisics_sim.src.simulation_class as sim
import phisics_sim.src.boris_push_class as Integrator


def create_particle(index: int, q: float, m: float,
                    pos0: np.ndarray, vel0: np.ndarray) -> prt.Particle:
    p = prt.Particle()
    p.generate(index, q, m, pos0.astype(float), vel0.astype(float))
    return p


def _draw_region_box(ax, x0, x1, y_extent, z_extent, color='C1', alpha=0.08):
    """Box trasparente per evidenziare regioni (condensatore, quadrupolo)."""
    X = [x0, x1]
    Y = [-y_extent, +y_extent]
    Z = [-z_extent, +z_extent]
    verts = [
        [(X[0], Y[0], Z[0]), (X[1], Y[0], Z[0]), (X[1], Y[1], Z[0]), (X[0], Y[1], Z[0])],
        [(X[0], Y[0], Z[1]), (X[1], Y[0], Z[1]), (X[1], Y[1], Z[1]), (X[0], Y[1], Z[1])],
        [(X[0], Y[0], Z[0]), (X[1], Y[0], Z[0]), (X[1], Y[0], Z[1]), (X[0], Y[0], Z[1])],
        [(X[0], Y[1], Z[0]), (X[1], Y[1], Z[0]), (X[1], Y[1], Z[1]), (X[0], Y[1], Z[1])],
        [(X[0], Y[0], Z[0]), (X[0], Y[1], Z[0]), (X[0], Y[1], Z[1]), (X[0], Y[0], Z[1])],
        [(X[1], Y[0], Z[0]), (X[1], Y[1], Z[0]), (X[1], Y[1], Z[1]), (X[1], Y[0], Z[1])],
    ]
    ax.add_collection3d(Poly3DCollection(verts, facecolors=color, alpha=alpha, linewidths=0.2))


def plot_proiezioni(POS: np.ndarray, x_cap=None, x_quad=None, title_suffix="", Np_expected=None, steps_expected=None):
    """POS shape attesa: (Np, steps, 3). Plotta x–y, x–z, y–z + bande regioni."""
    assert POS.ndim == 3 and POS.shape[2] == 3, f"POS shape inattesa: {POS.shape}"
    Np, steps, _ = POS.shape
    if Np_expected is not None and steps_expected is not None:
        if Np != Np_expected or steps != steps_expected:
            print(f"[WARN] POS shape={POS.shape} ma ci si aspettava (Np={Np_expected}, steps={steps_expected})")

    fig, axs = plt.subplots(1, 3, figsize=(14, 5))

    for i in range(Np):
        x = POS[i, :, 0]; y = POS[i, :, 1]; z = POS[i, :, 2]
        axs[0].plot(x, y, lw=0.9)
        axs[1].plot(x, z, lw=0.9)
        axs[2].plot(y, z, lw=0.9)

    for ax in axs[:2]:
        if x_cap is not None:
            ax.axvspan(x_cap[0], x_cap[1], color='C0', alpha=0.10, label="Capacitore")
        if x_quad is not None:
            ax.axvspan(x_quad[0], x_quad[1], color='C1', alpha=0.08, label="Quadrupolo")

    axs[0].set_xlabel("x [m]"); axs[0].set_ylabel("y [m]"); axs[0].set_title(f"x–y {title_suffix}")
    axs[1].set_xlabel("x [m]"); axs[1].set_ylabel("z [m]"); axs[1].set_title(f"x–z {title_suffix}")
    axs[2].set_xlabel("y [m]"); axs[2].set_ylabel("z [m]"); axs[2].set_title(f"y–z {title_suffix}")

    handles, labels = [], []
    if x_cap is not None:  handles += [plt.Line2D([0],[0], color='C0', lw=6, alpha=0.3)]; labels += ["Capacitore"]
    if x_quad is not None: handles += [plt.Line2D([0],[0], color='C1', lw=6, alpha=0.3)]; labels += ["Quadrupolo"]
    if handles:
        axs[0].legend(handles, labels, loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_3d(POS: np.ndarray, x_cap=None, x_quad=None, title="Traiettorie 3D", Np_expected=None, steps_expected=None):
    """Plot 3D con box trasparente su regioni utili."""
    assert POS.ndim == 3 and POS.shape[2] == 3, f"POS shape inattesa: {POS.shape}"
    Np, steps, _ = POS.shape
    yE = np.nanmax(np.abs(POS[:, :, 1]))*1.1 + 1e-6
    zE = np.nanmax(np.abs(POS[:, :, 2]))*1.1 + 1e-6

    fig = plt.figure(figsize=(7.0, 5.5))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(Np):
        ax.plot3D(POS[i, :, 0], POS[i, :, 1], POS[i, :, 2], lw=0.9)

    if x_cap is not None:
        _draw_region_box(ax, x_cap[0], x_cap[1], yE, zE, color='C0', alpha=0.10)
    if x_quad is not None:
        _draw_region_box(ax, x_quad[0], x_quad[1], yE, zE, color='C1', alpha=0.08)

    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_speed_norms(vel_norm: np.ndarray, dt: float, labels=None,
                     show_c: bool = False, yscale: str = "linear", title: str = "Norme delle velocità"):
    """Plotta \|\|v\|\|(t) per ogni particella."""
    assert vel_norm.ndim == 2, f"atteso (Np, steps), ottenuto {vel_norm.shape}"
    Np, steps = vel_norm.shape

    t = np.arange(steps) * dt
    if steps == 0:
        raise ValueError("vel_norm ha zero steps: nulla da plottare.")
    t_end = t[-1] if steps > 1 else dt

    if t_end < 1e-9:
        scale, unit = 1e12, "ps"
    elif t_end < 1e-6:
        scale, unit = 1e9, "ns"
    elif t_end < 1e-3:
        scale, unit = 1e6, "µs"
    elif t_end < 1.0:
        scale, unit = 1e3, "ms"
    else:
        scale, unit = 1.0, "s"

    t_plot = t * scale
    labels = [f"p{i}" for i in range(Np)] if labels is None else labels

    plt.figure(figsize=(8.5, 4.8))
    for i in range(Np):
        plt.plot(t_plot, vel_norm[i], lw=1.1, label=labels[i])

    if show_c:
        plt.axhline(phy.c, ls="--", lw=1.0, alpha=0.7, label="c")

    plt.yscale(yscale)
    plt.xlabel(f"t [{unit}]")
    plt.ylabel("||v|| [m/s]")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.show()


# ====================== NUOVO: utility per v(x) ======================
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


def plot_speed_vs_position(vel_norm: np.ndarray, POS: np.ndarray, *,
                           axis: str = "x", labels=None,
                           x_cap=None, x_quad=None,
                           title: str = r"||v|| vs posizione"):
    """
    Plotta ||v|| in funzione della coordinata scelta (x,y,z).
    Se axis='x', ombreggia le regioni di condensatore/quadrupolo se fornite.
    """
    idx_map = {"x": 0, "y": 1, "z": 2}
    assert axis in idx_map, "axis deve essere 'x', 'y' oppure 'z'"
    ax_idx = idx_map[axis]

    Np, steps = vel_norm.shape
    assert POS.shape[:2] == (Np, steps), "POS e vel_norm non compatibili"

    labels = [f"p{i}" for i in range(Np)] if labels is None else labels

    plt.figure(figsize=(8.5, 4.8))
    for i in range(Np):
        coord = POS[i, :, ax_idx]
        plt.plot(coord, vel_norm[i], lw=1.0, label=labels[i])

    if axis == "x":
        if x_cap is not None:
            plt.axvspan(x_cap[0], x_cap[1], color="C0", alpha=0.10, label="Capacitore")
        if x_quad is not None:
            plt.axvspan(x_quad[0], x_quad[1], color="C1", alpha=0.08, label="Quadrupolo")

    plt.xlabel(f"{axis} [m]")
    plt.ylabel("||v|| [m/s]")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.show()
# =====================================================================


if __name__ == "__main__":
    dt = 1.0e-10
    steps = 5_00

    pos0 = np.array([0.0, 0.0, 0.0])
    vel0 = np.array([0.0, 0.0, 0.0])
    p = create_particle(0, phy.e, phy.mp, pos0, vel0)
    p_minus = create_particle(1, phy.e, phy.mp, pos0 + np.array([0.0, 1e-6, 0.0]), vel0)
    particles = [p]

    Np = len(particles)

    sigma = 1.0e-6
    d = 1.0e-3
    base = 1.0e-1
    height = 1.0e-1
    normal_cap = np.array([1.0, 0.0, 0.0])
    epsilon_r = 1.0
    pos = np.array([0.0, 0.0, 0.0])
    c = cap.Capacitor(
        sigma, d, base, height, normal_cap, epsilon_r, pos,
        edge_delta=0.05 * d,  # smussatura lungo la normale (esempio: 5% dello spessore)
        lateral_delta=0.0  # opzionale: smussatura sui bordi laterali (0 = off)
    )

    capacitors = [c]

    integ = Integrator.Boris_push()
    sim = sim.Simulation(dt, steps, Np)

    hist_len = steps
    R_max_ret = 1.0

    t0 = time.time()

    sim.sim(particles=particles, capacitors=capacitors, dipoles=[], multipoles=[], hist_len=hist_len,
            R_max=R_max_ret, integrator=integ, PRINT_FORCES=False, PRINT_EVERY=1)

    t1 = time.time()


    print("Tempo di simulazione: ", t1-t0, "\t Tempo fisico: ", dt*steps, "\t", "delta = ", (t1-t0)/(dt*steps))

    # ----------------- NUOVO: calcolo robusto delle norme e plot -----------------
    # Provo ad usare sim.vels se esiste (steps, Np, 3); se non esiste, uso differenze da POS.
    vels_arr = getattr(sim, "vels", None)  # può non esserci nella tua Simulation
    vel_norm = compute_vel_norm_matrix(sim.POS, dt, vels_arr)  # -> (Np, steps)

    labels = [f"p{i}" for i in range(Np)]

    # ||v||(t)
    plot_speed_norms(vel_norm, dt, labels=labels, show_c=False,
                     yscale="linear", title="Norme delle velocità — test condensatore")

    # ----------------- NUOVO: ||v|| vs posizione -----------------
    x_cap_span = (pos[0] - 0.5*d, pos[0] + 0.5*d)
    plot_speed_vs_position(
        vel_norm, sim.POS, axis="x", labels=labels,
        x_cap=x_cap_span, x_quad=None,
        title=r"||v|| vs x — test condensatore"
    )

    # ----------------- Plot delle proiezioni (come prima) -----------------
    plot_proiezioni(
        sim.POS,
        x_cap=x_cap_span,
        title_suffix="— condensatore",
        Np_expected=Np, steps_expected=steps
    )
