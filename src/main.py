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
# se hai uno "spawner" lo puoi usare al posto della create_particle() qui sotto

# ===========================
# Utility fisiche e plotting
# ===========================
def v_from_kinetic_E_MeV_proton(E_MeV: float):
    """Restituisce (v, gamma, Brho) per un protone di energia cinetica E_MeV."""
    m = phy.mp
    q = phy.e
    E = E_MeV * 1.0e6 * phy.e
    gamma = 1.0 + E / (m * phy.c**2)
    beta  = np.sqrt(1.0 - 1.0/gamma**2)
    v     = beta * phy.c
    p     = gamma * m * v
    Brho  = p / q
    return v, gamma, Brho

def thin_lens_G_from_f(Brho: float, L: float, f_target: float) -> float:
    """G = Brho / (f*L)"""
    return Brho / (f_target * L)

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
    # Sanity opzionale
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

    # legende sobrie
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

# ===========================
# Configurazione principale
# ===========================
if __name__ == "__main__":

    # ------------ SCELTE UTENTE ------------
    # Proton beam energy [MeV]
    E_MeV       = 1.0          # cambia in 10.0 per 10 MeV
    use_skew    = False         # True => quadrupolo ruotato di 45°
    use_cap     = False        # True => aggiungi un "kicker" elettrico leggero
    f_target    = 0.50         # focale desiderata [m] per tarare G
    flight_L    = 0.60         # lunghezza totale di volo da simulare [m]

    # Geometria magnete Halbach (quadrupolo)
    R_bore      = 1.0e-3       # raggio foro (m)
    L_mag       = 0.20         # lunghezza magnete (m)
    x0_mag      = 0.30         # posizione del centro magnete (m)
    Nseg        = 96           # segmenti della corona (>=64 consigliato)
    tilt        = (np.pi/4) if use_skew else 0.0
    fringe_delta= 0.001         # frangia liscia (~1 mm)

    # Kicker elettrico (se attivo)
    Ekick       = 1.0e4        # [V/m] ~10 kV/m
    L_cap       = 0.02         # [m] 2 cm utili
    x0_cap      = 0.05         # centro del condensatore (m)

    # Offset iniziale per eccitare F/D
    y0          = 0.2e-5       # 0.2 mm
    z0          = 0.0

    # ------------ DERIVATI FISICI ------------
    v_par, gamma, Brho = v_from_kinetic_E_MeV_proton(E_MeV)
    q, m = +phy.e, phy.mp

    # gradiente dal focale target
    G_target = thin_lens_G_from_f(Brho, L_mag, f_target) * 10.0

    # time step e steps per coprire flight_L
    # (~2000-5000 step lungo il volo: dt adattato alla velocità)
    dt    = 1.0e-11 if E_MeV <= 5.0 else 5.0e-12
    steps = int(flight_L / (v_par * dt)) + 1

    print(f"[INFO] Prot.: E={E_MeV:.2f} MeV  beta={v_par/phy.c:.4f}  gamma={gamma:.5f}  Bρ={Brho:.3e} T·m")
    print(f"[INFO] Target: f={f_target:.3f} m  =>  G={G_target:.3f} T/m  (L={L_mag:.2f} m)")
    print(f"[INFO] Integrazione: dt={dt:.2e} s, steps={steps}, flight≈{flight_L:.2f} m")

    # ------------ OGGETTI DI CAMPO ------------
    multipoles = []

    quad = mpc.HalbachMultipoleRing(
        order=2, R=R_bore, L=L_mag,
        pos=np.array([x0_mag, 0.0, 0.0], dtype=float),
        axis=np.array([1.0, 0.0, 0.0], dtype=float),
        m0=1e-3, Nseg=Nseg, tilt=tilt, fringe_delta=fringe_delta
    )
    # calibrazione del gradiente vicino all'asse
    G_meas = quad.calibrate_to_G(G_target=G_target, dr=1.0e-4, iters=8)
    print(f"[INFO] Calibrazione quadrupolo: G_meas={G_meas:.3f} T/m (target={G_target:.3f})")
    multipoles.append(quad)

    # Condensatore (opzionale: piccolo kicker per introdurre v_perp controllata)
    capacitors = []
    if use_cap:
        # piastre perpendicolari a y (campo lungo y) estese in z e x con finestra top-hat:
        c = cap.Capacitor()
        # NB: usa i parametri secondo la tua classe; se richiede generate(...):
        # c.generate(V=..., plate_dist=..., center=..., normal=..., L=...) ecc.
        # Nel dubbio, usa il metodo semplificato "box" se presente.
        # Qui: costruiamo un condensatore fittizio uniforme su [x0_cap-L_cap/2, x0_cap+L_cap/2]
        # Se la tua classe ha interfaccia diversa, adatta questi 4 numeri:
        c.center = np.array([x0_cap, 0.0, 0.0], float)
        c.axis   = np.array([1.0, 0.0, 0.0], float)
        c.L      = L_cap
        c.E0     = Ekick
        capacitors.append(c)

    # ------------ PARTICELLE ------------
    particles = []
    p0 = np.array([0.0, y0, z0], dtype=float)
    v0 = np.array([v_par, 0.0, 0.0], dtype=float)
    particles.append(create_particle(0, q, m, p0, v0))
    Np = len(particles)

    # ------------ SIMULAZIONE ------------
    integ = Integrator.Boris_push()
    S = sim.Simulation(dt, steps, Np)

    # HistoryBuffer / opzioni LW (se la tua Simulation li richiede)
    hist_len   = steps                     # conserva tutta la storia
    R_max_ret  = 1.0                       # raggio per ricerca ritardo (m)

    t0 = time.time()
    # La firma di sim(...) può variare tra le tue versioni. Proviamo in ordine.
    ran = False
    try:
        S.sim(particles=particles, capacitors=capacitors, dipoles=[], multipoles=multipoles, hist_len=hist_len,
              R_max=R_max_ret, integrator=integ, PRINT_FORCES=False, PRINT_EVERY=0)
        ran = True
    except TypeError:
        try:
            S.sim(particles=particles, capacitors=capacitors, dipoles=[], multipoles=multipoles, hist_len=hist_len,
                  R_max=R_max_ret, integrator=integ, PRINT_FORCES=False, PRINT_EVERY=0)
            ran = True
        except TypeError:
            S.sim(particles=particles, capacitors=capacitors, dipoles=[], multipoles=multipoles, hist_len=hist_len,
                  R_max=R_max_ret, integrator=integ, PRINT_FORCES=False, PRINT_EVERY=0)
            ran = True
    t1 = time.time()

    if not ran:
        raise RuntimeError("Impossibile chiamare Simulation.sim con le firme provate.")

    print(f"[DONE] Simulazione completata in {t1 - t0:.3f} s  (Np={Np}, steps={steps})")

    # ------------ PLOT ------------
    x_cap_span  = (x0_cap - 0.5*L_cap, x0_cap + 0.5*L_cap) if use_cap else None
    x_quad_span = (x0_mag - 0.5*L_mag, x0_mag + 0.5*L_mag)

    plot_proiezioni(
        S.POS,
        x_cap=x_cap_span, x_quad=x_quad_span,
        title_suffix="— condensatore + Halbach" if use_cap else "— Halbach",
        Np_expected=Np, steps_expected=steps
    )

    plot_3d(
        S.POS,
        x_cap=x_cap_span, x_quad=x_quad_span,
        title="Traiettorie 3D — condensatore + Halbach" if use_cap else "Traiettorie 3D — Halbach",
        Np_expected=Np, steps_expected=steps
    )
