import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


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


def plot_proiezioni(POS: np.ndarray, cap=None, quad=None, title_suffix="", Np_expected=None, steps_expected=None):
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
        if cap is not None:
            for c in cap:
                x_cap = (c.pos[0] - 0.5 * c.d, c.pos[0] + 0.5 * c.d)
                ax.axvspan(x_cap[0], x_cap[1], color='C0', alpha=0.10, label="Capacitore")
        if quad is not None:
            for q in quad:
                x_quad = (q.pos[0] - 0.5 * q.L, q.pos[0] + 0.5 * q.L)
                ax.axvspan(x_quad[0], x_quad[1], color='C1', alpha=0.08, label="Quadrupolo")

    axs[0].set_xlabel("x [m]"); axs[0].set_ylabel("y [m]"); axs[0].set_title(f"x–y {title_suffix}")
    axs[1].set_xlabel("x [m]"); axs[1].set_ylabel("z [m]"); axs[1].set_title(f"x–z {title_suffix}")
    axs[2].set_xlabel("y [m]"); axs[2].set_ylabel("z [m]"); axs[2].set_title(f"y–z {title_suffix}")

    # legende sobrie
    handles, labels = [], []
    if cap is not None:  handles += [plt.Line2D([0],[0], color='C0', lw=6, alpha=0.3)]; labels += ["Capacitore"]
    if quad is not None: handles += [plt.Line2D([0],[0], color='C1', lw=6, alpha=0.3)]; labels += ["Quadrupolo"]
    if handles:
        axs[0].legend(handles, labels, loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.show()



def plot_3d(POS: np.ndarray, cap=None, quad=None, title="Traiettorie 3D", Np_expected=None, steps_expected=None):
    """Plot 3D con box trasparente su regioni utili."""
    assert POS.ndim == 3 and POS.shape[2] == 3, f"POS shape inattesa: {POS.shape}"
    Np, steps, _ = POS.shape
    yE = np.nanmax(np.abs(POS[:, :, 1]))*1.1 + 1e-6
    zE = np.nanmax(np.abs(POS[:, :, 2]))*1.1 + 1e-6

    fig = plt.figure(figsize=(7.0, 5.5))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(Np):
        ax.plot3D(POS[i, :, 0], POS[i, :, 1], POS[i, :, 2], lw=0.9)

    if cap is not None:
        for c in cap:
            x_cap = (c.pos[0] - 0.5 * c.d, c.pos[0] + 0.5 * c.d)
            _draw_region_box(ax, x_cap[0], x_cap[1], yE, zE, color='C0', alpha=0.10)
    if quad is not None:
        for q in quad:
            x_quad = (q.pos[0] - 0.5 * q.L, q.pos[0] + 0.5 * q.L)
            _draw_region_box(ax, x_quad[0], x_quad[1], yE, zE, color='C1', alpha=0.08)

    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
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


def plot_speed_norms(vel_norm: np.ndarray, dt: float, labels=None,
                     show_c: bool = False, yscale: str = "linear", title: str = "Norme delle velocità"):
    """
    Plotta v_i(t) per ogni particella i.
    Parameters
    ----------
    vel_norm : np.ndarray
        Array di shape (Np, steps) con le norme delle velocità già calcolate.
    dt : float
        Time step [s].
    labels : list[str] | None
        Etichette per le particelle (len == Np). Se None, usa "p0", "p1", ...
    show_c : bool
        Se True disegna una riga orizzontale a v = c.
    yscale : {"linear","log"}
        Scala dell'asse y.
    title : str
        Titolo del grafico.
    """
    assert vel_norm.ndim == 2, f"atteso (Np, steps), ottenuto {vel_norm.shape}"
    Np, steps = vel_norm.shape

    # Asse dei tempi con unità "furbe"
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

    # Etichette predefinite
    if labels is None:
        labels = [f"p{i}" for i in range(Np)]
    else:
        assert len(labels) == Np, "labels deve avere lunghezza Np"

    plt.figure(figsize=(8.5, 4.8))
    for i in range(Np):
        plt.plot(t_plot, vel_norm[i], lw=1.1, label=labels[i])

    if show_c:
        try:
            import physics as _phy
            plt.axhline(_phy.c, ls="--", lw=1.0, alpha=0.7, label="c")
        except Exception:
            pass  # se physics non è disponibile qui, ignoro la linea di c

    plt.yscale(yscale)
    plt.xlabel(f"t [{unit}]")
    plt.ylabel("||v|| [m/s]")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.show()



# ---------- util: versore e frame del piano ----------
def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n == 0.0:
        raise ValueError("Direzione nulla")
    return v / n


def _plane_frame(n_hat):
    """Ritorna (u_hat, v_hat, n_hat) ortonormali dato n_hat."""
    n_hat = _unit(n_hat)
    # vettore 'tmp' non parallelo a n_hat
    tmp = np.array([1.0, 0.0, 0.0]) if abs(n_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u_hat = np.cross(n_hat, tmp); u_hat /= np.linalg.norm(u_hat)
    v_hat = np.cross(n_hat, u_hat)
    return u_hat, v_hat, n_hat


# ---------- crossing-based density ----------
def plane_crossings_density(POS, P0, n_hat, nbins=(80, 80), half_extent=(5e-3, 5e-3),
                            normalize=False, return_uv=False, vels=None, dt=None):
    """
    Costruisce la mappa di densità su un piano tramite gli attraversamenti delle traiettorie.

    Parameters
    ----------
    POS : array (Np, steps, 3)
        Traiettorie.
    P0 : array-like (3,)
        Un punto del piano.
    n_hat : array-like (3,)
        Normale al piano (direzione).
    nbins : (nx, ny)
        Numero di bin su u e v.
    half_extent : (Ux, Vy)
        Estensioni massime in metri su u e v (da -Ux a +Ux, -Vy a +Vy).
    normalize : bool
        Se True, normalizza per area del bin (densità areale [counts/m^2]).
    return_uv : bool
        Se True, ritorna anche gli array u,v degli impatti (utile per analisi extra).
    vels : array opzionale (steps, Np, 3) o (Np, steps, 3)
        Se fornito, può essere usato per calcolare direzioni al crossing (non obbligatorio).
    dt : float
        Solo se vuoi derivare direzioni dagli spostamenti; non necessario per la densità.

    Returns
    -------
    H : 2D array (ny, nx)
        Mappa di densità (o counts) già pronta per imshow.
    extent : [umin, umax, vmin, vmax]
        Extent da passare a imshow.
    (u_hits, v_hits) : opzionale
        Coordinate (u,v) degli impatti individuati.
    """
    POS = np.asarray(POS, float)
    assert POS.ndim == 3 and POS.shape[2] == 3, f"POS atteso (Np, steps, 3), trovato {POS.shape}"
    Np, steps, _ = POS.shape

    P0 = np.asarray(P0, float)
    u_hat, v_hat, n_hat = _plane_frame(n_hat)

    # proiezione signed distances su n_hat
    D = np.tensordot(POS - P0, n_hat, axes=([2], [0]))  # (Np, steps)

    u_hits = []
    v_hits = []

    # per ogni particella cerca i cambi di segno tra step k->k+1
    for i in range(Np):
        d = D[i]                      # (steps,)
        x = POS[i]                    # (steps,3)
        sign = np.sign(d)
        # indici dove cambia segno (escludo zeri esatti per robustezza)
        mask = (sign[:-1] * sign[1:] < 0.0)
        ks = np.nonzero(mask)[0]      # step k in cui avviene crossing tra k e k+1

        for k in ks:
            d0, d1 = d[k], d[k+1]
            # protezione (non dovrebbe succedere dopo mask, ma meglio robusto)
            denom = (d0 - d1)
            if denom == 0.0:
                continue
            alpha = d0 / (d0 - d1)         # frazione tra k -> k+1
            if alpha < 0.0 or alpha > 1.0:
                continue
            Xk  = x[k]
            Xk1 = x[k+1]
            Xc  = Xk + alpha*(Xk1 - Xk)    # punto di attraversamento

            # coordinate 2D nel piano
            r   = Xc - P0
            uu  = np.dot(r, u_hat)
            vv  = np.dot(r, v_hat)
            u_hits.append(uu)
            v_hits.append(vv)

    u_hits = np.array(u_hits, float)
    v_hits = np.array(v_hits, float)

    # istogramma 2D
    Ux, Vy = float(half_extent[0]), float(half_extent[1])
    nx, ny = int(nbins[0]), int(nbins[1])
    u_edges = np.linspace(-Ux, Ux, nx+1)
    v_edges = np.linspace(-Vy, Vy, ny+1)

    H, ue, ve = np.histogram2d(u_hits, v_hits, bins=[u_edges, v_edges])
    # np.histogram2d ritorna H shape (nx, ny); per imshow serve (ny, nx) e origin='lower'
    H = H.T  # (ny, nx)

    if normalize:
        du = (2*Ux) / nx
        dv = (2*Vy) / ny
        bin_area = du * dv
        H = H / bin_area  # densità [counts/m^2]

    extent = [-Ux, +Ux, -Vy, +Vy]

    if return_uv:
        return H, extent, (u_hits, v_hits)
    else:
        return H, extent


def plot_plane_density(H, extent, title="Densità su piano", cmap="viridis",
                       cbar_label="counts", aspect="equal"):
    # Converti extent da [m] a [mm]
    extent_mm = [1e3 * extent[0], 1e3 * extent[1],
                 1e3 * extent[2], 1e3 * extent[3]]

    plt.figure(figsize=(6.2, 5.4))
    im = plt.imshow(H, origin='lower', extent=extent_mm, cmap=cmap, aspect=aspect)
    cbar = plt.colorbar(im)
    cbar.set_label(cbar_label)
    plt.xlabel("x [mm]")
    plt.ylabel("z [mm]")
    plt.title(title)
    plt.tight_layout()
    plt.show()
