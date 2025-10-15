import numpy as np
import phisics_sim.src.physics as phy
import matplotlib.pyplot as plt

__all__ = ["Quadrupole", "gradient_from_focal", "_gradient_from_focal"]

# ----------------- utils -----------------
def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n == 0.0:
        raise ValueError("Null direction")
    return v / n

def _frame(axis):
    """Returns an orthonormal triad (u1, u2, n_hat) with n_hat parallel to axis."""
    n_hat = _unit(axis)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(n_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u1 = _unit(np.cross(n_hat, tmp))
    u2 = np.cross(n_hat, u1)
    return u1, u2, n_hat

# ----------------- optics: gradient from focal length -----------------
def gradient_from_focal(Q: float, m: float, v0: np.ndarray, L: float, focal: float) -> float:
    """
    Quadrupole gradient from target focal length (thin-lens optics):
      1/f = (G * L) / (Bρ),   Bρ = γ m |v| / Q  =>  G = (Bρ) / (f * L)
    """
    v = float(np.linalg.norm(v0))
    gamma = phy.gamma_from_v(v0)
    Brho = (gamma * m * v) / Q  # [T·m]
    return Brho / (float(L) * float(focal))

# backward-compatible alias
def _gradient_from_focal(Q: float, m: float, v0: np.ndarray, L: float, focal: float) -> float:
    return gradient_from_focal(Q, m, v0, L, focal)

# ----------------- ideal quadrupole -----------------
class Quadrupole:
    """
    Ideal magnetic quadrupole (linear field in the transverse plane).

    Parameters
    ----------
    R : float
        Aperture radius (m).
    L : float
        Effective length (m).
    pos : array-like
        Center position (m).
    axis : array-like
        Magnet axis direction (beam direction).
    g : float
        Gradient [T/m] (magnitude); F/D sign is selected by 'polarity'.
    roll : float
        Rotation around the axis (rad). roll=0 => 'normal', roll=pi/4 => 'skew'.
    polarity : int
        +1 / -1 to swap F <-> D.

    Local field convention (axes (u1, u2) already rotated by 'roll'):
        B_local = ( +g_eff * xi,  -g_eff * eta,  0 )
    """
    def __init__(self, R, L, pos, axis, g=1.0, roll=0.0, polarity=+1):
        self.R   = float(R)
        self.L   = float(L)
        self.pos = np.asarray(pos, float)
        self.axis = _unit(axis)
        self.roll = float(roll)
        self.g    = float(g)
        self.polarity = int(np.sign(polarity)) if polarity != 0 else +1

        # Local basis: rotate the transverse plane by 'roll'
        u1, u2, n_hat = _frame(self.axis)
        c, s = np.cos(self.roll), np.sin(self.roll)
        self.u1    =  c * u1 + s * u2
        self.u2    = -s * u1 + c * u2
        self.n_hat = n_hat

        # Local->lab matrix (columns are the local basis vectors)
        self.Rmat = np.column_stack([self.u1, self.u2, self.n_hat])

    def _local_coords(self, x: np.ndarray):
        rel  = np.asarray(x, float) - self.pos
        xi   = float(rel @ self.u1)
        eta  = float(rel @ self.u2)
        zeta = float(rel @ self.n_hat)
        return xi, eta, zeta

    def _inside(self, xi: float, eta: float, zeta: float) -> bool:
        return (abs(zeta) <= 0.5 * self.L) and (xi * xi + eta * eta <= self.R * self.R)

    def magnetic_field(self, x: np.ndarray, t: float = 0.0) -> np.ndarray:
        """
        Returns B(x) in Tesla. Field is ideal and piecewise:
        zero outside the bore and constant-gradient inside.
        """
        xi, eta, zeta = self._local_coords(x)
        if not self._inside(xi, eta, zeta):
            return np.zeros(3)

        g_eff = self.g * (+1.0 if self.polarity >= 0 else -1.0)

        # NOTE: no second rotation here — 'roll' is already embedded in (u1,u2).
        # Ideal field in the transverse plane:
        B_local = np.array([+g_eff * xi, -g_eff * eta, 0.0], dtype=float)

        # Back to lab coordinates
        return self.Rmat @ B_local


def plot_quadrupole_transverse_from_class(q: Quadrupole, n: int = 180):
    """
    Sample B from a Quadrupole object on a section orthogonal to its axis (zeta = 0).

    Shows:
      - contourf of |B|
      - streamplot of the transverse components (B·u1, B·u2)

    Parameters
    ----------
    q : Quadrupole
        The quadrupole instance.
    n : int
        Grid resolution per side in the transverse plane.
    """
    # Orthonormal transverse basis (already in the class)
    u1, u2, n_hat = q.u1, q.u2, q.n_hat

    # Section center
    x0 = q.pos

    # Grid in the (u1, u2) plane
    R = q.R
    s = np.linspace(-R, R, n)
    S1, S2 = np.meshgrid(s, s, indexing="xy")

    # Lab coordinates of section points: x = x0 + s1*u1 + s2*u2  (zeta = 0)
    P = (
        x0.reshape(1, 1, 3)
        + S1[..., None] * u1.reshape(1, 1, 3)
        + S2[..., None] * u2.reshape(1, 1, 3)
    )

    # Compute B at each point
    By = np.empty_like(S1, dtype=float)
    Bz = np.empty_like(S2, dtype=float)
    Bmag = np.empty_like(S1, dtype=float)
    for i in range(n):
        for j in range(n):
            B = q.magnetic_field(P[i, j, :])
            # Project B onto the transverse plane (components along u1, u2)
            By[i, j] = B @ u1
            Bz[i, j] = B @ u2
            Bmag[i, j] = np.linalg.norm(B)

    # Mask outside the aperture
    mask = (S1**2 + S2**2) <= (R**2)
    By = np.where(mask, By, np.nan)
    Bz = np.where(mask, Bz, np.nan)
    Bmag = np.where(mask, Bmag, np.nan)

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    im = ax.contourf(S1, S2, Bmag, levels=28, cmap="viridis")
    plt.colorbar(im, ax=ax, label="|B| [T]")

    # Correct streamlines: (u, v) = (By, Bz), no extra rotation
    ax.streamplot(S1, S2, By, Bz, density=1.1, linewidth=1.0, arrowsize=1.0)

    # Aperture boundary
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(R * np.cos(th), R * np.sin(th), "k-", lw=1)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"coord $u_1$ [m]")
    ax.set_ylabel(r"coord $u_2$ [m]")
    ax.set_title("Magnetic quadrupole — transverse section from class")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    q = Quadrupole(
        R=0.5,
        L=1.0,
        pos=np.zeros(3),
        axis=np.array([1.0, 0.0, 0.0]),
        g=1.0,
        roll=0.0,
        polarity=+1,
    )
    B = q.magnetic_field(np.array([0.0, 0.0, 0.5]))
    print("B =", B)
    plot_quadrupole_transverse_from_class(q)
