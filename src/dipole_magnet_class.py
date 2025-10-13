import numpy as np
import physics as phy

def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n == 0.0:
        raise ValueError("Direzione nulla")
    return v / n

def _frame(axis):
    """Restituisce un terna ortonormale (u1, u2, n_hat) con n_hat || axis."""
    n_hat = _unit(axis)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(n_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u1 = _unit(np.cross(n_hat, tmp))   # perpendicolare a n_hat e tmp
    u2 = np.cross(n_hat, u1)           # completa la terna destra
    return u1, u2, n_hat

def _gradient_from_focal(self, Q: float, m: float, v0: np.ndarray):
    rho = Q / (phy.c * m * np.linalg.norm(v0) * phy.gamma_from_v(v0))
    g = rho / (self.L * self.focal)
    return g


class Dipole_mag:
    """
    Quadrupolo magnetico ideale (campo lineare nel piano trasverso).
    - R: raggio dell’apertura (m)
    - L: lunghezza efficace (m)
    - pos: centro (m)
    - axis: direzione dell’asse magnete
    - g: gradiente [T/m]; segno = polarità (F/D)
    - roll: rotazione attorno all’asse (rad). roll=0 -> 'normale', roll=pi/4 -> 'skew'
    """
    def __init__(self, intensity, R, L, pos, axis, roll=0.0):
        self.intensity = intensity
        self.R   = float(R)
        self.L   = float(L)
        self.pos = np.asarray(pos, float)
        self.axis = _unit(axis)
        self.roll = roll

        # Base locale
        u1, u2, n_hat = _frame(self.axis)

        # Rotazione di roll nel piano trasverso (per quadrupolo skew)
        c, s = np.cos(roll), np.sin(roll)
        self.u1   =  c*u1 + s*u2
        self.u2   = -s*u1 + c*u2
        self.n_hat = n_hat

        # Matrice di rotazione locale->laboratorio (colonne = base locale)
        self.Rmat = np.column_stack([self.u1, self.u2, self.n_hat])

    def _local_coords(self, x: np.ndarray):
        rel = np.asarray(x, float) - self.pos
        xi   = rel @ self.u1
        eta  = rel @ self.u2
        zeta = rel @ self.n_hat
        return xi, eta, zeta

    def _inside(self, xi: float, eta: float, zeta: float) -> bool:
        return (abs(zeta) <= 0.5*self.L) and (xi*xi + eta*eta <= self.R*self.R)

    def magnetic_field(self, x: np.ndarray, t: float = 0.0, polarity: int = +1) -> np.ndarray:
        """
        Ritorna B(x) in Tesla. polarity=+1/-1 inverte F<->D.
        """
        xi, eta, zeta = self._local_coords(x)
        if not self._inside(xi, eta, zeta):
            return np.zeros(3)

        Rot = np.column_stack([[np.cos(self.roll), -np.sin(self.roll), 0.0],
                        [np.sin(self.roll),  np.cos(self.roll), 0.0],
                        [0.0,                0.0,               1.0]])

        # Campo locale ideale: B_local = (-g*xi, +g*eta, 0)
        B_local = self.intensity * Rot @ np.array([1.0, 0.0, 0.0])

        # Trasforma in laboratorio: B_lab = Rmat @ B_local
        return self.Rmat @ B_local





def plot_B_field_2d(halbach, extent, N=121, plane="yz", soft=0.0, bore_frac=0.8, title=None):
    # sezione a x = x0 (centro magnete)
    x0 = halbach.pos
    u1, u2, n_hat = halbach.u1, halbach.u2, halbach.n_hat

    y = np.linspace(-extent, extent, N)
    z = np.linspace(-extent, extent, N)
    Y, Z = np.meshgrid(y, z, indexing="xy")

    Babs = np.full_like(Y, np.nan, dtype=float)
    By   = np.full_like(Y, np.nan, dtype=float)
    Bz   = np.full_like(Y, np.nan, dtype=float)

    for i in range(N):
        for j in range(N):
            r_vec = Y[i,j]*u1 + Z[i,j]*u2
            if np.linalg.norm(r_vec) <= bore_frac*halbach.R:   # solo interno foro
                x = x0 + r_vec
                B = halbach.magnetic_field(x, 0.0)
                Babs[i,j] = np.linalg.norm(B)
                By[i,j]   = np.dot(B, u1)
                Bz[i,j]   = np.dot(B, u2)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(Babs.T, origin="lower", extent=[y[0], y[-1], z[0], z[-1]], aspect="equal")
    ax.streamplot(y, z, By.T, Bz.T, density=1.1, linewidth=1.0, arrowsize=1.0, color="tab:blue")
    plt.colorbar(im, ax=ax, label="|B| [T]")
    ax.set_xlabel("coord y [m]"); ax.set_ylabel("coord z [m]")
    ax.set_title(title or "Magnete dipolare (sezione trasversa)")
    plt.tight_layout()
    plt.show()

# ----------------- Esempio rapido -----------------
if __name__ == "__main__":
    q = Dipole_mag(intensity=2.0, R=1.0, L=1.0, pos=np.zeros(3), axis=np.array([1.0, 0.0, 0.0]), roll=np.pi * 0)
    B = q.magnetic_field(np.array([0.5, 0.5, 0.5]))
    print("B =", B)
    plot_B_field_2d(q, 1.0, 124, "yz")
