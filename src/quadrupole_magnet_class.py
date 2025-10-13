import numpy as np
import phisics_sim.src.physics as phy
import matplotlib.pyplot as plt

__all__ = ["Quadrupole", "gradient_from_focal", "_gradient_from_focal"]

# ----------------- util -----------------
def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n == 0.0:
        raise ValueError("Direzione nulla")
    return v / n

def _frame(axis):
    """Restituisce una terna ortonormale (u1, u2, n_hat) con n_hat || axis."""
    n_hat = _unit(axis)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(n_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u1 = _unit(np.cross(n_hat, tmp))
    u2 = np.cross(n_hat, u1)
    return u1, u2, n_hat

# ----------------- ottica: G da focale -----------------
def gradient_from_focal(Q: float, m: float, v0: np.ndarray, L: float, focal: float) -> float:
    """
    Gradiente quadrupolare dal target di focale (ottica sottile):
      1/f = (G L) / (Bρ),   Bρ = γ m |v| / Q  =>  G = (Bρ)/(f L)
    """
    v = float(np.linalg.norm(v0))
    gamma = phy.gamma_from_v(v0)
    Brho = (gamma * m * v) / Q  # [T·m]
    return Brho / (float(L) * float(focal))


# alias retro-compatibile
def _gradient_from_focal(Q: float, m: float, v0: np.ndarray, L: float, focal: float) -> float:
    return gradient_from_focal(Q, m, v0, L, focal)

# ----------------- quadrupolo ideale -----------------
class Quadrupole:
    """
    Quadrupolo magnetico ideale (campo lineare nel piano trasverso).
    Parametri:
      - R: raggio apertura (m)
      - L: lunghezza efficace (m)
      - pos: centro (m)
      - axis: direzione dell’asse magnete (verso il fascio)
      - g: gradiente [T/m] (modulo); il segno di F/D lo dà 'polarity'
      - roll: rotazione attorno all’asse (rad). roll=0 => 'normale', roll=pi/4 => 'skew'
      - polarity: +1 / -1 per scambiare F<->D
    Convenzione campo locale (assi (u1,u2) ruotati di 'roll'):
        B_local = ( +g_eff * eta,  -g_eff * xi,  0 )
    """
    def __init__(self, R, L, pos, axis, g=1.0, roll=0.0, polarity=+1):
        self.R   = float(R)
        self.L   = float(L)
        self.pos = np.asarray(pos, float)
        self.axis = _unit(axis)
        self.roll = float(roll)
        self.g    = float(g)
        self.polarity = int(np.sign(polarity)) if polarity != 0 else +1

        # base locale: ruoto il piano trasverso di 'roll'
        u1, u2, n_hat = _frame(self.axis)
        c, s = np.cos(self.roll), np.sin(self.roll)
        self.u1   =  c*u1 + s*u2
        self.u2   = -s*u1 + c*u2
        self.n_hat = n_hat

        # matrice locale->lab (colonne = base locale)
        self.Rmat = np.column_stack([self.u1, self.u2, self.n_hat])

    def _local_coords(self, x: np.ndarray):
        rel = np.asarray(x, float) - self.pos
        xi   = float(rel @ self.u1)
        eta  = float(rel @ self.u2)
        zeta = float(rel @ self.n_hat)
        return xi, eta, zeta

    def _inside(self, xi: float, eta: float, zeta: float) -> bool:
        return (abs(zeta) <= 0.5*self.L) and (xi*xi + eta*eta <= self.R*self.R)

    def magnetic_field(self, x: np.ndarray, t: float = 0.0) -> np.ndarray:
        xi, eta, zeta = self._local_coords(x)
        if not self._inside(xi, eta, zeta):
            return np.zeros(3)

        g_eff = self.g * ( +1.0 if self.polarity >= 0 else -1.0 )

        # ATTENZIONE: niente doppia rotazione! Il roll è già nella base (u1,u2).
        # Campo ideale nel piano trasverso:
        B_local = np.array([+g_eff * xi, -g_eff * eta, 0.0], dtype=float)

        # in laboratorio
        return self.Rmat @ B_local


def plot_quadrupole_transverse_from_class(q, n=180):
    """
    Campiona B dal tuo oggetto Quadrupole, su una sezione ortogonale all'asse.
    Mostra:
      - contourf di |B|
      - streamplot (By,Bz) nel piano trasverso
    """
    # costruisci una base ortonormale (u1,u2) nel piano trasverso: la classe l'ha già!
    u1, u2, n_hat = q.u1, q.u2, q.n_hat

    # centro sezione
    x0 = q.pos

    # parametri della griglia nel piano (u1,u2)
    R = q.R
    s = np.linspace(-R, R, n)
    S1, S2 = np.meshgrid(s, s, indexing="xy")

    # coordinate dei punti nel laboratorio: x = x0 + s1*u1 + s2*u2 (sezione a zeta=0)
    P = x0.reshape(1,1,3) + S1[...,None]*u1.reshape(1,1,3) + S2[...,None]*u2.reshape(1,1,3)

    # calcola B in ogni punto
    By = np.empty_like(S1, dtype=float)
    Bz = np.empty_like(S2, dtype=float)
    Bmag = np.empty_like(S1, dtype=float)
    for i in range(n):
        for j in range(n):
            B = q.magnetic_field(P[i,j,:])
            # proietta B sul piano trasverso (componenti lungo u1,u2)
            By[i,j] = B @ u1
            Bz[i,j] = B @ u2
            Bmag[i,j] = np.linalg.norm(B)

    # maschera apertura
    mask = (S1**2 + S2**2) <= (R**2)
    By = np.where(mask, By, np.nan)
    Bz = np.where(mask, Bz, np.nan)
    Bmag = np.where(mask, Bmag, np.nan)

    fig, ax = plt.subplots(figsize=(6.2,6.2))
    im = ax.contourf(S1, S2, Bmag, levels=28, cmap="viridis")
    plt.colorbar(im, ax=ax, label="|B| [T]")

    # STREAMLINE CORRETTE: (u,v) = (By, Bz) e non ruotate
    ax.streamplot(S1, S2, By, Bz, color="C0", density=1.1, linewidth=1.0, arrowsize=1.0)

    # bordo apertura
    th = np.linspace(0, 2*np.pi, 400)
    ax.plot(R*np.cos(th), R*np.sin(th), "k-", lw=1)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("coord $u_1$ [m]")
    ax.set_ylabel("coord $u_2$ [m]")
    ax.set_title("Magnete quadrupolare — sezione trasversa dalla classe")
    plt.tight_layout()
    plt.show()

# ----------------- Esempio rapido -----------------

def plot_quadrupole_transverse_from_class(q, n=180):
    """
    Campiona B dal tuo oggetto Quadrupole, su una sezione ortogonale all'asse.
    Mostra:
      - contourf di |B|
      - streamplot (By,Bz) nel piano trasverso
    """
    # costruisci una base ortonormale (u1,u2) nel piano trasverso: la classe l'ha già!
    u1, u2, n_hat = q.u1, q.u2, q.n_hat

    # centro sezione
    x0 = q.pos

    # parametri della griglia nel piano (u1,u2)
    R = q.R
    s = np.linspace(-R, R, n)
    S1, S2 = np.meshgrid(s, s, indexing="xy")

    # coordinate dei punti nel laboratorio: x = x0 + s1*u1 + s2*u2 (sezione a zeta=0)
    P = x0.reshape(1,1,3) + S1[...,None]*u1.reshape(1,1,3) + S2[...,None]*u2.reshape(1,1,3)

    # calcola B in ogni punto
    By = np.empty_like(S1, dtype=float)
    Bz = np.empty_like(S2, dtype=float)
    Bmag = np.empty_like(S1, dtype=float)
    for i in range(n):
        for j in range(n):
            B = q.magnetic_field(P[i,j,:])
            # proietta B sul piano trasverso (componenti lungo u1,u2)
            By[i,j] = B @ u1
            Bz[i,j] = B @ u2
            Bmag[i,j] = np.linalg.norm(B)

    # maschera apertura
    mask = (S1**2 + S2**2) <= (R**2)
    By = np.where(mask, By, np.nan)
    Bz = np.where(mask, Bz, np.nan)
    Bmag = np.where(mask, Bmag, np.nan)

    fig, ax = plt.subplots(figsize=(6.2,6.2))
    im = ax.contourf(S1, S2, Bmag, levels=28, cmap="viridis")
    plt.colorbar(im, ax=ax, label="|B| [T]")

    # STREAMLINE CORRETTE: (u,v) = (By, Bz) e non ruotate
    ax.streamplot(S1, S2, By, Bz, color="C0", density=1.1, linewidth=1.0, arrowsize=1.0)

    # bordo apertura
    th = np.linspace(0, 2*np.pi, 400)
    ax.plot(R*np.cos(th), R*np.sin(th), "k-", lw=1)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("coord $u_1$ [m]")
    ax.set_ylabel("coord $u_2$ [m]")
    ax.set_title("Magnete quadrupolare — sezione trasversa dalla classe")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    q = Quadrupole(R=0.5, L=1.0, pos=np.zeros(3), axis=np.array([1.0, 0.0, 0.0]), g=1.0, roll=0.0, polarity=+1)
    B = q.magnetic_field(np.array([0.0, 0.0, 0.5]))
    print("B =", B)
    plot_quadrupole_transverse_from_class(q)
