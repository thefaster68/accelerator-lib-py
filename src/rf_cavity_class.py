import numpy as np

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


class RF_cavity:
    """
    Cavità rf con campo oscillante interno.
    Opzionalmente, i bordi lungo la normale e quelli laterali possono essere
    smussati con una finestra tanh per evitare discontinuità numeriche.

    Convenzioni:
    - n_hat punta dall'"ingresso" verso "l'usicta" della cavità.
      All'interno del gap: E = (sigma / (ε0 εr)) * n_hat.
    - Volume attivo: spessore d lungo n_hat, base lungo l_hat, height lungo m_hat.
    """
    sigma: float = 0.0               # densità di carica superficiale [C/m^2]
    d:     float = 0.1               # spessore del gap [m]
    base:  float = 1.0               # estensione lungo l_hat [m]
    height: float = 1.0              # estensione lungo m_hat [m]
    n_hat: np.ndarray = np.zeros(3)  # normale del gap (verso +)
    epsilon_r: float = 1.0           # costante dielettrica relativa
    pos: np.ndarray = np.zeros(3)    # centro del volume attivo

    # Nuovi parametri (opzionali)
    edge_delta: float = 0.0          # larghezza transizione lungo n_hat [m]
    lateral_delta: float = 0.0       # larghezza transizione laterale [m]

    def __init__(self, E0: float, d_0: float, R: float,
                 normal: np.ndarray, position: np.ndarray,
                 omega: float, phi: float,
                 delta: float | None = None):
        self.E0 = E0
        self.d = float(d_0)
        self.R = R
        normal = np.asarray(normal, dtype=float)
        self.axis = _unit(normal)
        self.pos = np.asarray(position, dtype=float)
        u1, u2, n_hat = _frame(self.axis)
        self.u1 = u1
        self.u2 = u2
        self.n_hat = n_hat
        self.omega = omega
        self.phi = phi

        # matrice locale->lab (colonne = base locale)
        self.Rmat = np.column_stack([self.u1, self.u2, self.n_hat])
        # Smussature (default 0 = comportamento precedente)
        self.delta = float(delta) if delta is not None else 0.0

    # ---------------------- Geometria locale ----------------------

    def _local_coords(self, x: np.ndarray):
        rel = np.asarray(x, float) - self.pos
        xi   = float(rel @ self.u1)
        eta  = float(rel @ self.u2)
        zeta = float(rel @ self.n_hat)
        return xi, eta, zeta

    def _inside(self, xi: float, eta: float, zeta: float) -> bool:
        return (abs(zeta) <= 0.5*self.d) and (xi*xi + eta*eta <= self.R*self.R)

    # ---------------------- Finestre lisce ----------------------
    @staticmethod
    def _window_tanh(s: float, half_width: float, delta: float) -> float:
        """
        Finestra 1D liscia: ~1 per |s| < half_width, ~0 fuori.
        delta controlla la rapidità della transizione (delta -> 0 => gradino).
        """
        if delta <= 0.0:
            return 1.0 if abs(s) <= half_width else 0.0
        return 0.5 * (np.tanh((s + half_width) / delta) - np.tanh((s - half_width) / delta))

    def electric_field(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Campo elettrico nella cavità.
        - Se delta == 0.0:
          E = E0 * n_hat dentro, 0 fuori.
        - Altrimenti: E = E0 * w_n(s_n) * n_hat  (finestre tanh).
        """
        xi, eta, zeta = self._local_coords(x)
        if not self._inside(xi, eta, zeta):
            return np.zeros(3)

        E_zeta = self.E0 * np.cos(self.omega * t + self.phi) * self._window_tanh(zeta, 0.5 * self.d, self.delta)
        E_loc = np.array([0, 0, E_zeta])

        E_lab = self.Rmat @ E_loc

        return E_lab



