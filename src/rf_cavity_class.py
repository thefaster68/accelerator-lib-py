import numpy as np

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


class RF_cavity:
    """
    RF cavity with an oscillating internal field.
    Optionally, the edges along the normal and the lateral sides can be
    smoothed using a tanh window to avoid numerical discontinuities.

    Conventions:
    - n_hat points from the cavity "entrance" to the "exit".
      Inside the gap: E = (sigma / (ε0 εr)) * n_hat.
    - Active volume: thickness d along n_hat, base along l_hat, height along m_hat.
    """
    sigma: float = 0.0               # surface charge density [C/m^2]
    d:     float = 0.1               # gap thickness [m]
    base:  float = 1.0               # extent along l_hat [m]
    height: float = 1.0              # extent along m_hat [m]
    n_hat: np.ndarray = np.zeros(3)  # gap normal (toward +)
    epsilon_r: float = 1.0           # relative permittivity
    pos: np.ndarray = np.zeros(3)    # center of the active volume

    # Optional smoothing parameters
    edge_delta: float = 0.0          # transition width along n_hat [m]
    lateral_delta: float = 0.0       # lateral transition width [m]

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

        # local->lab rotation (columns = local basis)
        self.Rmat = np.column_stack([self.u1, self.u2, self.n_hat])
        # Smoothing (default 0 = previous sharp-step behavior)
        self.delta = float(delta) if delta is not None else 0.0

    # ---------------------- Local geometry ----------------------

    def _local_coords(self, x: np.ndarray):
        rel  = np.asarray(x, float) - self.pos
        xi   = float(rel @ self.u1)
        eta  = float(rel @ self.u2)
        zeta = float(rel @ self.n_hat)
        return xi, eta, zeta

    def _inside(self, xi: float, eta: float, zeta: float) -> bool:
        return (abs(zeta) <= 0.5 * self.d) and (xi * xi + eta * eta <= self.R * self.R)

    # ---------------------- Smooth windows ----------------------
    @staticmethod
    def _window_tanh(s: float, half_width: float, delta: float) -> float:
        """
        Smooth 1D window: ~1 for |s| < half_width, ~0 outside.
        delta controls the sharpness of the transition (delta -> 0 => step function).
        """
        if delta <= 0.0:
            return 1.0 if abs(s) <= half_width else 0.0
        return 0.5 * (np.tanh((s + half_width) / delta) - np.tanh((s - half_width) / delta))

    def electric_field(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Electric field inside the cavity.
        - If delta == 0.0:
          E = E0 * n_hat inside, 0 outside.
        - Otherwise: E = E0 * w_n(s_n) * n_hat  (tanh window).
        """
        xi, eta, zeta = self._local_coords(x)
        if not self._inside(xi, eta, zeta):
            return np.zeros(3)

        E_zeta = self.E0 * np.cos(self.omega * t + self.phi) * self._window_tanh(zeta, 0.5 * self.d, self.delta)
        E_loc = np.array([0.0, 0.0, E_zeta])

        E_lab = self.Rmat @ E_loc
        return E_lab
