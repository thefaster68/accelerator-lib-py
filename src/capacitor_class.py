import numpy as np
import phisics_sim.src.physics as phy


class Capacitor:
    """
    Condensatore planare rettangolare con campo (ideale) uniforme all'interno.
    Opzionalmente, i bordi lungo la normale e quelli laterali possono essere
    smussati con una finestra tanh per evitare discontinuità numeriche.

    Convenzioni:
    - n_hat punta dalla piastra “-” verso la piastra “+”.
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

    def __init__(self, sigma_0: float, d_0: float, base: float, height: float,
                 normal: np.ndarray, Epsilon_r: float, position: np.ndarray,
                 edge_delta: float | None = None, lateral_delta: float | None = None):
        self.sigma = float(sigma_0)
        self.d = float(d_0)
        self.base = float(base)
        self.height = float(height)
        normal = np.asarray(normal, dtype=float)
        self.n_hat = normal / np.linalg.norm(normal)
        self.epsilon_r = float(Epsilon_r)
        self.pos = np.asarray(position, dtype=float)

        # Smussature (default 0 = comportamento precedente)
        self.edge_delta = float(edge_delta) if edge_delta is not None else 0.0
        self.lateral_delta = float(lateral_delta) if lateral_delta is not None else 0.0

        # base ortonormale locale {n_hat, l_hat, m_hat}
        self._make_local_basis()

    # ---------------------- Geometria locale ----------------------
    def _make_local_basis(self):
        n = self.n_hat
        A = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        l_hat = np.cross(n, A);  l_hat /= np.linalg.norm(l_hat)
        m_hat = np.cross(n, l_hat)
        self._l_hat = l_hat
        self._m_hat = m_hat

    def _coords_locali(self, x: np.ndarray):
        """Coordinate locali (s_n, s_l, s_m) rispetto al centro self.pos."""
        x_rel = np.asarray(x, dtype=float) - self.pos
        s_n = float(np.dot(self.n_hat, x_rel))
        s_l = float(np.dot(self._l_hat, x_rel))
        s_m = float(np.dot(self._m_hat, x_rel))
        return s_n, s_l, s_m

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

    # ---------------------- API esistente ----------------------
    def phi_gap_rf(self, x: np.ndarray, t: float) -> float:
        """
        Potenziale 'ideale' del gap (solo lungo n_hat), con saturazione lineare fuori.
        Manteniamo per retro-compatibilità; con smoothing attivo non usiamo il gradiente numerico.
        """
        ndotx = float(np.dot(self.n_hat, x - self.pos))
        pref = -(self.sigma / (phy.epsilon_0 * self.epsilon_r))
        if self.in_capacitor(x):
            return pref * ndotx
        else:
            return pref * (0.5 * self.d) * np.sign(ndotx)

    def in_capacitor(self, x: np.ndarray) -> bool:
        """Test 'duro' del vecchio comportamento (senza smoothing)."""
        s_n, s_l, s_m = self._coords_locali(x)
        return (abs(s_n) <= 0.5 * self.d
                and abs(s_l) <= 0.5 * self.base
                and abs(s_m) <= 0.5 * self.height)

    def electric_field(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Campo elettrico nel gap.
        - Se edge_delta == lateral_delta == 0.0: comportamento precedente (gradino).
          E = E0 * n_hat dentro, 0 fuori.
        - Altrimenti: E = E0 * w_n(s_n) * w_l(s_l) * w_m(s_m) * n_hat  (finestre tanh).
        """
        E0 = (self.sigma / (phy.epsilon_0 * self.epsilon_r))
        s_n, s_l, s_m = self._coords_locali(x)

        if self.edge_delta <= 0.0 and self.lateral_delta <= 0.0:
            return (E0 * self.n_hat) if self.in_capacitor(x) else np.zeros(3, dtype=float)

        wn = self._window_tanh(s_n, 0.5 * self.d, max(self.edge_delta, 0.0))
        wl = self._window_tanh(s_l, 0.5 * self.base, max(self.lateral_delta, 0.0)) if self.base > 0.0 else 1.0
        wm = self._window_tanh(s_m, 0.5 * self.height, max(self.lateral_delta, 0.0)) if self.height > 0.0 else 1.0
        mask = float(wn * wl * wm)
        return (E0 * mask) * self.n_hat
