import numpy as np
import phisics_sim.src.physics as phy

class Boris_push:

    def step(self, x: np.ndarray, v: np.ndarray, q: float, m: float, dt: float,
        E_half: np.ndarray, B_half: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gamma = phy.gamma_from_v(v)
        p = gamma * m * v
        p_minus = p + q * E_half * (0.5 * dt)
        gamma_minus = np.sqrt(1.0 + np.dot(p_minus, p_minus) / (m * m * phy.c * phy.c))
        t_vec = (q * B_half / (gamma_minus * m)) * (0.5 * dt)
        t2 = float(np.dot(t_vec, t_vec))
        s_vec = (2.0 * t_vec) / (1.0 + t2)
        p_prime = p_minus + np.cross(p_minus, t_vec)
        p_plus = p_minus + np.cross(p_prime, s_vec)
        p_new = p_plus + q * E_half * (0.5 * dt)
        gamma_new = np.sqrt(1.0 + np.dot(p_new, p_new) / (m * m * phy.c * phy.c))
        v_new = p_new / (gamma_new * m)
        x_new = x + 0.5 * (v + v_new) * dt
        return x_new, v_new