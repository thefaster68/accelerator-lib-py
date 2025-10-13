import numpy as np
import physics as phy

class Dipole:
    def __init__(self, pos:np.ndarray, moment: np.ndarray):
        self.pos = pos
        self.moment = moment

    def potential_vector(self, r_rel: np.ndarray, t: float) -> np.ndarray:
        R2 = float(np.dot(r_rel, r_rel))
        if R2 == 0.0:
            return np.zeros(3)
        A = phy.mu_0/(4*np.pi) * np.cross(self.moment, r_rel) / (R2*np.sqrt(R2))
        return A

    def magnetic_field(self, x: np.ndarray, t: float) -> np.ndarray:
        B = phy.curl_central(x, self.pos, self.potential_vector, t)
        return B