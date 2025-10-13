import numpy as np
import phisics_sim.src.physics as phy

class Particle:
    q:         float = 0.0
    mass:      float = 0.0
    index:       int = 0
    pos: np.ndarray = np.zeros(3)
    vel: np.ndarray = np.zeros(3)

    def generate(self, index: int, q: float, mass: float, pos0: np.ndarray, vel0: np.ndarray):
        self.index = index
        self.q = q
        self.mass = mass
        self.pos = pos0
        self.vel = vel0

    def intraparticle_force(self, Q: float, epsilon_r: float, pos_other: np.ndarray, vel_other: np.ndarray) -> np.ndarray:
        F = Q * self.q / (4.0 * np.pi * phy.epsilon_0 * epsilon_r * np.dot(pos_other - self.pos, pos_other - self.pos)) * \
            np.linalg.norm(pos_other - self.pos)\
            * (np.linalg.norm(pos_other - self.pos) + 1.0 / (phy.c * phy.c) *
            np.cross(np.cross(self.vel, vel_other), (pos_other - self.pos)))
        return F

    def min_interaction_rad(self, Q: float, epsilon_r: float, pos_other: np.ndarray, vel_other: np.ndarray) -> float:

        R_min = Q * self.q / (4.0 * np.pi * phy.epsilon_0 * epsilon_r * 2.22e-16) * \
                np.sqrt(np.dot((np.linalg.norm(pos_other - self.pos) + 1.0 / (phy.c * phy.c) *
                np.cross(np.cross(self.vel, vel_other), (pos_other - self.pos))),(np.linalg.norm(pos_other - self.pos) + 1.0 / (phy.c * phy.c) *
                np.cross(np.cross(self.vel, vel_other), (pos_other - self.pos)))))

        return R_min
