
import numpy as np
import random as rnd
import phisics_sim.src.particle_class as prt

__all__ = ["particle_grid_spawner"]


def _unit(v: np.ndarray) -> np.ndarray:
    """
    Returns the norm of a vector v, with a check in case v = 0.
    """
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0.0:
        raise ValueError("Direzione nulla: impossibile normalizzare un vettore nullo.")
    return v / n


def _frame_from_axis(axis: np.ndarray):
    """
    Given an axis (unit vector), returns an orthonormal triad (u, v, n)
    with n parallel to the axis and u, v lying in the transverse plane.
    """
    n = _unit(axis)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(n, a))
    v = _unit(np.cross(n, u))
    return u, v, n


def particle_grid_spawner(
    particles: list,
    n_rows: int,
    n_cols: int,
    dx: float,
    dy: float,
    q: float,
    mass: float,
    v0,
    center: np.ndarray | list | tuple = (0.0, 0.0, 0.0),
    beam_dir: np.ndarray | list | tuple = (1.0, 0.0, 0.0),
    jitter_rel: float = 1e-6,
    start_index: int = 0,
):
    """
    Fills 'particles' with an n_rows × n_cols grid centered at 'center' and
    placed in the transverse plane orthogonal to 'beam_dir'.
    The mean velocity is aligned with the beam direction.
    Parameters
    ----------
    particles : list
        List to be filled with prt.Particle() objects.
    n_rows, n_cols : int
        Number of rows and columns of the transverse grid.
    dx, dy : float
        Steps along the two transverse unit vectors (meters). The grid is centered.
    q, mass : float
        Charge and mass of the particles.
    v0 : float | array-like
        If float: magnitude of the mean velocity; direction taken from 'beam_dir'.
        If array-like with shape (3,): full mean velocity; 'beam_dir' is ignored.
    center : array-like with shape (3,)
        Grid center in global coordinates (meters).
    beam_dir : array-like with shape (3,)
        Beam direction; used only if v0 is a scalar.
    jitter_rel : float
        Relative Gaussian noise on the velocity components (default 1e-6).
    start_index : int
        Starting index to assign to particles (auto-increments).
    Returns
    -------
    list : the same 'particles' list, now filled.
    """

    center = np.asarray(center, dtype=float)
    if np.ndim(v0) == 0:
        n = _unit(np.asarray(beam_dir, dtype=float))
        avg_vel = float(v0) * n
    else:
        avg_vel = np.asarray(v0, dtype=float)
        n = _unit(avg_vel)

    u, v, _ = _frame_from_axis(n)

    cx = (n_cols - 1) / 2.0
    ry = (n_rows - 1) / 2.0

    idx = start_index
    for r in range(n_rows):
        for c in range(n_cols):
            du = (c - cx) * dx
            dv = (r - ry) * dy
            pos = center + du * u + dv * v

            vel = np.zeros(3, dtype=float)
            for k in range(3):
                sigma = abs(avg_vel[k]) * jitter_rel
                vel[k] = rnd.gauss(avg_vel[k], sigma)

            p = prt.Particle()
            p.generate(idx, q, mass, pos, vel)
            particles.append(p)
            idx += 1

    return particles


def particle_circle_spawner(
    particles: list,
    Np: int,
    R: float,
    center: np.ndarray,
    axis: np.ndarray,
    q: float,
    mass: float,
    v0:float
):
    """
    Populate 'particles' with particles placed equally spaced on an oriented circle.

    :param particles:
        the list of particles to populate
    :param Np:
        the number of particles to place on the circle
    :param R:
        the radius of the circle
    :param center:
        the center of the circle
    :param axis:
        the propagation axis of the particles
    :param q:
        the charge of the particles
    :param mass:
        the mass of the particles
    :param v0:
        the speed (magnitude of velocity) of the particles
    :return: the populated list of particles
    """


    t_i = np.linspace(0, 2*np.pi * (1 - 1/Np), Np)
    u, v, n = _frame_from_axis(axis)
    B = np.column_stack([u, v, n])
    for i in range(Np):
        x_loc = np.array([R*np.cos(t_i[i]), R*np.sin(t_i[i]), 0.0])
        p = prt.Particle(); p.generate(i, q, mass, center + B @ x_loc, v0*axis)
        particles.append(p)
    return particles
