
import numpy as np
import random as rnd
import phisics_sim.src.particle_class as prt

__all__ = ["particle_grid_spawner"]


def _unit(v: np.ndarray) -> np.ndarray:
    """
    Ritorna la norma di un vettore v + controllo sel v = 0
    """
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0.0:
        raise ValueError("Direzione nulla: impossibile normalizzare un vettore nullo.")
    return v / n


def _frame_from_axis(axis: np.ndarray):
    """
    Dato un asse (versore) restituisce una terna ortonormale (u, v, n)
    con n parallelo all'asse e u, v nel piano trasverso.
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
    Popola 'particles' con una griglia n_rows × n_cols centrata in 'center' e
    disposta nel piano trasverso ortogonale a 'beam_dir'.
    La velocità media è allineata alla direzione del fascio.

    Parametri
    ---------
    particles : list
        Lista da riempire con oggetti prt.Particle().
    n_rows, n_cols : int
        Numero di righe e colonne della griglia trasversa.
    dx, dy : float
        Passi lungo i due versori trasversi (metri). La griglia è centrata.
    q, mass : float
        Carica e massa delle particelle.
    v0 : float | array-like
        Se float: modulo della velocità media, direzione presa da 'beam_dir'.
        Se array-like shape (3,): velocità media completa; 'beam_dir' viene ignorato.
    center : array-like shape (3,)
        Centro della griglia in coordinate globali (metri).
    beam_dir : array-like shape (3,)
        Direzione del fascio; usata solo se v0 è scalare.
    jitter_rel : float
        Rumore gaussiano relativo sulle componenti della velocità (default 1e-6).
    start_index : int
        Indice iniziale da assegnare alle particelle (incrementa automaticamente).

    Ritorna
    -------
    list : la stessa lista 'particles' riempita.
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
    Popola particles con particelle disposte su una circonferenza orientata equidistanti tra loro

    :param particles:
        la lista di particelle da popolare
    :param Np:
        il numero di particelle da disporre sulla circonferenza
    :param R:
        il raggio della circonferenza
    :param center:
        il centro della circonferenza
    :param axis:
        l'asse di propagazione delle particelle
    :param q:
        la carica delle particelle
    :param mass:
        la massa delle particelle
    :param v0:
        il modulo della velocità delle particelle
    :return: la lista di particelle popolata
    """

    t_i = np.linspace(0, 2*np.pi * (1 - 1/Np), Np)
    u, v, n = _frame_from_axis(axis)
    B = np.column_stack([u, v, n])
    for i in range(Np):
        x_loc = np.array([R*np.cos(t_i[i]), R*np.sin(t_i[i]), 0.0])
        p = prt.Particle(); p.generate(i, q, mass, center + B @ x_loc, v0*axis)
        particles.append(p)
    return particles
