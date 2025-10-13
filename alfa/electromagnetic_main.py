import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

epsilon_0 = 8.854e-12
epsilon_r = 1.0


def lorentz_transform(vel):
    c = 1e8
    gamma = 1.0 / np.sqrt(1.0 - pow(np.linalg.norm(vel) / c, 2))
    return gamma

# --- Definizione dei potenziali ---
def scalar_pot(pos, normal, t):
    sigma = 1e-3
    n_hat = normal / np.linalg.norm(normal)
    return sigma / (epsilon_0 * epsilon_r) * np.dot(n_hat, pos)

def point_pot(pos, _, t):
    q = -1e-5
    r = np.linalg.norm(pos)
    return q / (4 * np.pi * epsilon_0 * epsilon_r * r) if r > 0 else 0.0

def alternate_scalar_pot(pos, normal, t):
    # potenziale uniforme fra due piastre piane (gap)
    sigma = 1e-5
    n_hat = normal / np.linalg.norm(normal)
    return sigma / (epsilon_0 * epsilon_r) * np.dot(n_hat, pos)

# --- Gradiente numerico rimane unico ---
def gradient(f_scalar, pos, normal, t):
    delta = 1e-6
    grad = np.zeros(3)
    for i in range(3):
        d = np.zeros(3); d[i] = delta
        grad[i] = (f_scalar(pos + d, normal, t)
                 - f_scalar(pos - d, normal, t)) / (2 * delta)
    return grad

# --- Campo magnetico uniforme ---
def uniform_B(t):
    return np.array([0.0, 0.0, 1.0])  # B lungo z

# --- Aggiornamento accelerazione con potenziale alternato (ciclotrone) ---
def Update_acc(pos, vel, t, normal, d, f_scalar, potential_type, q_charge, m_mass):
    n_hat = normal / np.linalg.norm(normal)
    B = uniform_B(t)

    if potential_type == 'alternate':
        # Frequenza ciclotrone (rad/s)
        omega_c = q_charge * np.linalg.norm(B) / (m_mass * lorentz_transform(vel))

        # Campo elettrico solo nella zona di gap: |n·pos| <= d
        if abs(np.dot(n_hat, pos)) <= d:
            # E0 dalla pendenza del potenziale
            E_plate = -gradient(f_scalar, pos, normal, t)
            # segnale alternato al ritmo della frequenza ciclotrone
            E = E_plate * np.sign(np.sin(omega_c * t))
            F_magnetic = np.zeros(3)
        else:
            # All'interno dei dee: solo forza di Lorentz magnetica
            E = np.zeros(3)
            F_magnetic = q_charge * np.cross(vel, B)
    else:
        # mantiene le modalità "plate" e "point" originali
        if potential_type == 'plate':
            if abs(np.dot(n_hat, pos)) <= d:
                E = -gradient(f_scalar, pos, normal, t)
                F_magnetic = np.zeros(3)
            else:
                E = np.zeros(3)
                F_magnetic = q_charge * np.cross(vel, B)
        elif potential_type == 'point':
            E = -gradient(f_scalar, pos, normal, t)
            F_magnetic = q_charge * np.cross(vel, B)
        else:
            raise ValueError("potential_type deve essere 'plate', 'point' o 'alternate'")

    F_electric = q_charge * E
    F_tot = F_electric + F_magnetic
    return 1.0/(lorentz_transform(vel) * m_mass) * (F_tot - np.dot(vel, F_tot) / (1e16) * vel)

# --- Parametri di simulazione ---
dt       = 1e-8
steps    = 200000
q_charge = 1e-5
m_mass   = 1e-9
normal   = np.array([1.0, 0.0, 0.0])
d        = 1e-3  # metà gap ridotto per rendere distinti i dee
distance_limit = d

# --- Scegli il potenziale: 'plate', 'point' o 'alternate' ---
potential_type = 'alternate'
if potential_type == 'plate':
    f_scalar = scalar_pot
elif potential_type == 'point':
    f_scalar = point_pot
elif potential_type == 'alternate':
    f_scalar = alternate_scalar_pot
else:
    raise ValueError("potential_type deve essere 'plate', 'point' o 'alternate'")

# Inizializza traiettoria
pos = np.array([0.0, 0.0, 0.0])
vel = np.array([0.0, 0.0, 0.0])  # velocità iniziale orizzontale
traj = np.zeros((steps, 3))

# Integrazione Verlet
a = Update_acc(pos, vel, 0.0, normal, d, f_scalar, potential_type, q_charge, m_mass)
for i in range(steps):
    t = i * dt
    pos = pos + vel * dt + 0.5 * a * dt**2
    a_new = Update_acc(pos, vel, t + dt, normal, d, f_scalar, potential_type, q_charge, m_mass)
    vel = vel + 0.5 * (a + a_new) * dt
    a = a_new
    traj[i] = pos
    print(i, "\t pos=", pos, "\t vel=", vel, "\t |vel|=", np.linalg.norm(vel))

# Plot 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect((1, 1, 1))
ax.plot(traj[:,0], traj[:,1], traj[:,2], lw=1, color='blue')

# Disegna gap
if potential_type == 'plate' or potential_type == 'alternate':
    n_hat = normal / np.linalg.norm(normal)
    aux = np.array([1,0,0]) if abs(n_hat.dot([1,0,0]))<0.9 else np.array([0,1,0])
    u = np.cross(n_hat, aux); u /= np.linalg.norm(u)
    v = np.cross(n_hat, u)
    uv = traj.dot(np.vstack((u,v)).T)
    u_min,u_max = uv[:,0].min()-0.1, uv[:,0].max()+0.1
    v_min,v_max = uv[:,1].min()-0.1, uv[:,1].max()+0.1

    verts = []
    for offset in (-d*n_hat, +d*n_hat):
        for uu in (u_min, u_max):
            for vv in (v_min, v_max):
                verts.append(offset + u*uu + v*vv)
    verts = np.array(verts)
    faces = [[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]]
    poly3d = [[verts[i] for i in face] for face in faces]
    coll = Poly3DCollection(poly3d, facecolor='red', edgecolor='k', alpha=0.15)
    ax.add_collection3d(coll)

ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
plt.show()
