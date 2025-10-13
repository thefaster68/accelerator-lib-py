import numpy as np
import phisics_sim.src.physics as phy  # deve definire epsilon_0 e c

# ---------- 1) Storia temporale ----------
class HistoryBuffer:
    """
    Buffer circolare di lunghezza 'hist_len' con campionamento fisso dt.
    Conserva t, pos(N,3), vel(N,3).
    """
    def __init__(self, N: int, hist_len: int, dt: float):
        self.N = N
        self.hist_len = hist_len
        self.dt = float(dt)
        self.t = np.full(hist_len, np.nan, float)
        self.pos = np.zeros((N, hist_len, 3), float)
        self.vel = np.zeros((N, hist_len, 3), float)
        self.head = -1  # nessun dato ancora

    def push(self, t: float, pos: np.ndarray, vel: np.ndarray):
        assert pos.shape == (self.N, 3) and vel.shape == (self.N, 3)
        self.head = (self.head + 1) % self.hist_len
        self.t[self.head] = t
        self.pos[:, self.head, :] = pos
        self.vel[:, self.head, :] = vel

    def valid(self) -> bool:
        return self.head >= 0 and not np.isnan(self.t[self.head])

    def _unroll_indices(self):
        """Restituisce l'ordine temporale crescente degli indici del buffer."""
        if self.head < 0:
            return np.array([], int)
        idx = np.arange(self.hist_len, dtype=int)
        return np.concatenate([idx[self.head+1:], idx[:self.head+1]])

    def get_arrays_unrolled(self, j: int):
        """Ritorna (times, pos_j, vel_j) in ordine temporale crescente."""
        order = self._unroll_indices()
        times = self.t[order]
        mask = ~np.isnan(times)
        times = times[mask]
        pos_j = self.pos[j, order, :][mask]
        vel_j = self.vel[j, order, :][mask]
        return times, pos_j, vel_j

# ---------- 2) Tempo ritardato via bisezione + interpolazione lineare ----------
def _interp_state(times: np.ndarray, pos_j: np.ndarray, vel_j: np.ndarray, k: int, s: float):
    """
    Interpola tra campioni k (tau_k) e k+1 (tau_{k+1}) con parametro s in [0,1].
    Ritorna (r_j(τ), v_j(τ), a_j(τ) approx) con accelerazione costante sul segmento.
    """
    r0, r1 = pos_j[k], pos_j[k+1]
    v0, v1 = vel_j[k], vel_j[k+1]
    tau0, tau1 = times[k], times[k+1]
    dtau = tau1 - tau0
    r = (1.0 - s) * r0 + s * r1
    v = (1.0 - s) * v0 + s * v1
    # a costante sul segmento (stima da differenza finita)
    a = (v1 - v0) / dtau if dtau > 0 else np.zeros(3)
    tau = tau0 + s * dtau
    return tau, r, v, a

def _find_retarded_time_for_pair(ri_t, t, times, pos_j, c):
    """
    Bracketing dell'ultima radice f(τ)=t-τ-||ri(t)-rj(τ)||/c con f monotona decrescente.
    Cerca k tale che f[k] >= 0 e f[k+1] <= 0, poi fa bisezione su s in [0,1].
    Restituisce (k, s) oppure None se la storia non copre la radice.
    """
    if len(times) < 2:
        return None

    R = ri_t[None, :] - pos_j                  # (M,3)
    d = np.linalg.norm(R, axis=1)              # (M,)
    f = (t - times) - d / c                    # (M,)

    # indici di cambiamento di segno: positivo->negativo (>=0 poi <=0)
    sign_change = np.where((f[:-1] >= 0.0) & (f[1:] <= 0.0))[0]
    if sign_change.size == 0:
        return None
    k = int(sign_change[-1])  # l'ultimo vicino a t

    # bisezione su s in [0,1] con interpolazione lineare di rj
    tau0, tau1 = times[k], times[k+1]
    r0, r1 = pos_j[k], pos_j[k+1]

    s_lo, s_hi = 0.0, 1.0
    for _ in range(20):
        s = 0.5 * (s_lo + s_hi)
        tau = tau0 + s * (tau1 - tau0)
        rj_s = (1.0 - s) * r0 + s * r1
        f_s = (t - tau) - np.linalg.norm(ri_t - rj_s) / c
        if f_s >= 0.0:
            s_lo = s
        else:
            s_hi = s
    s = 0.5 * (s_lo + s_hi)
    return k, s
# ---------- 3) Campi LW e forza totale ----------
def _lw_fields_at_i_from_j(ri_t, t, times, pos_j, vel_j, qi, qj, vi_t, eps_r, kappa_floor=1e-8):
    """
    Calcola E_j(ri,t) e B_j(ri,t) via Liénard–Wiechert (interp. lineare).
    Restituisce (E,B) oppure (None,None) se non si trova il tempo ritardato.
    """
    c = phy.c
    eps0 = phy.epsilon_0
    k_pref = 1.0 / (4.0 * np.pi * eps0 * eps_r)

    pair = _find_retarded_time_for_pair(ri_t, t, times, pos_j, c)
    if pair is None:
        return None, None
    k, s = pair

    # Interpola rj, vj, aj al tempo ritardato
    tau, rj, vj, aj = _interp_state(times, pos_j, vel_j, k, s)

    R = ri_t - rj
    R_norm = np.linalg.norm(R)
    if R_norm == 0.0:
        return None, None
    n = R / R_norm

    beta  = vj / c
    beta2 = np.dot(beta, beta)
    if beta2 >= 1.0:
        beta = beta / (np.sqrt(beta2) * (1 + 1e-12))  # clamp
        beta2 = np.dot(beta, beta)
    gamma = 1.0 / np.sqrt(1.0 - beta2)
    betadot = aj / c

    kappa = 1.0 - np.dot(n, beta)
    if kappa < kappa_floor:
        kappa = kappa_floor  # evita esplosioni numeriche (beaming quasi tangente)

    # Termini LW
    # Near-field (1/R^2) con fattore 1/gamma^2
    term_nf = (n - beta) / (gamma*gamma * kappa**3 * R_norm*R_norm)
    # Radiation (1/R)
    term_rad = np.cross(n, np.cross(n - beta, betadot)) / (kappa**3 * c * R_norm)

    E = k_pref * qj * (term_nf + term_rad)
    B = 1.0 / phy.c * np.cross(n, E)  # LW: sempre B = n x E
    return E, B

def forces_LW_pairwise(
    t: float,
    pos: np.ndarray,   # (N,3) at time t
    vel: np.ndarray,   # (N,3) at time t
    q:   np.ndarray,   # (N,)
    hist: HistoryBuffer,
    epsilon_r: float = 1.0,
    R_max: float | None = None,  # opzionale cutoff "pratico" (sconsigliato per accuratezza radiativa)
) -> np.ndarray:
    """
    Forze totali sui N corpi con campi Liénard–Wiechert (retardati) + Lorentz.
    Complessità O(N^2) (senza cutoff). Non applica azione-reazione.
    """
    assert hist.valid(), "HistoryBuffer vuoto: fai almeno una push() per inizializzare."

    N = pos.shape[0]
    F = np.zeros_like(pos)

    # Precarica le storie di tutti i j in RAM una volta
    times_all = []
    pos_all   = []
    vel_all   = []
    for j in range(N):
        tj, pj, vj = hist.get_arrays_unrolled(j)
        times_all.append(tj)
        pos_all.append(pj)
        vel_all.append(vj)

    for i in range(N):
        ri_t = pos[i]
        vi_t = vel[i]
        qi   = q[i]

        for j in range(N):
            if j == i:
                continue

            if R_max is not None:
                # controllo grezzo con la posizione attuale di j (non ritardata): può scartare troppo poco o troppo
                if np.linalg.norm(pos[j] - ri_t) > R_max:
                    continue

            E, B = _lw_fields_at_i_from_j(
                ri_t, t, times_all[j], pos_all[j], vel_all[j],
                qi, q[j], vi_t, epsilon_r
            )
            if E is None:
                continue
            F[i] += qi * (E + np.cross(vi_t, B))

    return F


def _find_retarded_state_for_pair(times: np.ndarray,
                                  pos_j: np.ndarray,
                                  vel_j: np.ndarray,
                                  r_i_now: np.ndarray,
                                  t_now: float,
                                  c: float) -> tuple[float, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Trova τ in [t_k, t_{k+1}] s.t. f(τ)=t_now-τ - |r_i - r_j(τ)|/c = 0
    usando bisezione su s in [0,1] con interpolazione lineare tra k e k+1.
    Ritorna (tau, r_tau, v_tau, a_tau) oppure None se non esiste bracket.
    """
    # usa SOLO storia <= t_now
    mask = times <= t_now
    times = times[mask]; pos_j = pos_j[mask]; vel_j = vel_j[mask]
    if times.size < 2:
        return None

    # f_k sui nodi registrati
    R = np.linalg.norm(r_i_now - pos_j, axis=1)
    f = (t_now - times) - (R / c)

    # cerca l'ultimo intervallo con f[k] >= 0 e f[k+1] <= 0 (radice "vicina" al presente)
    idx = None
    for k in range(len(times) - 1):
        if f[k] >= 0.0 and f[k + 1] <= 0.0:
            idx = k
    if idx is None:
        return None

    t0, t1 = float(times[idx]), float(times[idx + 1])
    r0, r1 = pos_j[idx].astype(float), pos_j[idx + 1].astype(float)
    v0, v1 = vel_j[idx].astype(float), vel_j[idx + 1].astype(float)
    dt_seg = t1 - t0
    if dt_seg <= 0.0:
        return None

    # bisezione su s in [0,1] (τ = t0 + s*dt_seg)
    lo, hi = 0.0, 1.0
    for _ in range(30):  # 2^-30 ~ 1e-9: più che sufficiente
        s = 0.5 * (lo + hi)
        tau = t0 + s * dt_seg
        r_tau = r0 + s * (r1 - r0)
        v_tau = v0 + s * (v1 - v0)
        f_mid = (t_now - tau) - (np.linalg.norm(r_i_now - r_tau) / c)
        if f_mid > 0.0:
            lo = s
        else:
            hi = s
    s = 0.5 * (lo + hi)
    tau = t0 + s * dt_seg
    r_tau = r0 + s * (r1 - r0)
    v_tau = v0 + s * (v1 - v0)
    a_tau = (v1 - v0) / dt_seg
    return tau, r_tau, v_tau, a_tau


def _lw_EB_from_source(qj: float,
                       r_i: np.ndarray,
                       r_tau: np.ndarray,
                       v_tau: np.ndarray,
                       a_tau: np.ndarray,
                       epsilon_r: float,
                       kappa_floor: float = 1e-6,
                       R_floor: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """
    Campi LW di j su i al tempo t (già risolto τ e stato ritardato).
    """
    R_vec = r_i - r_tau
    R = float(np.linalg.norm(R_vec))
    if R < R_floor:
        return np.zeros(3), np.zeros(3)
    n = R_vec / R
    beta = v_tau / phy.c
    beta2 = float(np.dot(beta, beta))
    if beta2 >= 1.0:
        beta = beta / (np.sqrt(beta2) * 1.000001)
        beta2 = float(np.dot(beta, beta))
    gamma = 1.0 / np.sqrt(1.0 - beta2)
    kappa = 1.0 - float(np.dot(n, beta))
    if kappa < kappa_floor:
        kappa = kappa_floor

    pref = qj / (4.0 * np.pi * phy.epsilon_0 * float(epsilon_r))
    term_near = (n - beta) / (gamma**2 * (kappa**3) * (R**2))
    term_rad  = np.cross(n, np.cross(n - beta, a_tau / phy.c)) / (kappa**3 * R)
    E = pref * (term_near + term_rad)
    B = 1.0 / phy.c * np.cross(n, E)
    return E, B


def fields_LW_pairwise(t_now: float,
                       pos_now: np.ndarray,   # shape (N,3) posizioni target (t_now)
                       vel_now: np.ndarray,   # shape (N,3) (serve solo per |F| dopo; Boris usa E,B)
                       q:       np.ndarray,   # shape (N,)
                       hist:    'HistoryBuffer',
                       epsilon_r: float = 1.0,
                       R_max: float | None = None,
                       kappa_floor: float = 1e-6,
                       R_floor: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """
    Ritorna (E_pp, B_pp) con SOLO contributi particle-particle ritardati (LW).
    E_pp, B_pp: array (N,3).
    """
    N = pos_now.shape[0]
    E_pp = np.zeros((N, 3), dtype=float)
    B_pp = np.zeros((N, 3), dtype=float)

    # Per efficienza, estraiamo una volta sola la storia "unrolled" di ciascun j
    caches = []
    for j in range(N):
        times_j, pos_j, vel_j = hist.get_arrays_unrolled(j)
        caches.append((times_j, pos_j, vel_j))

    for i in range(N):
        ri = pos_now[i]
        for j in range(N):
            if j == i:
                continue
            # (opzionale) pre-taglio rozzo sulla distanza presente per velocizzare
            if R_max is not None:
                if np.linalg.norm(pos_now[i] - pos_now[j]) > R_max:
                    continue

            times_j, pos_j, vel_j = caches[j]
            ret = _find_retarded_state_for_pair(times_j, pos_j, vel_j, ri, t_now, phy.c)
            if ret is None:
                continue  # nessun contributo disponibile (storia insufficiente o t troppo piccolo)
            tau, r_tau, v_tau, a_tau = ret
            E_ij, B_ij = _lw_EB_from_source(q[j], ri, r_tau, v_tau, a_tau,
                                            epsilon_r=epsilon_r,
                                            kappa_floor=kappa_floor,
                                            R_floor=R_floor)
            E_pp[i] += E_ij
            B_pp[i] += B_ij

    return E_pp, B_pp





if __name__ == "__main__":
    # Parametri test
    N  = 4
    dt = 1e-12          # 1 ps
    steps_hist = 200    # storia: 200 ps
    eps_r = 1.0

    # Stato iniziale (t=0): due cariche a 1 cm, velocità costanti
    q = np.array([+1.0e-9, -1.0e-9, +1.0e-9, -1.0e-9])  # Coulomb grandi per vedere numeri
    pos0 = np.array([[0.0,      0.0, 0.0],
                     [0.01,     0.0, 0.0],
                     [0.02,     0.0, 0.0],
                     [-0.01,     0.0, 0.0]])   # 1 cm di separazione
    vel  = np.array([[0.0,  0.30*phy.c, 0.0],  # 0.3c lungo +y
                     [0.0, -0.10*phy.c, 0.0],
                     [0.0, -0.40*phy.c, 0.0],
                     [0.0, -0.01*phy.c, 0.0]]) # 0.1c lungo -y

    # Costruisci HistoryBuffer e riempi con traiettorie a velocità costante
    hist = HistoryBuffer(N, steps_hist, dt)
    pos = pos0.copy()
    t = 0.0
    for k in range(steps_hist):
        # traiettoria semplice: r(t) = r0 + v * t (N.B. zero accelerazione)
        pos = pos0 + vel * t
        hist.push(t, pos, vel)
        t += dt

    # Stato "corrente" (ultimo inserito)
    t_cur = hist.t[hist.head]
    pos_cur = hist.pos[:, hist.head, :].copy()
    vel_cur = hist.vel[:, hist.head, :].copy()

    # Calcola forze Liénard–Wiechert + Lorentz
    F = forces_LW_pairwise(t_cur, pos_cur, vel_cur, q, hist, epsilon_r=eps_r, R_max=None)

    # Stampa risultati
    np.set_printoptions(precision=6, suppress=True)
    print(f"t = {t_cur:.3e} s")

    for s in range(steps_hist):
        print(F[0][s])

    for i in range(N):
        print(f"F[{i}] = {F[i]}  |F| = {np.linalg.norm(F[i]):.6e} N")

        print("N =", N, ", steps_hist =", steps_hist)
        print("F shape:", F.shape, "(atteso: (N,3))")