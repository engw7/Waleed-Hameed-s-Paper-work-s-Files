"""Vectorised, parallel generator of (plant variant x partial-shading pattern x real (G,T)) training/evaluation samples.

Physics
  * Module: five-parameter single-diode model [De Soto, Klein, Beckman, Solar Energy 80 (2006) 78-88; Duffie & Beckman 2013],
    explicit inverse via the Lambert W function [Jain & Kapoor, Solar Energy Materials & Solar Cells 81 (2004) 269-277]:
      V(I) = Rsh (Iph + I0) - I (Rs + Rsh) - a W( I0 Rsh / a * exp( Rsh (Iph + I0 - I) / a ) ),   a = n N k T / q
  * Partial shading: standard bypass-diode series-string approach [Patel & Agarwal, IEEE Trans. Energy Conversion 23(1) (2008) 302-310;
    Kawamura et al. 2003]: the module is 3 substrings of N_sub = Ns/3 cells, each with a bypass diode (V_sub >= -0.7 V); inside a
    substring a group of m cells receives the reduced irradiance f*G, the other cells G; group voltages add at a common current;
    substring voltages add; the I-V curve of the series string is obtained by sweeping the common current.
  * Domain randomisation: each sample draws its own plant (Isc, Voc, Rs, Rsh, n) from manufacturing-tolerance / aging ranges;
    Iph and I0 follow in closed form from the datasheet conditions (Isc, Voc) for the drawn Rs, Rsh, n.
"""
import numpy as np
from scipy.special import lambertw

K_B, Q_E, T_REF = 1.380649e-23, 1.602176634e-19, 298.15
EG = 1.121 * Q_E
NOCT_COEF = (45.0 - 20.0) / 800.0
V_BYPASS = 0.7
NS, N_SUB = 54, 3
MU_ISC = 0.0006 * 9.87
PROBES = np.linspace(8.0, 35.0, 14)           # fixed probe voltages of the coarse I-V scan (V)
V_WINDOW = (8.0, 35.0)

# domain-randomisation boxes (multipliers on the nominal datasheet extraction, or absolute for Voc / n)
DR_BOX = dict(isc=(0.90, 1.05), voc=(0.95, 1.06), rs=(0.8, 2.4), rsh=(0.3, 1.5), n=(1.05, 1.40))
# OUT-OF-range box used only for held-out evaluation
OOR_BOX = dict(isc=(0.78, 0.86), voc=(1.08, 1.12), rs=(2.8, 3.8), rsh=(0.12, 0.25), n=(1.45, 1.55))
NOMINAL = dict(isc=9.87, voc=32.9, rs=0.18203422039485764, rsh=400.0, n=1.2)


def w_exp_fast(x):
    """Lambert W of exp(x) for real x (vectorised, no complex arithmetic): Newton on w + ln w = x for x > 1, Halley on w e^w = e^x otherwise."""
    x = np.asarray(x, float)
    out = np.empty_like(x)
    big = x > 1.0
    xb = x[big]
    w = np.maximum(xb - np.log(np.maximum(xb, 1.0001)), 1.0)
    for _ in range(8):
        w = w - (w + np.log(w) - xb) / (1.0 + 1.0 / w)
    out[big] = w
    xs = x[~big]
    y = np.exp(np.maximum(xs, -700.0))
    L = np.log1p(y)
    w = L / (1.0 + 0.5 * L)
    for _ in range(10):
        ew = np.exp(w)
        f = w * ew - y
        w = w - f / (ew * (w + 1.0) - (w + 2.0) * f / (2.0 * w + 2.0))
    out[~big] = np.where(y < 1e-300, y, w)
    return out


def plant_from_dr(rng, n, box):
    """Draw n plant variants. Returns raw single-diode parameter arrays (Iph_ref, I0_ref, Rs, Rsh_ref, n)."""
    isc = NOMINAL['isc'] * rng.uniform(*box['isc'], n)
    voc = NOMINAL['voc'] * rng.uniform(*box['voc'], n) if box['voc'][1] < 2 else rng.uniform(*box['voc'], n)
    rs = NOMINAL['rs'] * rng.uniform(*box['rs'], n)
    rsh = NOMINAL['rsh'] * rng.uniform(*box['rsh'], n)
    nn = rng.uniform(*box['n'], n)
    a = nn * NS * K_B * T_REF / Q_E
    i0 = (isc * (1 + rs / rsh) - voc / rsh) / (np.expm1(voc / a) - np.expm1(isc * rs / a))
    iph = i0 * np.expm1(voc / a) + voc / rsh
    return dict(iph=iph, i0=i0, rs=rs, rsh=rsh, n=nn)


def plant_nominal(n, severity=0.0):
    """Nominal plant, optionally perturbed exactly like SingleDiodePV.perturbed(severity)."""
    p = plant_from_dr(np.random.RandomState(0), 1, dict(isc=(1, 1), voc=(1, 1), rs=(1, 1), rsh=(1, 1), n=(1.2, 1.2)))
    q = {k: np.full(n, float(v[0])) for k, v in p.items()}
    q['iph'] *= 1 - 0.08 * severity; q['rs'] *= 1 + 0.8 * severity; q['rsh'] *= 1 - 0.4 * severity; q['n'] *= 1 + 0.05 * severity
    return q


def sample_patterns(rng, n, mode):
    """Shading patterns: m[:,k] shaded cells in substring k (0..18), f[:,k] irradiance factor of the shaded cells (0.1..0.85)."""
    m = np.zeros((n, N_SUB), int); f = np.ones((n, N_SUB))
    if mode == 'none':
        return m, f
    nsub = NS // N_SUB
    shaded = rng.uniform(size=(n, N_SUB)) < 0.5
    if mode == 'always':
        none = ~shaded.any(axis=1)
        shaded[none, rng.randint(0, N_SUB, none.sum())] = True
    if mode == 'mix':
        on = rng.uniform(size=n) < 0.6                       # 40 % of samples unshaded
        shaded &= on[:, None]
        none = on & ~shaded.any(axis=1)
        shaded[none, rng.randint(0, N_SUB, none.sum())] = True
    full = rng.uniform(size=(n, N_SUB)) < 0.5                # whole substring vs only some cells
    cells = np.where(full, nsub, rng.randint(1, nsub, (n, N_SUB)))
    m = np.where(shaded, cells, 0)
    f = np.where(shaded, rng.uniform(0.10, 0.85, (n, N_SUB)), 1.0)
    return m, f


def _group_voltage(I, G_g, T_amb, N_g, P):
    """Voltage of a group of N_g cells (per-sample arrays, shape (n,1)) at currents I (n, NI). Zero where N_g = 0."""
    Tc = T_amb + NOCT_COEF * G_g
    Tk = Tc + 273.15
    Gs = np.maximum(G_g, 1e-3)
    Iph = (Gs / 1000.0) * (P['iph'] + MU_ISC * (Tk - T_REF))
    I0 = P['i0'] * (Tk / T_REF) ** 3 * np.exp(EG / (P['n'] * K_B) * (1 / T_REF - 1 / Tk))
    a = P['n'] * np.maximum(N_g, 1) * K_B * Tk / Q_E
    Rsh = P['rsh'] * 1000.0 / Gs * np.maximum(N_g, 1) / NS
    Rs = P['rs'] * np.maximum(N_g, 1) / NS
    arg = np.log(I0 * Rsh / a) + Rsh * (Iph + I0 - I) / a
    V = Rsh * (Iph + I0) - I * (Rs + Rsh) - a * w_exp_fast(arg)
    return np.where(N_g > 0, V, 0.0), Iph


def string_curve(G, T, P, m, f, NI=600, I_grid=None):
    """Series-string I-V curve of the shaded module. Returns I_grid (n,NI), V (n,NI)."""
    n = len(G)
    G, T = G[:, None], T[:, None]
    nsub = NS // N_SUB
    Pc = {k: v[:, None] for k, v in P.items()}
    groups, iph_max = [], np.zeros((n, 1))
    for k in range(N_SUB):
        mk = m[:, k:k + 1]
        groups.append((G, nsub - mk)); groups.append((G * f[:, k:k + 1], mk))
    for Gg, Ng in groups:
        Tk = T + NOCT_COEF * Gg + 273.15
        iph = np.maximum(Gg, 1e-3) / 1000.0 * (Pc['iph'] + MU_ISC * (Tk - T_REF))
        iph_max = np.maximum(iph_max, np.where(Ng > 0, iph, 0.0))
    if I_grid is None:
        u = np.linspace(0.0, 1.0, NI)[None, :]
        I_grid = iph_max * 1.03 * np.sqrt(u)                  # denser near the knee where the maxima are
    V = np.zeros_like(I_grid)
    for k in range(N_SUB):
        Va, _ = _group_voltage(I_grid, groups[2 * k][0], T, groups[2 * k][1], Pc)
        Vb, _ = _group_voltage(I_grid, groups[2 * k + 1][0], T, groups[2 * k + 1][1], Pc)
        V += np.maximum(Va + Vb, -V_BYPASS)
    return I_grid, V


def global_mpp(I, V, G, T, P, m, f, window=V_WINDOW, refine=True):
    """Global MPP inside the converter window; optional local refinement on a fine current grid around the grid maximum."""
    Pw = np.where((V >= window[0]) & (V <= window[1]), V * I, -np.inf)
    k = np.clip(Pw.argmax(axis=1), 1, I.shape[1] - 2)
    r = np.arange(len(k))
    if refine:
        lo, hi = I[r, k - 1], I[r, k + 1]
        Ifine = lo[:, None] + (hi - lo)[:, None] * np.linspace(0, 1, 41)[None, :]
        _, Vf = string_curve(G, T, P, m, f, I_grid=Ifine)
        Pf = np.where((Vf >= window[0]) & (Vf <= window[1]), Vf * Ifine, -np.inf)
        j = Pf.argmax(axis=1)
        return Vf[r, j], Pf[r, j], Ifine[r, j]
    return V[r, k], Pw[r, k], I[r, k]


def simulate_chunk(args):
    """One worker task: draw n samples, build their I-V curves, return features / labels (top-level => picklable for 'spawn')."""
    seed, n, base_G, base_T, plant_mode, shade_mode, store_pv = args
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(base_G), n)
    G, T = base_G[idx], base_T[idx]
    if plant_mode == 'dr':
        P = plant_from_dr(rng, n, DR_BOX)
    elif plant_mode == 'oor':
        P = plant_from_dr(rng, n, OOR_BOX)
    elif plant_mode.startswith('nominal'):
        P = plant_nominal(n, float(plant_mode.split(':')[1]) if ':' in plant_mode else 0.0)
    else:
        raise ValueError(plant_mode)
    m, f = sample_patterns(rng, n, shade_mode)
    I, V = string_curve(G, T, P, m, f)
    Vg, Pg, Ig = global_mpp(I, V, G, T, P, m, f)
    Ip = np.empty((n, len(PROBES)))
    for i in range(n):                                        # I at the probe voltages: interpolate the monotone V(I) curve
        Ip[i] = np.interp(PROBES, V[i, ::-1], I[i, ::-1], left=I[i, -1], right=0.0)
    G_in = ((NS // N_SUB - m) * 1.0 + m * f).sum(axis=1) / NS * G        # cell-weighted mean irradiance (what a mean sensor sees)
    out = dict(G_in=G_in, G_base=G, T=T, I_probe=Ip, V_gmpp=Vg, P_gmpp=Pg, shaded=(m > 0).any(axis=1), n_shaded_cells=m.sum(axis=1),
               min_f=np.where(m > 0, f, 1.0).min(axis=1), iph=P['iph'], rs=P['rs'], rsh=P['rsh'], nn=P['n'])
    if store_pv:
        Ve = np.linspace(V_WINDOW[0], V_WINDOW[1], 120)
        Pe = np.empty((n, len(Ve)))
        for i in range(n):
            Pe[i] = Ve * np.interp(Ve, V[i, ::-1], I[i, ::-1], left=I[i, -1], right=0.0)
        out['V_eval'], out['P_eval'] = np.tile(Ve, (n, 1)), Pe
    return out


def generate(n_total, base_G, base_T, plant_mode, shade_mode, seed, n_workers=1, chunk=1500, store_pv=False):
    """Generate n_total samples. n_workers>1 => real multiprocessing ('spawn' context, ordered results => deterministic)."""
    sizes = [chunk] * (n_total // chunk) + ([n_total % chunk] if n_total % chunk else [])
    tasks = [(seed * 100003 + j, s, base_G, base_T, plant_mode, shade_mode, store_pv) for j, s in enumerate(sizes)]
    if n_workers > 1:
        import multiprocessing as mp
        with mp.get_context('spawn').Pool(n_workers) as pool:
            res = pool.map(simulate_chunk, tasks, chunksize=1)
    else:
        res = [simulate_chunk(t) for t in tasks]
    return {k: np.concatenate([r[k] for r in res]) for k in res[0]}
