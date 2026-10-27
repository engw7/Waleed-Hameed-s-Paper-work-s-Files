"""Environmental scenarios. seed=None reproduces the v3 scenarios exactly; seed=k gives a documented variant (see docs/PARAMETERS.md):
comprehensive: cloud-noise stream; step_change: step instant in [1200,2400] s; cloud_passing: random phase; shading: random order of the nine patterns;
real days: the k-th ranked clear / variable day of the held-out years (ranking rules fixed a priori); ramp_change is deterministic."""
import numpy as np, pandas as pd
from .config import CFG

PATTERNS = [[800, 800, 800], [800, 800, 400], [800, 800, 200], [800, 500, 500], [800, 600, 300], [1000, 1000, 300], [900, 700, 300], [700, 700, 700], [1000, 600, 200]]
STD4 = ['comprehensive', 'step_change', 'cloud_passing', 'ramp_change']
REAL = ['real_clear_day', 'real_variable_day']
ALL = ['step_change', 'cloud_passing', 'ramp_change', 'comprehensive', 'partial_shading', 'real_clear_day', 'real_variable_day']


def _ranked_days(pv):
    te = pv[pv.year > CFG['chrono']['calib_end']].copy(); te['date'] = te.t.dt.date
    agg = te.groupby('date')['G(i)'].agg(total='sum', var=lambda x: np.abs(np.diff(x.values)).sum())
    mon = pd.Series([d.month for d in agg.index], index=agg.index)
    clear_rank = agg[mon.isin([6, 7, 8])]['total'].sort_values(ascending=False).index.tolist()
    cand = agg[(agg.total >= 3000) & mon.isin(range(3, 12))]
    var_rank = (cand['var'] / cand['total']).sort_values(ascending=False).index.tolist()
    return te, clear_rank, var_rank


def create_scenarios(pv, seed=None):
    n = 3600; t = np.linspace(0, 3600, n); rs = lambda off: np.random.RandomState(off + (0 if seed is None else seed))
    noise = np.random.RandomState(42).normal(1, 0.1, n) if seed is None else rs(1000).normal(1, 0.1, n)
    phase = 0.0 if seed is None else rs(2000).uniform(0, 2 * np.pi)
    t_step = 1800 if seed is None else int(rs(4000).choice(np.arange(1200, 2401, 100)))
    sc = {}
    morning = 200 + 600 * (1 - np.cos(np.pi * t / 3600))
    sc['comprehensive'] = dict(G=np.clip(morning + 150 * np.sin(4 * np.pi * t / 3600) * noise, 50, 1200), T=15 + 20 * (t / 3600) + 3 * np.sin(6 * np.pi * t / 3600),
                               description='Varying irradiance with cloud effects and temperature changes')
    sc['step_change'] = dict(G=np.where(t < t_step, 400, 800).astype(float), T=np.full_like(t, 25.0), description=f'Irradiance step 400 -> 800 W/m2 at t = {t_step} s')
    sc['cloud_passing'] = dict(G=800 + 400 * np.sin(2 * np.pi * t / 600 + phase), T=25 + 5 * np.sin(2 * np.pi * t / 1800), description='Periodic cloud passing with slow temperature variation')
    sc['ramp_change'] = dict(G=300 + 600 * (t / 3600), T=20 + 15 * (t / 3600), description='Linear ramp of irradiance and temperature')
    for s in sc.values():
        s['time'] = t; s['events'] = [0]
    sc['step_change']['events'] = [0, int(np.searchsorted(t, t_step))]
    order = list(range(9)) if seed is None else rs(3000).permutation(9).tolist()
    pats = [PATTERNS[i] for i in order]
    G_sub = np.zeros((n, 3)); ev = []
    for k, pat in enumerate(pats):
        i0 = int(np.searchsorted(t, 400 * k)); ev.append(i0); G_sub[i0:] = pat
    sc['partial_shading'] = dict(time=t, G=G_sub.mean(axis=1), T=np.full_like(t, 25.0), G_sub=G_sub, events=ev, patterns=pats,
                                 description='Nine shading patterns (3 bypass-diode substrings), 400 s each; G input = mean substring irradiance')
    te, clear_rank, var_rank = _ranked_days(pv)
    k = 0 if seed is None else seed
    clear = clear_rank[k]
    variable = [d for d in var_rank if d != clear][k]
    for name, d in (('real_clear_day', clear), ('real_variable_day', variable)):
        day = te[(te.date == d) & (te['G(i)'] >= CFG['g_min'])]
        h = np.arange(len(day))
        sc[name] = dict(time=t, G=np.interp(np.linspace(0, len(day) - 1, n), h, day['G(i)'].values), T=np.interp(np.linspace(0, len(day) - 1, n), h, day['T2m'].values), events=[0],
                        description=f'Held-out PVGIS day {d} (daylight hours, {len(day)} hourly samples interpolated to {n} cycles)', date=str(d))
    if CFG['quick']:
        nq = CFG['quick_steps']
        for s in sc.values():
            for key in ('time', 'G', 'T', 'G_sub'):
                if key in s: s[key] = s[key][:nq]
            s['events'] = [e for e in s['events'] if e < nq]
            if 'patterns' in s: s['patterns'] = s['patterns'][:len(s['events'])]
    return sc


def startup_scenario(G=None, T=None, n=None, seed=None):
    G, T = (CFG['startup_cond'] if G is None else (G, T)); n = CFG['startup_steps'] if n is None else n
    return dict(time=np.arange(n, dtype=float), G=np.full(n, G), T=np.full(n, T), events=[0])
