"""Real PVGIS-SARAH3 series (Ramadi) and the module outputs computed from it with the single-diode model."""
import os, re
import numpy as np, pandas as pd
from .config import CFG
from pv_single_diode import SingleDiodePV, mpp_batch


def load_pvgis(path=None):
    path = path or CFG['data_file']
    if not os.path.isfile(path):
        raise FileNotFoundError(f"'{path}' not found. This pipeline never generates substitute data.")
    lines = open(path).read().splitlines()
    h = next((i for i, l in enumerate(lines) if l.startswith('time,')), None)
    if h is None:
        raise ValueError('no PVGIS header line starting with "time," found')
    meta = {l.split(':')[0].strip(): l.split(':', 1)[1].strip() for l in lines[:h] if ':' in l}
    body = [l for l in lines[h + 1:] if re.match(r'^\d{8}:\d{4},', l)]
    df = pd.DataFrame([l.split(',') for l in body], columns=lines[h].split(',')).drop(columns=['time']).astype(float)
    df.insert(0, 't', pd.to_datetime([l[:13] for l in body], format='%Y%m%d:%H%M'))
    for c in ('G(i)', 'T2m', 'H_sun'):
        if c not in df.columns:
            raise ValueError(f'missing column {c}')
    if df[['G(i)', 'T2m']].isna().any().any():
        raise ValueError('NaNs in G(i)/T2m')
    df['month'], df['hour'], df['year'] = df.t.dt.month, df.t.dt.hour, df.t.dt.year
    return df, meta


def add_module_outputs(pv, pvm):
    Vm, Im, Pm, Tc = mpp_batch(pvm, pv['G(i)'].to_numpy(), pv['T2m'].to_numpy())
    pv = pv.copy(); pv['Vm'], pv['Im'], pv['P'], pv['Tcell'] = Vm, Im, Pm, Tc
    return pv


def audit(pv, pvm):
    a = {'n_rows': int(len(pv))}
    exp = pd.date_range(pv.t.iloc[0], periods=len(pv), freq='h')
    a.update(n_duplicate_timestamps=int(pv.t.duplicated().sum()), time_axis_regular=bool((pv.t.values == exp.values).all()))
    night = pv['H_sun'] <= 0
    a.update(frac_night=float(night.mean()), G_positive_at_night=int(((pv['G(i)'] > 0) & night).sum()), T_night_mean=float(pv.loc[night, 'T2m'].mean()), T_day_mean=float(pv.loc[~night, 'T2m'].mean()),
             T_min=float(pv['T2m'].min()), T_max=float(pv['T2m'].max()), G_max=float(pv['G(i)'].max()))
    day = pv['G(i)'] >= CFG['g_min']
    k = pv.loc[day, ['G(i)', 'T2m']].round(2)
    a.update(months_covered=int(pv.month.nunique()), min_daytime_rows_per_month=int(pv[day].groupby('month').size().min()),
             n_distinct_GT_all=int(pv[['G(i)', 'T2m']].round(2).drop_duplicates().shape[0]), n_distinct_GT_daytime=int(len(k.value_counts())), n_daytime_rows=int(day.sum()),
             max_repeats_of_a_daytime_pair=int(k.value_counts().max()), frac_daytime_rows_in_repeated_pairs=float(k.duplicated(keep=False).mean()))
    if 'P' in pv:
        a.update(P_max_when_G_zero=float(pv.loc[pv['G(i)'] == 0, 'P'].max()), P_equals_VmIm=bool(np.allclose(pv.P, pv.Vm * pv.Im)), corr_G_P=float(np.corrcoef(pv['G(i)'], pv.P)[0, 1]))
    return a


def get_pv_series(use_cache=True):
    """(pv with module outputs, pvm, meta) -- cached because the batch MPP solve takes ~15 s."""
    from .config import CACHE
    pvm = SingleDiodePV()
    pv, meta = load_pvgis()
    cp = os.path.join(CACHE, 'pv_with_outputs.pkl')
    if use_cache and os.path.exists(cp):
        return pd.read_pickle(cp), pvm, meta
    pv = add_module_outputs(pv, pvm)
    pv.to_pickle(cp)
    return pv, pvm, meta
