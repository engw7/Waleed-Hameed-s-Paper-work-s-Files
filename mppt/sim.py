"""Closed-loop simulator: runs one controller on one scenario/plant and returns the per-cycle log DataFrame."""
import numpy as np, pandas as pd
from .config import CFG


def run_algorithm(alg, plant, scn, truth, noise_rng=None):
    """noise_rng: if given (with CFG['noise']) the controller receives noisy V, I and irradiance; the log always holds TRUE delivered values."""
    noisy = noise_rng is not None
    if noisy:
        nz = CFG['noise']; plant = __import__('mppt.plants', fromlist=['x']).SensorNoisePlant(plant, noise_rng, nz['sigma_v'], nz['sigma_i'], nz['sigma_g'])
    rec = []
    for i, (g, t) in enumerate(zip(scn['G'], scn['T'])):
        if 'G_sub' in scn:
            plant.set_shading(scn['G_sub'][i])
        if noisy:
            plant.begin_cycle(g, t); g_ctrl = plant.sense_G(g)
        else:
            g_ctrl = g
        r = alg.mppt_step(plant, g_ctrl, t, i)
        if hasattr(plant, 'extras'): r.update(plant.extras())
        if noisy:
            r['V_true_mean'], r['I_true_mean'], r['P_true_mean'] = plant.true_cycle()
            r['V'], r['I'], r['P'] = r['V_true_mean'], r['I_true_mean'], r['P_true_mean']
        rec.append(r)
    df = pd.DataFrame(rec)
    df['scan'] = df['scan'].fillna(0).astype(int) if 'scan' in df else 0
    df.insert(0, 'time', scn['time']); df.insert(1, 'G', scn['G']); df.insert(2, 'T', scn['T'])
    df['mpp_voltage'], df['mpp_power'] = truth[:, 0], truth[:, 1]
    df['efficiency'] = 100 * df.P / df.mpp_power
    df['duty'] = 1 - df.V_cmd / CFG['v_bus']                      # ideal-converter duty (display); the converter study computes the real duty
    df['compute_us'] = df.pop('compute_ns') / 1e3
    return df
