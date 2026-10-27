"""Regression check: the refactored library must reproduce the v3 notebook's baseline numbers (seed 42 models, legacy scenarios)."""
import os, sys, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, pandas as pd
from mppt.config import CFG, CACHE, ROOT
from mppt.data import get_pv_series
from mppt.scenarios import create_scenarios
from mppt import models as MD
from mppt.nn import VrefPredictor
from mppt.evaluate import build_controllers, run_suite, metrics_table, CORE

if __name__ == '__main__':
    t0 = time.time()
    pv, pvm, _ = get_pv_series(); sc = create_scenarios(pv)
    v2, prot = MD.train_v2(pv, pvm)
    for k in v2: [v2[k][n].dump(os.path.join(CACHE, f'v2_{k}_{n}')) for n in v2[k] if hasattr(v2[k][n], 'dump')]
    data, cost = MD.hybrid_data(pv)
    hyb = MD.train_hybrid_dnn(data, seed=CFG['seed'], name='Hybrid DNN seed42'); hyb.trained.dump(os.path.join(CACHE, 'hybrid_seed42'))
    print('trained in', round(time.time() - t0), 's')
    ctr = build_controllers(ann=MD.v2_predictor(v2['S1']['complex'], 'ANN-MPPT'), poly=v2['S1']['poly2'], dnn_est=hyb.estimator, names=CORE)
    t1 = time.time(); logs = run_suite(sc, ctr, pvm); print('closed loop', round(time.time() - t1), 's')
    mt = metrics_table(logs, sc); mt.to_csv(os.path.join(CACHE, 'regress_v3_metrics.csv'), index=False)
    ref = pd.read_csv(os.path.join(ROOT, 'reference', 'v3', 'metrics_by_scenario.csv')).rename(columns={'algorithm': 'controller'})
    j = mt.merge(ref, on=['plant', 'scenario', 'controller'], suffixes=('_new', '_v3'))
    j['d_eff'] = j['tracking_efficiency_%_new'] - j['tracking_efficiency_%_v3']
    print(j.groupby('controller').d_eff.agg(['mean', 'min', 'max']).round(5))
    print('max |d_eff| overall:', j.d_eff.abs().max())
    print(j.loc[j.d_eff.abs() > 5e-3, ['plant', 'scenario', 'controller', 'tracking_efficiency_%_new', 'tracking_efficiency_%_v3']].to_string())
