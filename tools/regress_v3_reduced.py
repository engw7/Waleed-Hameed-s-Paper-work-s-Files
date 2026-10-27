"""Reduced regression gate: cached models (no retraining), all 7 scenarios, both plants. Non-ML controllers must reproduce v3 exactly; ML controllers use freshly trained seed-42 models, so only a tolerance applies."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, pandas as pd
from mppt.config import CFG, CACHE, ROOT
from mppt.data import get_pv_series
from mppt.scenarios import create_scenarios
from mppt import models as MD
from mppt.nn import VrefPredictor
from mppt.evaluate import build_controllers, run_suite, metrics_table, PO, IC, AN, SC, DO, HY

if __name__ == '__main__':
    t0 = time.time(); pv, pvm, _ = get_pv_series(); sc = create_scenarios(pv)
    hyb = MD.Trained.load(os.path.join(CACHE, 'hybrid_seed42')); ann = MD.Trained.load(os.path.join(CACHE, 'v2_S1_complex'))
    from mppt.nn import ScanEstimator
    ctr = build_controllers(ann=MD.v2_predictor(ann, 'ANN-MPPT'), dnn_est=ScanEstimator('dnn', hyb.mlp), names=[PO, IC, AN, SC, DO, HY])
    logs = run_suite(sc, ctr, pvm); mt = metrics_table(logs, sc); mt.to_csv(os.path.join(ROOT, 'logs', 'regress_reduced_metrics.csv'), index=False)
    ref = pd.read_csv(os.path.join(ROOT, 'reference', 'v3', 'metrics_by_scenario.csv')).rename(columns={'algorithm': 'controller', 'shading_success_%': 'gmpp_ref'})
    j = mt.merge(ref, on=['plant', 'scenario', 'controller'], suffixes=('_new', '_v3'))
    j['d_eff'] = j['tracking_efficiency_%_new'] - j['tracking_efficiency_%_v3']
    print('seconds', round(time.time() - t0)); print(j.groupby('controller').d_eff.agg(['mean', 'min', 'max']).round(5))
    s = j[j.scenario == 'partial_shading'][['plant', 'controller', 'gmpp_success_%', 'gmpp_ref']]; print(s.to_string())
    j.to_csv(os.path.join(ROOT, 'logs', 'regress_reduced_join.csv'), index=False)
