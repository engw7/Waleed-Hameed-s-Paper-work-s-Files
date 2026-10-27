"""Main evaluation (nine controllers, seven scenarios, nominal + mismatched plant) and the extra comparators PSO and Scan+Trees (experiment 2.2).
Models: ANN-MPPT and Poly2 on the v2 training protocol S1, stage-1 DNN seed 42 (design fixed before any test-year result), gradient-boosted trees and ridge on the same
16 inputs/data as the DNN. Outputs (all under results/): metrics_by_scenario.csv, metrics_headline.csv, metrics_coldstart.csv, mismatch_sweep.csv, metrics_shading_episodes.csv,
baselines/estimator_offline.csv, baselines/baselines.csv, hybrid_offline_metrics.csv, hybrid_conformal_coverage.csv, tables/Table_*.  Run logs are pickled to cache/main_logs.pkl for the figures."""
import os, sys, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, pandas as pd

def get_models(pv, pvm, data):
    from mppt.config import CFG, CACHE
    from mppt import models as MD
    from mppt.nn import ScanEstimator
    p = os.path.join(CACHE, 'v2_S1')
    protocols, _rows = MD.v2_protocols(pv, pvm); tr = protocols['S1']['train']
    if os.path.exists(p + '_complex.keras'):
        ann = MD.Trained.load(p + '_complex')
    else:
        v2, _ = MD.train_v2(pv, pvm, protocols=({'S1': protocols['S1']}, _rows)); ann = v2['S1']['complex']; ann.dump(p + '_complex')
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.linear_model import Ridge
    from mppt.nn import PolyPredictor
    poly = PolyPredictor('Poly2(G,T) S1', make_pipeline(StandardScaler(), PolynomialFeatures(2), Ridge(alpha=1e-6)).fit(tr[['G', 'T']].to_numpy(float), tr.V_mpp.to_numpy(float)))
    hp = os.path.join(CACHE, 'hybrid_seed42')
    if os.path.exists(hp + '.keras'): hyb = MD.Trained.load(hp)
    else: hyb = MD.train_hybrid_dnn(data, seed=CFG['seed'], name='Hybrid DNN seed42').trained; hyb.dump(hp)
    return ann, poly, hyb


def conformal(hyb, data):
    from mppt.config import CFG
    from mppt import models as MD
    feat = lambda d: MD.hybrid_features(d, True)
    pred = lambda d: np.clip(hyb.mlp.predict(feat(d)), CFG['v_min_op'], CFG['v_max_op'])
    res_c = np.abs(data['calib']['V_gmpp'] - pred(data['calib'])); rows = []
    for alpha in CFG['conformal_alpha']:
        n = len(res_c); q = float(np.quantile(res_c, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method='higher'))
        for k in ('T1', 'T2', 'T3', 'T4'):
            d = data[k]; r = np.abs(d['V_gmpp'] - pred(d))
            rows.append({'test_set': MD.TEST_NAMES[k], 'nominal': 1 - alpha, 'empirical_coverage': float(np.mean(r <= q)), 'interval_half_width_V': q, 'n': int(r.size)})
            if k == 'T1' and 'shaded' in d:
                for lab, m in (('T1 | shaded', d['shaded']), ('T1 | unshaded', ~d['shaded'])):
                    if m.sum() > 20: rows.append({'test_set': lab, 'nominal': 1 - alpha, 'empirical_coverage': float(np.mean(r[m] <= q)), 'interval_half_width_V': q, 'n': int(m.sum())})
    return pd.DataFrame(rows)


def headline(mt, cs, names):
    from mppt.scenarios import STD4, REAL
    rows = []
    for c in names:
        r = {'controller': c}
        for plant in ('nominal', 'mismatched'):
            m = mt[(mt.plant == plant) & (mt.controller == c)]
            r[f'eff_std4_{plant}_%'] = m[m.scenario.isin(STD4)]['tracking_efficiency_%'].mean(); r[f'eff_real_days_{plant}_%'] = m[m.scenario.isin(REAL)]['tracking_efficiency_%'].mean()
            sh = m[m.scenario == 'partial_shading'].iloc[0]; r[f'shading_eff_{plant}_%'] = sh['tracking_efficiency_%']; r[f'gmpp_success_{plant}_%'] = sh['gmpp_success_%']; r[f'wrong_peak_{plant}_%'] = sh['wrong_peak_%']
        n = mt[(mt.plant == 'nominal') & (mt.controller == c) & mt.scenario.isin(STD4)]
        r['event_convergence_cycles'] = n.convergence_cycles.mean(); r['osc_amp_V'] = n.osc_amp_V.mean(); r['osc_amp_P_%'] = n['osc_amp_P_%'].mean()
        r['scan_overhead_std4_%'] = n['scan_overhead_%'].mean(); r['compute_us_per_cycle'] = mt[(mt.plant == 'nominal') & (mt.controller == c)].compute_us_per_cycle.mean()
        b = cs[(cs.protocol == 'B') & (cs.controller == c)]; a = cs[(cs.protocol == 'A') & (cs.controller == c)]
        r['cold_start_B_conv_cycles'] = b.convergence_cycles.mean(); r['cold_start_B_not_converged'] = f'{int(b.convergence_cycles.isna().sum())}/{len(b)}'; r['cold_start_A_not_converged'] = f'{int(a.convergence_cycles.isna().sum())}/{len(a)}'
        rows.append(r)
    return pd.DataFrame(rows)


def run():
    from mppt.config import CFG, RESULTS, CACHE
    from mppt.data import get_pv_series
    from mppt.scenarios import create_scenarios
    from mppt import models as MD
    from mppt.nn import ScanEstimator
    from mppt.evaluate import build_controllers, run_suite, metrics_table, cold_start, ALL_CTRL
    from mppt.controllers import PerturbAndObserveMPPT
    from mppt.tables import save_table
    t0 = time.time(); pv, pvm, _ = get_pv_series(); sc = create_scenarios(pv); data, cost = MD.hybrid_data(pv)
    ann, poly, hyb = get_models(pv, pvm, data); trees = MD.train_trees(data, seed=0); ridge = MD.train_ridge(data)
    kw = dict(ann=MD.v2_predictor(ann, 'ANN-MPPT'), poly=poly, dnn_est=ScanEstimator('dnn', hyb.mlp), tree_est=trees.estimator, pso_seed=0)
    ctr = build_controllers(**kw); names = list(ctr)
    logs = run_suite(sc, ctr, pvm); mt = metrics_table(logs, sc); mt.to_csv(os.path.join(RESULTS, 'metrics_by_scenario.csv'), index=False)
    # cold start: every controller, nominal plant
    cs, cs_logs = cold_start(pvm, lambda v0: {n: f(v0) for n, f in build_controllers(**kw).items()}); cs.to_csv(os.path.join(RESULTS, 'metrics_coldstart.csv'), index=False)
    hl = headline(mt, cs, names); hl.to_csv(os.path.join(RESULTS, 'metrics_headline.csv'), index=False)
    # per-episode shading outcome (nominal plant)
    import metrics as M
    scn = sc['partial_shading']; ep = []
    for n in names:
        df = logs['nominal']['partial_shading'][n]; ev = scn['events'] + [len(df)]; flags = M.gmpp_success(df.P, df.mpp_power, df.scan, scn['events'])[1]
        for j in range(len(scn['events'])):
            seg = df.iloc[ev[j]:ev[j + 1]]; ep.append({'controller': n, 'episode': j + 1, 'substring_G': str(scn['patterns'][j]), 'efficiency_%': 100 * seg.P.sum() / seg.mpp_power.sum(), 'success': flags[j]})
    pd.DataFrame(ep).to_csv(os.path.join(RESULTS, 'metrics_shading_episodes.csv'), index=False)
    # mismatch severity sweep
    rows = []
    for sev in CFG['mismatch_sweep']:
        lg = run_suite({k: sc[k] for k in ('comprehensive', 'partial_shading')}, ctr, pvm, plants=((f'{sev}', sev),)); m = metrics_table(lg, sc)
        m['severity'] = sev; rows.append(m)
    pd.concat(rows).rename(columns={'plant': 'severity_label'}).to_csv(os.path.join(RESULTS, 'mismatch_sweep.csv'), index=False)
    # estimator comparison (offline, 2019-20) and conformal coverage
    off = pd.concat([MD.eval_hybrid(m, data).assign(estimator=n, params=m.n_params, macs=m.macs) for n, m in (('DNN (128,128,64), seed 42', MD.HybridModel('dnn', 'mlp', lambda X: hyb.mlp.predict(X), None, hyb.mlp.n_params(), hyb.mlp.macs(), hyb.seconds)),
                                                                                                           ('Gradient-boosted trees', trees), ('Ridge (linear)', ridge),
                                                                                                           ('Best-probe rule (no ML)', MD.HybridModel('probe', 'rule', lambda X: __import__('pv_hybrid_data').PROBES[np.argmax(np.asarray(X)[:, 2:] * __import__('pv_hybrid_data').PROBES, axis=1)], None, 0, 14, 0.0)))])
    off.to_csv(os.path.join(RESULTS, 'baselines', 'estimator_offline.csv'), index=False)
    MD.eval_hybrid(MD.HybridModel('dnn', 'mlp', lambda X: hyb.mlp.predict(X), None, hyb.mlp.n_params(), hyb.mlp.macs(), hyb.seconds), data).to_csv(os.path.join(RESULTS, 'hybrid_offline_metrics.csv'), index=False)
    conformal(hyb, data).to_csv(os.path.join(RESULTS, 'hybrid_conformal_coverage.csv'), index=False)
    pd.DataFrame([cost]).to_csv(os.path.join(RESULTS, 'data_generation_cost.csv'), index=False)
    base = hl[hl.controller.isin(['P&O', 'Scan+P&O (no ML)', 'Hybrid (DNN+P&O)', 'PSO', 'Tree scan+P&O'])]; base.to_csv(os.path.join(RESULTS, 'baselines', 'baselines.csv'), index=False)
    pickle.dump(dict(logs=logs, cold=cs_logs, scenarios=sc), open(os.path.join(CACHE, 'main_logs.pkl'), 'wb'))
    # tables
    short = {'eff_std4_nominal_%': 'Std. scen. eff. nominal (%)', 'eff_std4_mismatched_%': 'Std. scen. eff. mismatched (%)', 'shading_eff_nominal_%': 'Shading eff. nominal (%)', 'shading_eff_mismatched_%': 'Shading eff. mismatched (%)',
             'gmpp_success_nominal_%': 'GMPP success nominal (%)', 'event_convergence_cycles': 'Event conv. (cycles)', 'osc_amp_V': 'Osc. p-p (V)', 'compute_us_per_cycle': 'Compute (us/cycle)', 'cold_start_B_conv_cycles': 'Cold-start B (cycles)'}
    t = hl[['controller'] + list(short)].rename(columns=short); save_table('Table_Headline', t, 'Headline metrics of all controllers (quasi-static plant; nominal = model-matched, mismatched = perturbed plant, severity 1).', digits=2)
    from experiments.table_conceptual import build; build(mt)
    print('main evaluation done in', round(time.time() - t0), 's'); print(hl.round(3).to_string()); return hl


if __name__ == '__main__':
    run()
