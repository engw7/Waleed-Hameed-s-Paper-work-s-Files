"""Experiment 2.4-2.6: (a) multi-seed statistics with paired Wilcoxon tests, (b) sensor-noise robustness, (c) quantisation (float32 / float16 / int8).
Design (fixed before looking at results): 10 scenario seeds (cloud-noise stream, step instant, cloud phase, shading-pattern order, k-th ranked real test day); the stage-1 DNN is retrained
with 5 seeds (0-4) on the same data and paired with scenario seed s by d = s mod 5, so every Wilcoxon sample is one (scenario, DNN) unit and n = 10 (n = 5 can never reach p < 0.05).
Outputs under results/stats, results/noise, results/quantisation; Table_Stats, Table_Noise, Table_Quant; Fig32_seed_statistics, Fig33_noise, Fig34_quantisation."""
import os, sys, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, pandas as pd

_G = {}
def _init():
    from mppt.data import get_pv_series
    _G['pv'], _G['pvm'], _ = get_pv_series()


def _seed_job(args):
    s, mlp, ann_pred, poly, tree_est = args
    from mppt.evaluate import build_controllers, run_suite, metrics_table
    from mppt.scenarios import create_scenarios, STD4, REAL
    from mppt.nn import ScanEstimator
    sc = create_scenarios(_G['pv'], s); ctr = build_controllers(ann=ann_pred, poly=poly, dnn_est=ScanEstimator('dnn', mlp), tree_est=tree_est, pso_seed=s)
    mt = metrics_table(run_suite(sc, ctr, _G['pvm']), sc); mt['seed'] = s; mt['dnn_seed'] = s % 5; return mt


def get_seed_models(data, pv, pvm):
    from mppt.config import CFG, CACHE
    from mppt import models as MD
    out = {}
    for k in CFG['dnn_seeds']:
        p = os.path.join(CACHE, f'hybrid_s{k}')
        if os.path.exists(p + '.keras'): out[k] = MD.Trained.load(p)
        else: t = MD.train_hybrid_dnn(data, seed=k, name=f'DNN seed {k}').trained; t.dump(p); out[k] = t
    return out


def group_table(mt):
    from mppt.scenarios import STD4, REAL
    rows = []
    for (seed, plant, c), g in mt.groupby(['seed', 'plant', 'controller']):
        sh = g[g.scenario == 'partial_shading'].iloc[0]
        rows.append(dict(seed=seed, plant=plant, controller=c, eff_std4=g[g.scenario.isin(STD4)]['tracking_efficiency_%'].mean(), eff_real=g[g.scenario.isin(REAL)]['tracking_efficiency_%'].mean(),
                         eff_shading=sh['tracking_efficiency_%'], gmpp_success=sh['gmpp_success_%'], conv_cycles=g[g.scenario.isin(STD4)].convergence_cycles.mean()))
    return pd.DataFrame(rows)


def stats(models, data, pv, pvm, R, ann_pred, poly, tree_est):
    import multiprocessing as mp
    from mppt.config import CFG
    from scipy.stats import wilcoxon
    from mppt.tables import save_table
    import mppt_style as sty, matplotlib.pyplot as plt
    seeds = CFG['scenario_seeds']; jobs = [(s, models[s % 5].mlp if (s % 5) in models else models[sorted(models)[s % len(models)]].mlp, ann_pred, poly, tree_est) for s in seeds]
    t0 = time.time()
    with mp.get_context('spawn').Pool(max(1, min(len(jobs), CFG['n_workers'])), initializer=_init) as p: res = p.map(_seed_job, jobs, chunksize=1)
    mt = pd.concat(res); mt.to_csv(os.path.join(R, 'per_seed_scenario_metrics.csv'), index=False)
    gt = group_table(mt); gt.to_csv(os.path.join(R, 'per_seed_group_metrics.csv'), index=False)
    cols = ['eff_std4', 'eff_real', 'eff_shading', 'gmpp_success', 'conv_cycles']
    sm = gt.groupby(['plant', 'controller'])[cols].agg(['mean', 'std']); sm.columns = [f'{a}_{b}' for a, b in sm.columns]; sm = sm.reset_index(); sm['n'] = len(seeds); sm.to_csv(os.path.join(R, 'stats_summary.csv'), index=False)
    pairs = [('Hybrid (DNN+P&O)', 'Scan+P&O (no ML)'), ('Hybrid (DNN+P&O)', 'P&O'), ('Scan+P&O (no ML)', 'P&O'), ('Hybrid (DNN+P&O)', 'DNN scan-only (no P&O)')]; wr = []
    for plant in ('nominal', 'mismatched'):
        for a, b in pairs:
            for col in ('eff_std4', 'eff_real', 'eff_shading'):
                x = gt[(gt.plant == plant) & (gt.controller == a)].sort_values('seed')[col].to_numpy(); y = gt[(gt.plant == plant) & (gt.controller == b)].sort_values('seed')[col].to_numpy(); d = x - y
                try: stat, pv_ = wilcoxon(x, y, zero_method='wilcox', method='auto')
                except ValueError: stat, pv_ = np.nan, np.nan
                wr.append(dict(plant=plant, a=a, b=b, metric=col, n=len(d), median_diff_points=float(np.median(d)), mean_diff_points=float(d.mean()), wilcoxon_W=stat, p_value=pv_, a_better_in=int((d > 0).sum())))
    wr = pd.DataFrame(wr); wr.to_csv(os.path.join(R, 'wilcoxon.csv'), index=False)
    f = lambda m, s, d=2: f'{m:.{d}f} ± {s:.{d}f}'
    tb = pd.DataFrame([{'Plant': x.plant, 'Controller': x.controller, 'Std. scen. eff. (%)': f(x.eff_std4_mean, x.eff_std4_std), 'Real days eff. (%)': f(x.eff_real_mean, x.eff_real_std), 'Shading eff. (%)': f(x.eff_shading_mean, x.eff_shading_std),
                        'GMPP success (%)': f(x.gmpp_success_mean, x.gmpp_success_std, 1), 'Event conv. (cycles)': f(x.conv_cycles_mean, x.conv_cycles_std)} for x in sm.itertuples()])
    save_table('Table_Stats', tb, f'Mean ± std over {len(seeds)} scenario seeds (stage-1 DNN retrained with 5 seeds, paired by seed mod 5).')
    wt = wr.copy(); wt['p_value'] = wt.p_value.map(lambda v: 'n/a' if np.isnan(v) else f'{v:.4f}')
    save_table('Table_Wilcoxon', wt.rename(columns={'median_diff_points': 'median diff (points)', 'mean_diff_points': 'mean diff (points)', 'a_better_in': 'a better in n units'}), 'Paired two-sided Wilcoxon signed-rank tests (a versus b), units = scenario seeds.')
    stats_figure(sm)
    print('stats done', round(time.time() - t0), 's'); return sm, wr


def stats_figure(sm):
    import mppt_style as sty, matplotlib.pyplot as plt
    # figure: bars with std error bars (nominal and mismatched), three metric groups
    sty.apply_style(); ctr = [c for c in sty.COLORS if c in set(sm.controller)]; fig, axes = plt.subplots(1, 3, figsize=(sty.W2, 3.1)); fd = []
    for ax, (col, lab, lo) in zip(axes, (('eff_std4', 'Standard scenarios', 96), ('eff_real', 'Real days', 96), ('eff_shading', 'Partial shading', 70))):
        series = {sty.SHORT[c]: ([sm[(sm.plant == p) & (sm.controller == c)][f'{col}_mean'].iloc[0] for p in ('nominal', 'mismatched')], [sm[(sm.plant == p) & (sm.controller == c)][f'{col}_std'].iloc[0] for p in ('nominal', 'mismatched')]) for c in ctr}
        colors = {sty.SHORT[c]: sty.COLORS[c] for c in ctr}
        fd.append(sty.grouped_bars(ax, ['Nominal', 'Mismatched'], series, colors, ylim=(lo, 100.5), label_values=False, break_axis=True).assign(panel=lab)); ax.set_ylabel('Tracking efficiency (%)' if ax is axes[0] else ''); sty.panel(ax, 'abc'[list(axes).index(ax)], loc='bl', dy=0.16)
    h, l = axes[0].get_legend_handles_labels(); fig.tight_layout(rect=(0, 0.1, 1, 1)); sty.legend_fig(fig, h, l, ncol=5, y=-0.01); sty.save_figure(fig, 'Fig32_seed_statistics', pd.concat(fd, ignore_index=True))


def noise(models, data, pv, pvm, R, ann_pred, poly, tree_est):
    from mppt.config import CFG
    from mppt.evaluate import build_controllers, run_suite, metrics_table, CORE, PS, TR
    from mppt.scenarios import create_scenarios
    from mppt.nn import ScanEstimator
    from mppt.tables import save_table
    import mppt_style as sty, matplotlib.pyplot as plt
    sc = create_scenarios(pv); sub = {k: sc[k] for k in ('partial_shading', 'comprehensive')}; mlp = models[sorted(models)[0]].mlp if 42 not in models else models[42].mlp
    rows = []; ns = [None, 0, 1, 2, 3, 4] if not CFG['quick'] else [None, 0]
    for nseed in ns:
        ctr = build_controllers(ann=ann_pred, poly=poly, dnn_est=ScanEstimator('dnn', mlp), tree_est=tree_est, pso_seed=0, names=CORE + [PS, TR])
        m = metrics_table(run_suite(sub, ctr, pvm, plants=(('nominal', 0.0), ('mismatched', None)), noise_seed=nseed), sc); m['noise_seed'] = -1 if nseed is None else nseed; rows.append(m)
    r = pd.concat(rows); r.to_csv(os.path.join(R, 'noise_runs.csv'), index=False)
    g = r.assign(noisy=r.noise_seed >= 0).groupby(['plant', 'scenario', 'controller', 'noisy'])[['tracking_efficiency_%', 'gmpp_success_%', 'wrong_peak_%']].agg(['mean', 'std']); g.columns = [f'{a}_{b}' for a, b in g.columns]; g = g.reset_index(); g.to_csv(os.path.join(R, 'noise_summary.csv'), index=False)
    sh = g[(g.scenario == 'partial_shading') & (g.plant == 'nominal')]
    tb = pd.DataFrame([{'Controller': c, 'Eff. clean (%)': float(sh[(sh.controller == c) & ~sh.noisy]['tracking_efficiency_%_mean'].iloc[0]), 'Eff. noisy (%)': f"{sh[(sh.controller == c) & sh.noisy]['tracking_efficiency_%_mean'].iloc[0]:.2f} ± {sh[(sh.controller == c) & sh.noisy]['tracking_efficiency_%_std'].iloc[0]:.2f}",
                        'GMPP success clean (%)': float(sh[(sh.controller == c) & ~sh.noisy]['gmpp_success_%_mean'].iloc[0]), 'GMPP success noisy (%)': float(sh[(sh.controller == c) & sh.noisy]['gmpp_success_%_mean'].iloc[0])} for c in sh.controller.unique()])
    save_table('Table_Noise', tb, f"Nine shading patterns, nominal plant, sensor noise sigma_I = {CFG['noise']['sigma_i']:.1%}, sigma_V = {CFG['noise']['sigma_v']:.1%}, sigma_G = {CFG['noise']['sigma_g']:.0%} (noisy values are mean ± std over noise seeds; scoring uses true values).")
    sty.apply_style(); fig, ax = plt.subplots(figsize=(sty.W1 + 1.2, 2.8)); ctr = [c for c in sty.COLORS if c in set(sh.controller)]
    series = {sty.SHORT[c]: ([float(sh[(sh.controller == c) & ~sh.noisy]['tracking_efficiency_%_mean'].iloc[0]), float(sh[(sh.controller == c) & sh.noisy]['tracking_efficiency_%_mean'].iloc[0])], [0.0, float(sh[(sh.controller == c) & sh.noisy]['tracking_efficiency_%_std'].iloc[0])]) for c in ctr}
    fd = sty.grouped_bars(ax, ['Clean', 'Noisy'], series, {sty.SHORT[c]: sty.COLORS[c] for c in ctr}, ylim=(70, 100.5), label_values=False, break_axis=True); ax.set_ylabel('Shading efficiency (%)'); h, l = ax.get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.2, 1, 1)); sty.legend_fig(fig, h, l, ncol=3, y=-0.01); sty.save_figure(fig, 'Fig33_noise', fd); return g


def quantisation(models, data, pv, pvm, R):
    from mppt.config import CFG, RESULTS
    from mppt import models as MD
    from mppt.nn import QuantMLP, ScanEstimator
    from mppt.evaluate import build_controllers, run_suite, metrics_table, HY
    from mppt.scenarios import create_scenarios
    from mppt.tables import save_table
    import mppt_style as sty, matplotlib.pyplot as plt
    mlp = models[sorted(models)[0]].mlp if 42 not in models else models[42].mlp; Xc = MD.hybrid_features(data['val'])[:5000]
    var = {'float32': mlp, 'float16': QuantMLP(mlp, 'fp16'), 'int8': QuantMLP(mlp, 'int8', Xc)}; sc = {'partial_shading': create_scenarios(pv)['partial_shading']}; rows = []
    lat_p = os.path.join(RESULTS, 'hpc', 'latency.csv'); lat = pd.read_csv(lat_p) if os.path.exists(lat_p) else None
    for name, m in var.items():
        pr = lambda X, m=m: m.predict(X); hm = MD.HybridModel(name, 'q', pr, None, mlp.n_params(), mlp.macs(), 0.0); off = MD.eval_hybrid(hm, data).set_index('test_set')
        mt = metrics_table(run_suite(sc, build_controllers(dnn_est=ScanEstimator(name, m), names=[HY]), pvm, only=['partial_shading']), sc).set_index('plant')
        nbytes = mlp.n_params() * 4 if name == 'float32' else m.n_bytes(); ref = mlp.predict(Xc); dv = np.abs(m.predict(Xc) - ref)
        cyc = mlp.macs() * (3.0 if name != 'int8' else 1.0)
        rows.append(dict(precision=name, model_bytes=nbytes, rmse_V_indist=off.loc[MD.TEST_NAMES['T1'], 'RMSE_V'], rmse_V_shaded=off.loc[MD.TEST_NAMES['T3'], 'RMSE_V'], wrong_peak_pct_indist=off.loc[MD.TEST_NAMES['T1'], 'wrong_peak_%'],
                         max_abs_dev_vs_float32_V=float(dv.max()), shading_eff_nominal=mt.loc['nominal', 'tracking_efficiency_%'], shading_eff_mismatched=mt.loc['mismatched', 'tracking_efficiency_%'], gmpp_success_nominal=mt.loc['nominal', 'gmpp_success_%'],
                         measured_latency_us_numpy=(float(lat[lat.variant.str.startswith(name)].median_us.iloc[0]) if lat is not None and (lat.variant.str.startswith(name)).any() else np.nan),
                         latency_is='numpy emulation on the benchmark machine (not an MCU measurement)', ESTIMATE_cycles_assumed=cyc, ESTIMATE_assumption='3 cycles/MAC (float, Cortex-M4F-class) or 1 cycle/MAC (int8 SIMD); a rough analytic estimate, not a measurement'))
    q = pd.DataFrame(rows); q.to_csv(os.path.join(R, 'quantisation.csv'), index=False)
    tb = q[['precision', 'model_bytes', 'rmse_V_indist', 'wrong_peak_pct_indist', 'max_abs_dev_vs_float32_V', 'shading_eff_nominal', 'shading_eff_mismatched', 'gmpp_success_nominal', 'measured_latency_us_numpy']].rename(columns={
        'model_bytes': 'Size (bytes)', 'rmse_V_indist': 'RMSE V_gmpp (V)', 'wrong_peak_pct_indist': 'Wrong-peak (%)', 'max_abs_dev_vs_float32_V': 'Max dev. vs float32 (V)', 'shading_eff_nominal': 'Shading eff. nominal (%)', 'shading_eff_mismatched': 'Shading eff. mismatched (%)',
        'gmpp_success_nominal': 'GMPP success (%)', 'measured_latency_us_numpy': 'Latency, numpy emulation (us)'})
    save_table('Table_Quant', tb, 'Post-training quantisation of the stage-1 DNN (emulated in numpy). Latencies are those of the emulation on the benchmark machine, not MCU measurements.', digits=3)
    sty.apply_style(); fig, (a, b) = plt.subplots(1, 2, figsize=(sty.W2, 2.8)); x = np.arange(3); lab = list(q.precision)
    a.bar(x, q.model_bytes / 1024, 0.6, color=[sty.MATLAB[0], sty.MATLAB[4], sty.MATLAB[3]], edgecolor='k', lw=0.6); a.set_xticks(x); a.set_xticklabels(lab); a.set_ylabel('Model size (KiB)'); sty.panel(a, 'a')
    b.bar(x, q.max_abs_dev_vs_float32_V, 0.6, color=[sty.MATLAB[0], sty.MATLAB[4], sty.MATLAB[3]], edgecolor='k', lw=0.6); b.set_xticks(x); b.set_xticklabels(lab); b.set_ylabel('Max |V_est - V_est,float32| (V)'); sty.panel(b, 'b', loc='tl')
    fig.tight_layout(); sty.save_figure(fig, 'Fig34_quantisation', q[['precision', 'model_bytes', 'max_abs_dev_vs_float32_V', 'rmse_V_indist', 'shading_eff_nominal']]); return q


def run(parts=('stats', 'noise', 'quant')):
    from mppt.config import CFG, RESULTS
    from mppt.data import get_pv_series
    from mppt import models as MD
    from experiments.exp_main import get_models
    pv, pvm, _ = get_pv_series(); data, _c = MD.hybrid_data(pv); ann, poly, hyb = get_models(pv, pvm, data); models = get_seed_models(data, pv, pvm); models[42] = hyb
    trees = MD.train_trees(data, seed=0); ann_pred = MD.v2_predictor(ann, 'ANN-MPPT')
    for d in ('stats', 'noise', 'quantisation'): os.makedirs(os.path.join(RESULTS, d), exist_ok=True)
    seed_models = {k: v for k, v in models.items() if k != 42}
    if 'stats' in parts: stats(seed_models, data, pv, pvm, os.path.join(RESULTS, 'stats'), ann_pred, poly, trees.estimator)
    if 'noise' in parts: noise(models, data, pv, pvm, os.path.join(RESULTS, 'noise'), ann_pred, poly, trees.estimator)
    if 'quant' in parts: quantisation(models, data, pv, pvm, os.path.join(RESULTS, 'quantisation'))


if __name__ == '__main__' and '--figure-only' in sys.argv:
    from mppt.config import RESULTS
    stats_figure(pd.read_csv(os.path.join(RESULTS, 'stats', 'stats_summary.csv')))
elif __name__ == '__main__':
    run(tuple(a for a in sys.argv[1:] if not a.startswith('-')) or ('stats', 'noise', 'quant'))
