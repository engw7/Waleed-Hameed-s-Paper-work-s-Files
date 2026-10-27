"""Experiment 2.1 - design ablation of the stage-1 estimator (V_gmpp from 16 inputs). Design decisions use 2005-2014 (train) and 2015-2016 (early stopping) only;
2019-2020 are used for the reported numbers. One CSV per run table + summary, one table, one figure (Fig27_ablation)."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, pandas as pd

# (name, kwargs for train_hybrid_dnn / kind, seeds)   -- every variant is listed a priori; the current design is 'H(128,128,64)'
def variants(quick):
    s3 = [0, 1] if quick else [0, 1, 2]; s5 = [0, 1] if quick else [0, 1, 2, 3, 4]
    return [('DNN (32,32)', dict(hidden=(32, 32)), s3), ('DNN (64,64)', dict(hidden=(64, 64)), s3), ('DNN (128,128,64) [current]', dict(hidden=(128, 128, 64)), s5),
            ('DNN (256,256,128)', dict(hidden=(256, 256, 128)), s3), ('DNN (128,128,128,64)', dict(hidden=(128, 128, 128, 64)), s3),
            ('Current + dropout 0.1', dict(hidden=(128, 128, 64), dropout=0.1), s3), ('Current + L2 1e-4', dict(hidden=(128, 128, 64), l2=1e-4), s3),
            ('Ridge (linear, 16 inputs)', dict(kind='ridge'), [0]), ('Current, G and T only (no scan currents)', dict(hidden=(128, 128, 64), scan=False), s3)]


def run():
    from mppt.config import CFG, RESULTS
    from mppt.data import get_pv_series
    from mppt import models as MD
    from mppt.evaluate import build_controllers, run_suite, metrics_table, HY
    from mppt.scenarios import create_scenarios
    from mppt.tables import save_table
    pv, pvm, _ = get_pv_series(); data, _c = MD.hybrid_data(pv); sc = {'partial_shading': create_scenarios(pv)['partial_shading']}
    out_dir = os.path.join(RESULTS, 'ablation'); rows = []
    for name, kw, seeds in variants(CFG['quick']):
        for seed in seeds:
            kw2 = dict(kw); kind = kw2.pop('kind', 'dnn')
            m = MD.train_ridge(data, name=name) if kind == 'ridge' else MD.train_hybrid_dnn(data, seed=seed, name=name, **kw2)
            off = MD.eval_hybrid(m, data).set_index('test_set'); k1, k3 = MD.TEST_NAMES['T1'], MD.TEST_NAMES['T3']
            logs = run_suite(sc, build_controllers(dnn_est=m.estimator, names=[HY]), pvm, only=['partial_shading']); mt = metrics_table(logs, sc).set_index('plant')
            rows.append(dict(variant=name, seed=seed, params=m.n_params, macs=m.macs, fit_seconds=m.seconds, epochs=m.epochs,
                             test_rmse_V_indist=off.loc[k1, 'RMSE_V'], test_mae_V_indist=off.loc[k1, 'MAE_V'], wrong_peak_pct_indist=off.loc[k1, 'wrong_peak_%'],
                             test_rmse_V_shaded_nominal=off.loc[k3, 'RMSE_V'], wrong_peak_pct_shaded_nominal=off.loc[k3, 'wrong_peak_%'],
                             shading_eff_nominal=mt.loc['nominal', 'tracking_efficiency_%'], shading_eff_mismatched=mt.loc['mismatched', 'tracking_efficiency_%'],
                             gmpp_success_nominal=mt.loc['nominal', 'gmpp_success_%'], gmpp_success_mismatched=mt.loc['mismatched', 'gmpp_success_%'],
                             closed_loop_wrong_peak_pct_nominal=mt.loc['nominal', 'wrong_peak_%']))
            print(name, seed, {k: round(v, 4) for k, v in rows[-1].items() if isinstance(v, float)}, flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'ablation_runs.csv'), index=False)
    r = pd.DataFrame(rows); order = list(dict.fromkeys(r.variant))
    cols = [c for c in r.columns if c not in ('variant', 'seed')]
    g = r.groupby('variant', sort=False)[cols]; mean, std = g.mean(), g.std(ddof=1).fillna(0.0)
    summ = mean.copy(); summ.columns = [c + '_mean' for c in cols]
    for c in cols: summ[c + '_std'] = std[c]
    summ.insert(0, 'n_seeds', g.size()); summ = summ.reset_index(); summ.to_csv(os.path.join(out_dir, 'ablation.csv'), index=False)
    f = lambda m, s, d=2: f'{m:.{d}f} ± {s:.{d}f}'
    tb = pd.DataFrame([{'Variant': x.variant, 'Seeds': int(x.n_seeds), 'Parameters': int(round(x.params_mean)), 'MACs': int(round(x.macs_mean)),
                        'RMSE V_gmpp, in-dist. (V)': f(x.test_rmse_V_indist_mean, x.test_rmse_V_indist_std, 3), 'Wrong-peak, in-dist. (%)': f(x.wrong_peak_pct_indist_mean, x.wrong_peak_pct_indist_std),
                        'Shading eff., nominal (%)': f(x.shading_eff_nominal_mean, x.shading_eff_nominal_std), 'Shading eff., mismatched (%)': f(x.shading_eff_mismatched_mean, x.shading_eff_mismatched_std),
                        'GMPP success, nominal (%)': f(x.gmpp_success_nominal_mean, x.gmpp_success_nominal_std, 1)} for x in summ.itertuples()])
    save_table('Table_Ablation', tb, 'Design ablation of the stage-1 estimator: offline error on the 2019-2020 test set and closed-loop results on the nine shading patterns (mean ± std over seeds).')
    figure(summ, r); return summ


def figure(summ, runs):
    import matplotlib.pyplot as plt
    import mppt_style as sty
    sty.apply_style(); fig, (a, b) = plt.subplots(1, 2, figsize=(sty.W2, 3.3)); y = np.arange(len(summ))[::-1]
    lab = [v.replace(' [current]', '*').replace('Current, G and T only (no scan currents)', 'G, T only') .replace('Ridge (linear, 16 inputs)', 'Ridge') for v in summ.variant]
    a.barh(y, summ.test_rmse_V_indist_mean, xerr=summ.test_rmse_V_indist_std, color=sty.MATLAB[0], edgecolor='k', lw=0.6, capsize=2, error_kw=dict(lw=0.8))
    a.set_yticks(y); a.set_yticklabels(lab, fontsize=7.5); a.set_xlabel('RMSE of $V_{gmpp}$, test 2019–2020 (V)'); a.set_xlim(left=0); sty.panel(a, 'a', loc='tr')
    b.barh(y, summ.shading_eff_nominal_mean, xerr=summ.shading_eff_nominal_std, color=sty.MATLAB[1], edgecolor='k', lw=0.6, capsize=2, error_kw=dict(lw=0.8))
    lo = float(max(0, (summ.shading_eff_nominal_mean - summ.shading_eff_nominal_std).min() - 1)); b.set_xlim(lo, 100); b.set_yticks(y); b.set_yticklabels([]); b.set_xlabel(f'Closed-loop shading efficiency (%), axis starts at {lo:.0f}'); sty.panel(b, 'b', loc='tl')
    fig.tight_layout(); fd = summ[['variant', 'n_seeds', 'test_rmse_V_indist_mean', 'test_rmse_V_indist_std', 'shading_eff_nominal_mean', 'shading_eff_nominal_std', 'gmpp_success_nominal_mean', 'params_mean']]
    sty.save_figure(fig, 'Fig27_ablation', fd)


if __name__ == '__main__':
    if '--figure-only' in sys.argv:
        from mppt.config import RESULTS
        figure(pd.read_csv(os.path.join(RESULTS, 'ablation', 'ablation.csv')), None)
    else:
        run()
