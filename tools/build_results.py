"""results/PAPER_NUMBERS.json and results/RESULTS.md, both generated from the CSV files (no typed numbers)."""
import os, sys, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, pandas as pd
from mppt.config import CFG, RESULTS, ROOT
R = lambda *p: os.path.join(RESULTS, *p)
RES = os.path.relpath(RESULTS, ROOT)
V3 = lambda *p: os.path.join(ROOT, 'reference', 'v3', *p)
rd = lambda *p: pd.read_csv(R(*p))
HY, SC, PO, IC, AN, DO, P2, PS, TR = 'Hybrid (DNN+P&O)', 'Scan+P&O (no ML)', 'P&O', 'IncCond', 'ANN-MPPT', 'DNN scan-only (no P&O)', 'Poly2(G,T)-MPPT', 'PSO', 'Tree scan+P&O'
V3MAP = {'event_convergence_cycles': 'event_convergence_steps', 'cold_start_B_conv_cycles': 'cold_start_B_conv_steps', 'gmpp_success_nominal_%': 'shading_success_nominal_%', 'gmpp_success_mismatched_%': 'shading_success_mismatched_%'}


def _get(df, col, flt):
    q = df
    for k, v in flt.items(): q = q[q[k] == v]
    return None if (len(q) == 0 or col not in q.columns) else q[col].iloc[0]


def paper_numbers():
    out = []
    def add(key, csv, flt, col, unit, old=None, note=''):
        try: df = pd.read_csv(R(csv))
        except FileNotFoundError: return
        v = _get(df, col, flt)
        if v is None: return
        v = v.item() if hasattr(v, 'item') else v; e = dict(key=key, value=v, unit=unit, source_csv=f'{RES}/{csv}', column=col, filter=flt)
        if old:
            try:
                od = pd.read_csv(V3(old[0])); ov = _get(od, old[2], old[1]); ov = ov.item() if hasattr(ov, 'item') else ov
                e['old_value'] = ov; e['changed'] = (not np.isclose(float(v), float(ov), rtol=1e-3, atol=5e-3)) if isinstance(v, (int, float)) and ov is not None and not isinstance(ov, str) else (v != ov)
            except Exception as ex: e['old_value'] = None
        if note: e['note'] = note
        out.append(e)
    for c, tag in ((PO, 'PO'), (IC, 'IncCond'), (AN, 'ANN'), (P2, 'Poly2'), (SC, 'ScanPO'), (DO, 'DNNonly'), (HY, 'Hybrid'), (PS, 'PSO'), (TR, 'ScanTrees')):
        for col, unit in (('eff_std4_nominal_%', '%'), ('eff_std4_mismatched_%', '%'), ('eff_real_days_nominal_%', '%'), ('shading_eff_nominal_%', '%'), ('shading_eff_mismatched_%', '%'), ('gmpp_success_nominal_%', '%'), ('gmpp_success_mismatched_%', '%'),
                          ('event_convergence_cycles', 'cycles'), ('osc_amp_V', 'V'), ('compute_us_per_cycle', 'us'), ('cold_start_B_conv_cycles', 'cycles'), ('scan_overhead_std4_%', '%')):
            oc = V3MAP.get(col, col)
            add(f'headline.{tag}.{col}', 'metrics_headline.csv', {'controller': c}, col, unit, old=('metrics_headline.csv', {'algorithm': c}, oc) if c not in (PS, TR) else None,
                note='GMPP-success criterion changed from v3 (>=95 % over the last 25 % of an episode) to >=98 % over the last 10 non-scan cycles' if 'gmpp_success' in col else '')
    for sev_col, key in (('RMSE_V', 'dnn.rmse_V'), ('MAE_V', 'dnn.mae_V'), ('P_ratio_%', 'dnn.static_power_ratio_pct'), ('wrong_peak_%', 'dnn.wrong_peak_pct')):
        for t, tag in (('T1 in-distribution (DR plants, mixed shading)', 'T1'), ('T4 OUT-OF-RANGE plants, mixed shading', 'T4')):
            add(f'{key}.{tag}', 'hybrid_offline_metrics.csv', {'test_set': t}, sev_col, 'V' if 'V' in sev_col else '%', old=('hybrid_offline_metrics.csv', {'test set': t, 'estimator': 'Hybrid DNN estimate'}, sev_col) if sev_col in ('RMSE_V', 'MAE_V', 'P_ratio_%') else None)
    for al, tag in ((0.9, '90'), (0.95, '95')):
        for t, tg in (('T1 in-distribution (DR plants, mixed shading)', 'T1'), ('T4 OUT-OF-RANGE plants, mixed shading', 'T4')):
            add(f'conformal.{tag}.{tg}.coverage', 'hybrid_conformal_coverage.csv', {'test_set': t, 'nominal': al}, 'empirical_coverage', '-', old=('hybrid_conformal_coverage.csv', {'test set': t, 'nominal': al}, 'empirical_coverage'))
            add(f'conformal.{tag}.{tg}.half_width', 'hybrid_conformal_coverage.csv', {'test_set': t, 'nominal': al}, 'interval_half_width_V', 'V', old=('hybrid_conformal_coverage.csv', {'test set': t, 'nominal': al}, 'interval_half_width_V'))
    for fn, col, key, unit in (('hpc/latency.csv', 'params', 'dnn.parameters', '-'), ('hpc/latency.csv', 'macs', 'dnn.macs', '-')): add(key, fn, {}, col, unit)
    lat = R('hpc', 'latency.csv')
    if os.path.exists(lat):
        for i, v in pd.read_csv(lat).iterrows():
            out.append(dict(key=f'hpc.latency_median_us.{v.variant}', value=float(v.median_us), unit='us', source_csv=f'{RES}/hpc/latency.csv', column='median_us', filter={'variant': v.variant}))
            out.append(dict(key=f'hpc.latency_fraction_of_cycle.{v.variant}', value=float(v.fraction_of_cycle_budget_median), unit='-', source_csv=f'{RES}/hpc/latency.csv', column='fraction_of_cycle_budget_median', filter={'variant': v.variant}))
    for w in (2, 4, 8):
        add(f'hpc.generation_speedup.{w}workers', 'hpc/generation_scaling.csv', {'workers': w}, 'speedup', 'x'); add(f'hpc.generation_efficiency.{w}workers', 'hpc/generation_scaling.csv', {'workers': w}, 'efficiency', '-')
        add(f'hpc.sweep_speedup.{w}workers', 'hpc/sweep_scaling.csv', {'workers': w}, 'speedup', 'x')
    add('hpc.generation_serial_seconds', 'hpc/generation_scaling.csv', {'workers': 1}, 'wallclock_s', 's'); add('hpc.scan_cycle_fraction_nominal_pct', 'hpc/scan_overhead.csv', {'plant': 'nominal', 'controller': HY}, 'scan_cycle_fraction_pct', '%')
    add('hpc.scan_energy_overhead_nominal_pct', 'hpc/scan_overhead.csv', {'plant': 'nominal', 'controller': HY}, 'scan_energy_overhead_pct', '%')
    for k, c in (('P&O', PO), ('Hybrid', HY), ('Scan+P&O', SC)):
        for col in ('shading_eff_nominal', 'gmpp_success_nominal'): pass
    ws = R('stats', 'wilcoxon.csv')
    if os.path.exists(ws):
        for i, v in pd.read_csv(ws).iterrows(): out.append(dict(key=f'stats.wilcoxon.{v.plant}.{v.a}.vs.{v.b}.{v.metric}.p', value=None if np.isnan(v.p_value) else float(v.p_value), unit='-', source_csv=f'{RES}/stats/wilcoxon.csv', column='p_value', filter={'plant': v.plant, 'a': v.a, 'b': v.b, 'metric': v.metric}))
    for fn, col, key, flt in (('converter/verification_summary.csv', 'max_dP_rel_pct', 'converter.max_dP_rel_pct', None), ('converter/verification_summary.csv', 'max_dV_V', 'converter.max_dV_V', None)):
        p = R(fn)
        if os.path.exists(p):
            for i, v in pd.read_csv(p).iterrows(): out.append(dict(key=f'{key}.{v["case"]}.large_jump_{bool(v["large_jump"])}', value=float(v[col]), unit='% of rated power' if 'pct' in col else 'V', source_csv=f'{RES}/{fn}', column=col, filter={'case': v['case'], 'large_jump': bool(v['large_jump'])}))
    json.dump(out, open(R('PAPER_NUMBERS.json'), 'w'), indent=1, default=str); return out


def _md(name):
    p = R('tables', name + '.md'); return open(p).read() if os.path.exists(p) else f'(table {name} not available)\n'


def results_md():
    def g(csv, col, **f):
        try: return float(_get(rd(csv), col, f))
        except Exception: return float('nan')
    hl = rd('metrics_headline.csv'); H = lambda c, col: g('metrics_headline.csv', col, controller=c)
    sh = lambda c: H(c, 'shading_eff_nominal_%'); gs = lambda c: H(c, 'gmpp_success_nominal_%')
    wil = rd('stats', 'wilcoxon.csv') if os.path.exists(R('stats', 'wilcoxon.csv')) else None
    def W(a, b, metric, plant='nominal'):
        if wil is None: return None
        q = wil[(wil.a == a) & (wil.b == b) & (wil.metric == metric) & (wil.plant == plant)]; return None if q.empty else q.iloc[0]
    def sig(a, b, metric):
        w = W(a, b, metric)
        if w is None: return 'not tested'
        p = w.p_value; return f"median difference {w.median_diff_points:+.2f} points, Wilcoxon p = {p:.3f} (n = {int(w.n)}) → " + ('significant at 0.05' if p < 0.05 else 'not significant at 0.05')
    ab = rd('ablation', 'ablation.csv') if os.path.exists(R('ablation', 'ablation.csv')) else None
    cv = rd('converter', 'converter_vs_static.csv') if os.path.exists(R('converter', 'converter_vs_static.csv')) else None
    ver = rd('converter', 'verification_summary.csv') if os.path.exists(R('converter', 'verification_summary.csv')) else None
    hpc = rd('hpc', 'hpc_summary.csv') if os.path.exists(R('hpc', 'hpc_summary.csv')) else None
    L = []
    L.append(f"1. **Global scan + local P&O gives the shading robustness.** On the nine shading patterns Scan+P&O (no learning) reaches {sh(SC):.2f} % efficiency and finds the global peak in {gs(SC):.0f} % of the patterns; P&O reaches {sh(PO):.2f} % / {gs(PO):.0f} %, IncCond {sh(IC):.2f} % / {gs(IC):.0f} %.")
    L.append(f"2. **The DNN does not add robustness on top of that.** Hybrid (DNN + P&O) shading efficiency {sh(HY):.2f} % versus Scan+P&O {sh(SC):.2f} %: {sig(HY, SC, 'eff_shading')}.")
    L.append(f"3. **Where the DNN helps:** the cold start / hand-off. Cold-start convergence (valid starts) Hybrid {H(HY, 'cold_start_B_conv_cycles'):.1f} cycles, Scan+P&O {H(SC, 'cold_start_B_conv_cycles'):.1f}, P&O {H(PO, 'cold_start_B_conv_cycles'):.1f}; the scan is assumed to take one control cycle and not to be slew-limited, which favours both scan-based controllers.")
    L.append(f"4. **On uniform irradiance the scan costs a little:** standard scenarios nominal plant, Hybrid {H(HY, 'eff_std4_nominal_%'):.2f} %, P&O {H(PO, 'eff_std4_nominal_%'):.2f} %, IncCond {H(IC, 'eff_std4_nominal_%'):.2f} %, plain ANN-MPPT {H(AN, 'eff_std4_nominal_%'):.2f} % (mismatched plant {H(AN, 'eff_std4_mismatched_%'):.2f} %). The plain ANN is a map the plant is also simulated with; it is an upper bound, not evidence for hardware.")
    if ab is not None:
        cur = ab[ab.variant.str.contains('current', case=False) & ab.variant.str.startswith('DNN')].iloc[0]; small = ab[ab.variant == 'DNN (32,32)']
        L.append(f"5. **Architecture is not the bottleneck.** Current 128-128-64 net: shading efficiency {cur.shading_eff_nominal_mean:.2f} %, test RMSE {cur.test_rmse_V_indist_mean:.2f} V; " + (f"the 32-32 net: {small.shading_eff_nominal_mean.iloc[0]:.2f} %, {small.test_rmse_V_indist_mean.iloc[0]:.2f} V." if len(small) else '') + " Closed-loop results differ far less than the offline error does, because P&O corrects the hand-off error.")
    L.append(f"6. **Other comparators:** PSO {sh(PS):.2f} % shading efficiency ({gs(PS):.0f} % success), Scan+Trees {sh(TR):.2f} % ({gs(TR):.0f} %), so a gradient-boosted-tree estimator does as well as the DNN in the same structure; PSO finds the global peak but converges slowly ({H(PS, 'event_convergence_cycles'):.1f} cycles per event vs {H(HY, 'event_convergence_cycles'):.1f} for Hybrid) and oscillates more ({H(PS, 'osc_amp_V'):.1f} V p-p).")
    if cv is not None and ver is not None:
        sm_, lg_ = ver[~ver.large_jump.astype(bool)], ver[ver.large_jump.astype(bool)]
        L.append(f"7. **Converter plant:** largest change of tracking efficiency versus the quasi-static plant {cv.delta_eff_points.abs().max():.2f} points; averaged-vs-switching deviation for small command steps at most {sm_.max_dP_rel_pct.max():.2f} % of rated power and {sm_.max_dV_V.max():.3f} V" + (f"; during large command jumps (scan probes) up to {lg_.max_dP_rel_pct.max():.1f} % / {lg_.max_dV_V.max():.2f} V because neither model has settled within the window." if len(lg_) else '.'))
    nz = R('noise', 'noise_summary.csv'); q = R('quantisation', 'quantisation.csv')
    if os.path.exists(nz):
        n = pd.read_csv(nz); n = n[(n.scenario == 'partial_shading') & (n.plant == 'nominal')]
        nv = lambda c, noisy: float(n[(n.controller == c) & (n.noisy == noisy)]['tracking_efficiency_%_mean'].iloc[0])
        qq = pd.read_csv(q) if os.path.exists(q) else None
        L.append(f"8. **Noise and precision:** with sensor noise (I 0.5 %, V 0.5 %, G 2 %) shading efficiency drops from {nv(HY, False):.2f} % to {nv(HY, True):.2f} % for Hybrid and from {nv(SC, False):.2f} % to {nv(SC, True):.2f} % for Scan+P&O (so the loss comes from P&O dithering on noisy power readings, not from the DNN), P&O {nv(PO, False):.2f} % to {nv(PO, True):.2f} %." + (f" Float16 leaves the estimate unchanged (max deviation {float(qq[qq.precision == 'float16'].max_abs_dev_vs_float32_V.iloc[0]):.2f} V); post-training int8 (numpy emulation) raises the test RMSE from {float(qq[qq.precision == 'float32'].rmse_V_indist.iloc[0]):.2f} to {float(qq[qq.precision == 'int8'].rmse_V_indist.iloc[0]):.2f} V with single outliers up to {float(qq[qq.precision == 'int8'].max_abs_dev_vs_float32_V.iloc[0]):.0f} V, yet the closed-loop shading efficiency stays at {float(qq[qq.precision == 'int8'].shading_eff_nominal.iloc[0]):.2f} % because P&O corrects the hand-off." if qq is not None else ''))
    if hpc is not None:
        s = hpc.set_index('quantity').value
        L.append(f"9. **HPC versus real time:** data generation speeds up {float(s['data generation speed-up at 4 workers']):.2f}× on four workers (efficiency {float(s['data generation efficiency at 4 workers']):.2f}); the controller decision takes {H(HY, 'compute_us_per_cycle'):.0f} µs per cycle and the DNN {float(s['DNN inference latency float32 numpy (median)']):.0f} µs (numpy) of a 1 s cycle. Parallel compute is needed for training data, ablations and multi-seed studies, not for the online controller.")
    L.append("10. **Limits:** simulation only (no hardware-in-the-loop, no bench test); one PV-module model family; scan assumed to take one control cycle; conformal intervals are valid only in distribution; no claim about hardware cost beyond emulated latency.")
    t = ['# Results (generated by `tools/build_results.py` from the files in `results/`)\n', '## What the data say / do not say\n', '\n'.join(L) + '\n']
    notes = {'Table_Headline': 'Shows how every controller performs on the standard scenarios and the nine shading patterns on two plants. It does not show hardware behaviour; the "mismatched" plant is a perturbed model, not a measured module.',
             'Table_Ablation': 'Shows how much the stage-1 estimator design matters offline and closed-loop. It does not show that a smaller network is safe outside the training distribution.',
             'Table_Stats': 'Shows seed-to-seed variability of the efficiency numbers. It does not include variability of the PV model itself.', 'Table_Wilcoxon': 'Paired tests over scenario seeds; with n = 10 only differences that are consistent in sign across all seeds can reach p < 0.05.',
             'Table_HPC': 'Shows where parallel compute pays off (data generation, sweeps) and that inference is far inside the cycle budget. int8/float16 timings are numpy emulation and not MCU measurements; worker counts above the CPU count are oversubscribed.',
             'Table_Converter_Verification': 'Shows that the averaged model agrees with a switching-level model for the replayed command sequences. It is not a hardware measurement.', 'Table_Converter_vs_Static': 'Shows that adding converter dynamics changes the efficiencies by the listed amounts.',
             'Table_Noise': 'Shows sensitivity to sensor noise of the stated size only.', 'Table_Quant': 'Shows accuracy/size under post-training quantisation; latency is an emulation on the benchmark machine.', 'Table_conceptual_comparison': 'Design columns describe our implementations; "Hybrid P&O (literature)" is not implemented here.',
             'Table_Literature': 'Intentionally unfilled: no listed paper was read for this study.'}
    t.append('\n## Tables\n')
    for n in ['Table_Headline', 'Table_Stats', 'Table_Wilcoxon', 'Table_Ablation', 'Table_HPC', 'Table_Converter_Verification', 'Table_Converter_vs_Static', 'Table_Noise', 'Table_Quant', 'Table_conceptual_comparison', 'Table_Literature']:
        t.append(f'### {n}\n\n{notes.get(n, "")}\n\n{_md(n)}')
    t.append('\n## Figures (thumbnails)\n')
    for p in sorted(glob.glob(R('figures', 'Fig*.png'))): t.append(f'<img src="figures/{os.path.basename(p)}" width="360" alt="{os.path.basename(p)}"> ')
    open(R('RESULTS.md'), 'w').write('\n'.join(t) + '\n')


NONML = ('P&O', 'IncCond', 'Scan+P&O (no ML)')


def changelog(out):
    mark = '<!-- GENERATED: do not edit below this line -->'; p = os.path.join(ROOT, 'CHANGELOG_ROUND4.md'); txt = open(p).read().split(mark)[0].rstrip() + '\n\n' + mark + '\n'
    rows = [e for e in out if 'old_value' in e and e['old_value'] is not None]; ch = [e for e in rows if e.get('changed')]
    def why(e):
        k = e['key']
        if 'gmpp_success' in k: return 'success criterion changed on purpose (>=98 % over the last 10 non-scan cycles, v3: >=95 % over the last 25 % of an episode)'
        if 'compute_us' in k: return ('wall-clock timing, depends on machine and run' if any(f'.{t}.' in k for t in ('PO', 'IncCond', 'ScanPO')) else 'the learned model is now evaluated with a numpy re-implementation of the forward pass (v3 timed the Keras/sklearn call); framework latencies are in Table_HPC')
        if any(f'.{t}.' in k for t in ('PO', 'IncCond', 'ScanPO')) : return 'UNEXPECTED for a non-learned controller (scenario/metric definition changed: convergence and oscillation use metrics.py definitions of Round 4)' if ('convergence' in k or 'osc' in k or 'cold_start' in k or 'scan_overhead' in k) else 'UNEXPECTED for a non-learned controller: investigate'
        if k.startswith('conformal') or k.startswith('dnn.'): return 'stage-1 DNN re-trained in the Round-4 pipeline (same data configuration, seed 42)'
        return 'learned model re-trained in the Round-4 pipeline and/or metric definition of Round 4 (metrics.py)'
    L = [f'\n## Every number that changed against `reference/v3` (old -> new -> reason)\n', f'{len(rows)} numbers have a v3 counterpart; {len(ch)} differ by more than 0.1 % relative / 0.005 absolute; the others are unchanged.\n', '| key | old (v3) | new | unit | reason |\n|---|---|---|---|---|']
    for e in ch:
        fo = lambda v: f'{v:.4g}' if isinstance(v, (int, float)) else str(v)
        L.append(f"| {e['key']} | {fo(e['old_value'])} | {fo(e['value'])} | {e['unit']} | {why(e)} |")
    hl = rd('metrics_headline.csv'); H = lambda c, col: float(hl[hl.controller == c][col].iloc[0]); st = rd('stats', 'stats_summary.csv') if os.path.exists(R('stats', 'stats_summary.csv')) else None
    L += ['\n## Claims of the earlier manuscript that the new evidence does not support\n', '| earlier claim | what the Round-4 data show |', '|---|---|',
          f"| ANN-MPPT tracking efficiency 98.7 %, P&O 94.15 %, IncCond 95.8 % | standard scenarios, nominal plant: ANN-MPPT {H(AN, 'eff_std4_nominal_%'):.2f} %, P&O {H(PO, 'eff_std4_nominal_%'):.2f} %, IncCond {H(IC, 'eff_std4_nominal_%'):.2f} %; the old figures are not reproducible with any metric definition |",
          '| tracking accuracy 99.57 %, R2 = 0.744, MSE table | no definition reproduces these; the quantities are dropped (docs/METRICS.md) |', '| convergence 2.8 steps (ANN) vs 6.2 / 8.5 | event convergence (control cycles): ' + f"ANN-MPPT {H(AN, 'event_convergence_cycles'):.2f}, P&O {H(PO, 'event_convergence_cycles'):.2f}, IncCond {H(IC, 'event_convergence_cycles'):.2f}, Hybrid {H(HY, 'event_convergence_cycles'):.2f}; integer per event, non-converging runs counted separately |",
          f"| ANN-MPPT is robust to partial shading | shading efficiency ANN-MPPT {H(AN, 'shading_eff_nominal_%'):.2f} % (GMPP success {H(AN, 'gmpp_success_nominal_%'):.0f} %), mismatched plant {H(AN, 'shading_eff_mismatched_%'):.2f} %; it is a single-peak map |",
          f"| oscillations are eliminated | steady-state voltage oscillation p-p: P&O {H(PO, 'osc_amp_V'):.2f} V, Scan+P&O {H(SC, 'osc_amp_V'):.2f} V, Hybrid {H(HY, 'osc_amp_V'):.2f} V (the hybrid ends in P&O); only the map-based controllers and the DNN scan-only variant hold a constant voltage |",
          f"| a DNN is needed for shading robustness | Scan+P&O without any learning reaches {H(SC, 'shading_eff_nominal_%'):.2f} % vs Hybrid {H(HY, 'shading_eff_nominal_%'):.2f} % |"]
    if os.path.exists(R('hpc', 'generation_scaling.csv')):
        g = pd.read_csv(R('hpc', 'generation_scaling.csv')).set_index('workers'); L.append(f"| Introduction: 1.3-2.1x speed-up on four workers | measured on this machine: training-data generation {g.speedup[4]:.2f}x on four workers (2 workers: {g.speedup[2]:.2f}x; 8 workers on 4 CPUs: {g.speedup[8]:.2f}x, oversubscribed) |")
    L.append(f"| controller compute 17.6 us per cycle (Hybrid, v3) | {H(HY, 'compute_us_per_cycle'):.1f} us per cycle with the numpy forward pass used in the closed loop (framework call timings are in Table_HPC) |")
    cp = R('converter', 'converter_vs_static.csv')
    if os.path.exists(cp):
        c = pd.read_csv(cp); big = c[c.delta_eff_points.abs() > 0.1]
        L += ['\n## Converter plant versus quasi-static plant: efficiency changes larger than 0.1 points\n', f'{len(big)} of {len(c)} scenario/controller pairs; the quasi-static plant stays the main result, the converter plant is a secondary study.\n', '| scenario | controller | static (%) | converter (%) | change (points) |', '|---|---|---|---|---|']
        L += [f"| {r.scenario} | {r.controller} | {r['tracking_efficiency_%_static']:.3f} | {r['tracking_efficiency_%_converter']:.3f} | {r.delta_eff_points:+.3f} |" for _, r in big.iterrows()]
    open(p, 'w').write(txt + '\n'.join(L) + '\n')


def build_all():
    out = paper_numbers(); results_md(); changelog(out)


if __name__ == '__main__':
    build_all()
