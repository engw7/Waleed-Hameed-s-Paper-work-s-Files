"""Experiment 2.3 - boost-converter plant. Averaged CCM model (duty as control variable, inner PI voltage loop) driven by the same MPPT controllers, verified against a switching-level
(PWM) simulation by replaying the recorded command sequences. The quasi-static plant remains the main result; every table says which plant it used.
Outputs results/converter/: converter_parameters.csv, step_metrics.csv, step_logs.csv, converter_vs_static.csv, verification.csv, verification_trace.csv; Table_Converter_Parameters,
Table_Converter_vs_Static, Table_Converter_Verification; Fig30_converter_step, Fig31_converter_verification."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, pandas as pd


def factory(model_scn_shaded):
    from mppt.converter import AveragedBoostPlant
    from pv_single_diode import ShadedString
    from mppt.config import CFG
    def make(scn, model):
        if 'G_sub' in scn: return AveragedBoostPlant(ShadedString(model, (CFG['v_min_op'], CFG['v_max_op'])), shaded=True)
        return AveragedBoostPlant(model)
    return make


def verify(scn, ctrl, i0, i1, label, pvm, R):
    """Run `ctrl` on the averaged converter for cycles [0, i1), then replay the calls i0.. of that run on the switching-level model starting from the recorded state."""
    from mppt.converter import AveragedBoostPlant, SwitchingBoostPlant
    from mppt.plants import truth_module
    from pv_single_diode import ShadedString
    from mppt.config import CFG
    from mppt.sim import run_algorithm
    sub = {k: (v[:i1] if hasattr(v, '__len__') and not isinstance(v, (str, list, dict)) and len(v) >= i1 else v) for k, v in scn.items()}
    sub['events'] = [e for e in scn['events'] if e < i1]
    shaded = 'G_sub' in scn
    mk = factory(None); avg = mk(sub, pvm); truth = np.column_stack([np.zeros(i1), np.ones(i1)])           # truth not needed here
    run_algorithm(ctrl, avg, sub, truth)
    # index of the first call of cycle i0
    calls_per_cycle = len(avg.calls) / i1; k0 = int(round(i0 * calls_per_cycle)) if abs(calls_per_cycle - round(calls_per_cycle)) < 1e-9 else None
    if k0 is None:                                                      # scan cycles make the number of calls per cycle non-constant: take the first call recorded at/after cycle i0 via the G record
        k0 = 0
    model = avg.model; sw = SwitchingBoostPlant(model, shaded=shaded); sw.set_state(avg.calls[k0][4])
    rows = []
    for k in range(k0, len(avg.calls)):
        G, T, Gs, Vc, _ = avg.calls[k]
        if shaded: sw.set_shading(Gs)
        o = sw.calculate_pv_characteristics(G, T, Vc); a = avg.outs[k]
        rows.append(dict(case=label, call=k, V_cmd=Vc, V_avg=a[0], V_sw=o[0], I_avg=a[1], I_sw=o[1], P_avg=a[2], P_sw=o[2], ripple_i_sw=sw.outs[-1][3], ripple_v_sw=sw.outs[-1][4]))
    df = pd.DataFrame(rows); df['jump_V'] = [abs(avg.calls[k][3] - avg.calls[k][4][0]) for k in range(k0, len(avg.calls))]; df['large_jump'] = df.jump_V > 2.0
    df['dV'] = (df.V_avg - df.V_sw).abs(); df['dP'] = (df.P_avg - df.P_sw).abs(); df['dP_rel_%'] = 100 * df.dP / 244.6        # percent of the rated module power (P_max,ref)
    return df, avg, k0


def trace_figure(tdf):
    import mppt_style as sty
    import matplotlib.pyplot as plt
    ta = tdf[tdf.model == 'averaged'][['t_s', 'v_pv', 'i_L', 'i_pv']].to_numpy(); ts = tdf[tdf.model == 'switching'][['t_s', 'v_pv', 'i_L', 'i_pv']].to_numpy()
    sty.apply_style(); fig, axs = plt.subplots(2, 2, figsize=(sty.W2, 4.2)); zl = (ts[-1, 0] - 1.0e-3, ts[-1, 0])
    for r, (col, yl) in enumerate((('v_pv', 'PV voltage (V)'), ('i_L', 'Inductor current (A)'))):
        for k, (ax, lim) in enumerate(zip(axs[r], ((0, ts[-1, 0]), zl))):
            ax.plot(ts[:, 0] * 1e3, ts[:, 1 + r], color=sty.MATLAB[0], lw=0.9, label='Switching-level'); ax.plot(ta[:, 0] * 1e3, ta[:, 1 + r], color=sty.MATLAB[1], lw=1.4, label='Averaged')
            ax.set_xlim(lim[0] * 1e3, lim[1] * 1e3); ax.set_xlabel('Time (ms)')
            if k == 0: ax.set_ylabel(yl)
            m = (ts[:, 0] >= lim[0]) & (ts[:, 0] <= lim[1]); lo, hi = ts[m, 1 + r].min(), ts[m, 1 + r].max(); pad = 0.1 * (hi - lo + 1e-9); ax.set_ylim(lo - pad, hi + pad)
            sty.panel(ax, 'abcd'[2 * r + k], loc='br' if r == 1 else 'tr')
    axs[0, 0].legend(loc='lower right'); fig.tight_layout(); sty.save_figure(fig, 'Fig31_converter_verification', tdf)


def run():
    from mppt.config import CFG, RESULTS, CACHE
    from mppt.data import get_pv_series
    from mppt.scenarios import create_scenarios
    from mppt import models as MD
    from mppt.nn import ScanEstimator
    from mppt.evaluate import build_controllers, run_suite, metrics_table, PO, IC, AN, SC, HY
    from mppt.converter import P as CP, PARAM_NOTES, AveragedBoostPlant, SwitchingBoostPlant
    from mppt.tables import save_table
    from mppt.figures import time_domain
    from experiments.exp_main import get_models
    import mppt_style as sty
    import matplotlib.pyplot as plt
    out = os.path.join(RESULTS, 'converter'); os.makedirs(out, exist_ok=True)
    pv, pvm, _ = get_pv_series(); sc = create_scenarios(pv); data, _c = MD.hybrid_data(pv); ann, poly, hyb = get_models(pv, pvm, data); trees = MD.train_trees(data, seed=0)
    kw = dict(ann=MD.v2_predictor(ann, 'ANN-MPPT'), poly=poly, dnn_est=ScanEstimator('dnn', hyb.mlp), tree_est=trees.estimator, pso_seed=0)
    prm = pd.DataFrame([dict(parameter=k, value=v, note=PARAM_NOTES.get(k, '')) for k, v in CP.items()]); prm.to_csv(os.path.join(out, 'converter_parameters.csv'), index=False)
    save_table('Table_Converter_Parameters', prm, 'Boost-converter parameters (averaged and switching-level models).', digits=4)
    t0 = time.time(); mk = factory(None)
    # (1) all controllers on the step scenario, nominal plant, averaged converter
    ctr = build_controllers(**kw); st = {'step_change': sc['step_change']}
    lg = run_suite(st, ctr, pvm, plants=(('nominal', 0.0),), plant_factory=mk); step_logs = lg['nominal']['step_change']
    mt = metrics_table(lg, sc); mt.to_csv(os.path.join(out, 'step_metrics.csv'), index=False)
    pd.concat([df[['time', 'P', 'V', 'D_conv', 'I_L', 'mpp_power', 'mpp_voltage', 'scan']].assign(controller=n) for n, df in step_logs.items()]).to_csv(os.path.join(out, 'step_logs.csv'), index=False)
    t_step = sc['step_change']['time'][sc['step_change']['events'][1]]
    show = [n for n in (PO, IC, SC, HY, AN) if n in step_logs]
    time_domain({n: step_logs[n] for n in show}, 'Fig30_converter_step', rows=('P', 'V', 'D', 'IL'), zoom=(t_step - 10, t_step + 40), controllers=show)
    # (2) converter plant vs quasi-static plant on the headline controllers
    names = [PO, IC, SC, AN, HY]; sub = {k: sc[k] for k in (('step_change', 'partial_shading') if CFG['quick'] else ('comprehensive', 'step_change', 'cloud_passing', 'ramp_change', 'partial_shading'))}
    lc = run_suite(sub, build_controllers(**kw, names=names), pvm, plants=(('nominal', 0.0),), plant_factory=mk); mc = metrics_table(lc, sc)
    ms = pd.read_csv(os.path.join(RESULTS, 'metrics_by_scenario.csv')); ms = ms[(ms.plant == 'nominal') & ms.scenario.isin(sub) & ms.controller.isin(names)]
    j = mc.merge(ms, on=['plant', 'scenario', 'controller'], suffixes=('_converter', '_static'))
    cmp = j[['scenario', 'controller', 'tracking_efficiency_%_static', 'tracking_efficiency_%_converter', 'gmpp_success_%_static', 'gmpp_success_%_converter', 'osc_amp_V_static', 'osc_amp_V_converter']].copy()
    cmp['delta_eff_points'] = cmp['tracking_efficiency_%_converter'] - cmp['tracking_efficiency_%_static']; cmp.to_csv(os.path.join(out, 'converter_vs_static.csv'), index=False)
    save_table('Table_Converter_vs_Static', cmp.rename(columns={'tracking_efficiency_%_static': 'Eff. static (%)', 'tracking_efficiency_%_converter': 'Eff. converter (%)', 'delta_eff_points': 'Delta (points)'})[['scenario', 'controller', 'Eff. static (%)', 'Eff. converter (%)', 'Delta (points)']],
               'Tracking efficiency on the quasi-static plant versus the averaged boost-converter plant (nominal PV model).', digits=2)
    # (3) verification against the switching-level model
    from mppt.evaluate import build_controllers as bc
    istep = sc['step_change']['events'][1]
    v1, avg1, k1 = verify(sc['step_change'], bc(**kw, names=[PO])[PO](), istep - 5, istep + 25, 'irradiance step (P&O)', pvm, out)
    v2, avg2, k2 = verify(sc['partial_shading'], bc(**kw, names=[HY])[HY](), 0, 30, 'shading pattern 1 incl. scan (Hybrid)', pvm, out)
    ver = pd.concat([v1, v2]); ver.to_csv(os.path.join(out, 'verification.csv'), index=False)
    summ = ver.groupby(['case', 'large_jump']).agg(calls=('call', 'size'), max_dV_V=('dV', 'max'), max_dP_W=('dP', 'max'), max_dP_rel_pct=('dP_rel_%', 'max'), mean_dP_rel_pct=('dP_rel_%', 'mean'), max_jump_V=('jump_V', 'max'), max_ripple_i_A=('ripple_i_sw', 'max'), max_ripple_v_V=('ripple_v_sw', 'max')).reset_index()
    summ.to_csv(os.path.join(out, 'verification_summary.csv'), index=False)
    save_table('Table_Converter_Verification', summ, 'Averaged versus switching-level model driven by the same command sequence: largest deviation of the window-averaged PV voltage and power (percent of the rated power). large_jump = command differs by more than 2 V from the PV voltage at the start of the window (scan probes, start-up): the averaged model and the switching model are both still settling there.', digits=3)
    # trace figure: window right after the irradiance step, both models from the same state
    G, T, Gs, Vc, s0 = avg1.calls[k1 + 5]
    ta, ts = [], []
    a = AveragedBoostPlant(avg1.model); a.set_state(s0); a.calculate_pv_characteristics(G, T, Vc, trace=ta)
    w = SwitchingBoostPlant(avg1.model); w.set_state(s0); w.calculate_pv_characteristics(G, T, Vc, trace=ts)
    ta, ts = np.array(ta), np.array(ts); tdf = pd.concat([pd.DataFrame(ta, columns=['t_s', 'v_pv', 'i_L', 'i_pv']).assign(model='averaged'), pd.DataFrame(ts, columns=['t_s', 'v_pv', 'i_L', 'i_pv']).assign(model='switching')]); tdf.to_csv(os.path.join(out, 'verification_trace.csv'), index=False)
    trace_figure(tdf)
    print('converter experiment done in', round(time.time() - t0), 's'); print(summ.round(4).to_string()); print(cmp.round(3).to_string()); return summ, cmp


if __name__ == '__main__' and '--figure-only' in sys.argv:
    from mppt.config import RESULTS
    trace_figure(pd.read_csv(os.path.join(RESULTS, 'converter', 'verification_trace.csv')))
elif __name__ == '__main__':
    run()
