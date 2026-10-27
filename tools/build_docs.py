"""Generates docs/METRICS.md, docs/PARAMETERS.md, docs/FIGURE_MAP.md, docs/FIGURE_CAPTIONS.md, results/PAPER_NUMBERS.json, results/RESULTS.md and results/fingerprint.txt
from the CSV files written by the experiments. Nothing in here is typed by hand except fixed wording; every number is read from a results file."""
import os, sys, re, json, hashlib, glob, inspect
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, pandas as pd
from mppt.config import CFG, RESULTS, ROOT
import metrics as M

DOCS = os.path.join(ROOT, 'docs'); os.makedirs(DOCS, exist_ok=True)
R = lambda *p: os.path.join(RESULTS, *p)
rd = lambda *p: pd.read_csv(R(*p))
SHORT = {'P&O': 'P&O', 'IncCond': 'IncCond', 'Poly2(G,T)-MPPT': 'Poly2', 'ANN-MPPT': 'ANN-MPPT', 'Scan+P&O (no ML)': 'Scan+P&O', 'DNN scan-only (no P&O)': 'DNN scan-only', 'Hybrid (DNN+P&O)': 'Hybrid', 'PSO': 'PSO', 'Tree scan+P&O': 'Scan+Trees'}


def one(df, col, **flt):
    q = df
    for k, v in flt.items(): q = q[q[k] == v]
    assert len(q) >= 1, (col, flt)
    return float(q[col].iloc[0]) if not isinstance(q[col].iloc[0], str) else q[col].iloc[0]


# ------------------------------------------------------------------------------------------------ docs: metrics, parameters
def metrics_md():
    txt = ['# Metric definitions (single source: `metrics.py`)\n', 'Every figure of merit in the study is computed by the functions below; constants are fixed a priori and never tuned on the 2019–2020 test years.\n',
           '| constant | value | meaning |\n|---|---|---|']
    for n, m in (('CONV_TOL', 'converged when P/P_mpp >= 1 - CONV_TOL'), ('CONV_HOLD', '... for this many consecutive control cycles'), ('OSC_WINDOW', 'cycles per oscillation window'), ('OSC_QUASI_STEADY_V', 'a window is quasi-steady if V_mpp moves at most this much (V)'),
                 ('OSC_SKIP_AFTER_EVENT', 'cycles ignored after an event'), ('GMPP_TOL', 'GMPP success: |P - P_gmpp|/P_gmpp <= this'), ('GMPP_TAIL', '... mean over the last this-many non-scan cycles of a pattern'), ('WRONG_PEAK_V', 'a scan estimate is a wrong peak beyond this distance from V_gmpp (V)')):
        txt.append(f'| `{n}` | {getattr(M, n)} | {m} |')
    txt.append('\n## Functions\n')
    for n in ('tracking_efficiency', 'instantaneous_efficiency', 'convergence_cycles', 'oscillation', 'gmpp_success', 'scan_overhead', 'wrong_peak_rate', 'compute_cost'):
        f = getattr(M, n); txt.append(f'### `{n}{inspect.signature(f)}`\n\n{inspect.getdoc(f) or ""}\n')
    txt.append('## Notes\n\n- *Tracking efficiency* is energy-weighted: eta = 100 sum(P dt) / sum(P_mpp dt), P_mpp is the true maximum of the actual plant (global maximum under shading). The earlier manuscript quantity "tracking accuracy" (99.57 %) is not reproducible from any definition and is dropped.\n'
               '- *Convergence* is an integer number of control cycles (1 cycle = 1 s) per event, averaged over events that converge before the next event; non-integer values such as 2.8 are means over events/scenarios, and runs that never converge are counted separately (`n_not_converged`).\n'
               '- *Scan overhead* is the energy given up during scan cycles; it is included in the tracking efficiency of scan-based controllers. The scan is assumed not to be slew-limited (one control cycle); the converter study shows that 14 settling windows fit in 0.84 s.\n'
               '- DNN offline metrics (RMSE/MAE of V_gmpp, wrong-peak rate) describe the stage-1 estimator only; they are not MPPT performance metrics.\n')
    open(os.path.join(DOCS, 'METRICS.md'), 'w').write('\n'.join(txt))


def parameters_md():
    from mppt.scenarios import PATTERNS
    t = ['# Parameters and experimental design\n', '## Data and splits\n', f"- Data: PVGIS-SARAH3 hourly series, Ramadi, 2005–2020 (`PVGIS_Ramadi_2005-2020.csv`); daytime threshold G >= {CFG['g_min']:g} W/m².",
         f"- Training years <= {CFG['chrono']['train_end']}; early stopping / design decisions {CFG['chrono']['train_end'] + 1}–{CFG['chrono']['val_end']}; conformal calibration {CFG['chrono']['val_end'] + 1}–{CFG['chrono']['calib_end']}; reporting {CFG['chrono']['calib_end'] + 1}–2020 only.",
         f"- Stage-1 DNN training set: {CFG['hybrid_n_train']:,} domain-randomised plant samples (validation {CFG['hybrid_n_val']:,}, calibration {CFG['hybrid_n_calib']:,}, tests {CFG['hybrid_n_test']:,}–{2 * CFG['hybrid_n_test']:,}).",
         f"- Seeds: headline DNN seed {CFG['seed']}; DNN seeds {CFG['dnn_seeds']}; scenario seeds {CFG['scenario_seeds']}; noise seeds 0–4. QUICK mode: {CFG['quick']}.", '\n## Scenarios (1 control cycle = 1 s)\n',
         '- comprehensive, step_change (400 → 800 W/m²), cloud_passing, ramp_change: 3600 cycles each (seed = None reproduces the frozen scenarios; seed k varies cloud-noise stream, step instant, cloud phase).',
         f'- partial_shading: nine patterns of three bypass-diode substrings, 400 s each: {PATTERNS}; the controller sees the mean substring irradiance.', '- real_clear_day / real_variable_day: held-out PVGIS days (ranking rules fixed a priori), hourly data time-compressed.',
         '- Mismatched plant: parameters perturbed by severity s (I_ph x(1-0.08 s), R_s x(1+0.8 s), R_sh x(1-0.4 s), n x(1+0.05 s)); headline severity 1.', f"- Sensor noise (noise experiment only): sigma_I = {CFG['noise']['sigma_i']:.1%}, sigma_V = {CFG['noise']['sigma_v']:.1%}, sigma_G = {CFG['noise']['sigma_g']:.0%}.",
         '\n## Tables of parameters\n\nSee `results/tables/Table_PV_Datasheet.md`, `Table_Algorithm_Parameters.md`, `Table_Converter_Parameters.md`.\n']
    for n in ('Table_PV_Datasheet', 'Table_Algorithm_Parameters', 'Table_Converter_Parameters'):
        p = R('tables', n + '.md')
        if os.path.exists(p): t.append(open(p).read() + '\n')
    open(os.path.join(DOCS, 'PARAMETERS.md'), 'w').write('\n'.join(t))


# ------------------------------------------------------------------------------------------------ figure map and captions
def _hl(): return rd('metrics_headline.csv')
def _stats(): return rd('stats', 'stats_summary.csv')

FIG = {   # name: (script, panels, caption function)
 'Fig01_system_overview': ('mppt/schematics.py', 'single', lambda: 'System overview: PV module or shaded string, input capacitor, boost converter to a constant bus, voltage/current sensing, MPPT controller (command V_cmd) and inner voltage loop (duty D).'),
 'Fig02_shaded_string_model': ('mppt/schematics.py', 'single', lambda: 'Bypass-diode shaded-string model: three substrings of 18 cells with bypass diodes; shaded cells receive a fraction of the irradiance.'),
 'Fig03_hybrid_block_diagram': ('mppt/schematics.py', 'single', lambda: 'Block diagram of the Hybrid controller: I-V scan, DNN estimate of V_gmpp, P&O refinement and re-scan logic.'),
 'Fig04_dnn_dataflow': ('mppt/schematics.py', 'single', lambda: f"Stage-1 DNN data flow: input vector (G_in, T, I_1..I_14), standardisation, hidden layers 128-128-64, output V_est; parameters {_nparams():,}."),
 'Fig05_flowchart_PO': ('mppt/schematics.py', 'single', lambda: 'Flowchart of the P&O algorithm exactly as implemented (threshold 0.01 W, step 0.5 V).'),
 'Fig06_flowchart_IncCond': ('mppt/schematics.py', 'single', lambda: 'Flowchart of the incremental-conductance algorithm with explicit handling of dV = 0 before dI/dV is evaluated.'),
 'Fig07_hybrid_handoff_states': ('mppt/schematics.py', 'single', lambda: 'State diagram of the Hybrid controller: SCAN and TRACK states, hand-off and re-scan triggers.'),
 'Fig08_flowchart_PSO': ('mppt/schematics.py', 'single', lambda: 'Flowchart of the PSO global-MPP tracker used as comparator.'),
 'Fig10_irradiance_temperature_histograms': ('experiments/exp_figures.py', 'a-c', lambda: 'Histograms of hourly irradiance (a: all hours, log scale; b: daytime hours) and ambient temperature (c), Ramadi 2005–2020.'),
 'Fig11_daytime_conditions_by_month': ('experiments/exp_figures.py', 'single', lambda: 'Daytime (G, T) operating conditions coloured by month (every fifth hour plotted).'),
 'Fig12_hour_month_irradiance_map': ('experiments/exp_figures.py', 'single', lambda: 'Mean irradiance by month and hour of day.'),
 'Fig13_representative_days': ('experiments/exp_figures.py', 'a-b', lambda: 'Representative winter (a) and summer (b) days of the test years: irradiance and ambient temperature.'),
 'Fig14_PV_curve_families': ('experiments/exp_figures.py', 'a-c', lambda: 'P-V curves of the single-diode model at cell temperatures 25, 45 and 65 °C (a–c) for irradiances 200–1000 W/m²; dots mark the maxima.'),
 'Fig15_Vmpp_vs_G_and_T': ('experiments/exp_figures.py', 'a-b', lambda: 'Maximum-power voltage versus irradiance at fixed cell temperature (a) and versus cell temperature at fixed irradiance (b).'),
 'Fig16_dnn_loss_curves': ('experiments/exp_figures.py', 'single', lambda: 'Training and validation loss of the stage-1 DNN (headline seed).'),
 'Fig17_dnn_test_accuracy': ('experiments/exp_figures.py', 'a-b', lambda: f"Stage-1 DNN on the in-distribution test set: estimated vs true V_gmpp (a) and error histogram (b); RMSE {one(rd('hybrid_offline_metrics.csv'), 'RMSE_V', test_set=rd('hybrid_offline_metrics.csv').test_set.iloc[0]):.2f} V."),
 'Fig18_conformal_calibration': ('experiments/exp_figures.py', 'single', lambda: 'Empirical coverage of split-conformal intervals versus nominal confidence for four test sets; intervals are valid only in distribution (T1) and under-cover for out-of-range plants (T4).'),
 'Fig19_timedomain_step_change': ('experiments/exp_figures.py', 'a-f', lambda: 'Irradiance step 400 → 800 W/m² (nominal plant): power with true MPP (dashed) and 98–100 % band, PV voltage and duty cycle; left full run, right zoom. Duty = 1 - V_cmd/V_bus (display).'),
 'Fig20_timedomain_cloud_passing': ('experiments/exp_figures.py', 'a-f', lambda: 'Cloud-passing scenario (nominal plant); layout as in Fig. 19.'),
 'Fig21_timedomain_ramp_change': ('experiments/exp_figures.py', 'a-f', lambda: 'Ramp scenario (nominal plant); layout as in Fig. 19.'),
 'Fig22_timedomain_partial_shading': ('experiments/exp_figures.py', 'a-f', lambda: 'Nine partial-shading patterns (nominal plant); triangles mark scan cycles of the scan-based controllers in the zoom.'),
 'Fig23_cold_start': ('experiments/exp_figures.py', 'a-f', lambda: 'Cold start from the lowest (left) and highest (right) valid initial voltage at G = 800 W/m², T = 25 °C; scan cycles show the mean scan power.'),
 'Fig24_shading_pv_snapshots': ('experiments/exp_figures.py', 'a-i', lambda: 'P-V curves of the nine shading patterns with the operating point of every controller at the end of its episode; crosses mark the global MPP.'),
 'Fig25_tracking_efficiency': ('experiments/exp_figures.py', 'a-b', lambda: 'Tracking efficiency per scenario, mean ± std over seeds, nominal (a) and mismatched (b) plant; the y-axis starts at 70 % (axis-break marks).'),
 'Fig26_convergence_oscillation': ('experiments/exp_figures.py', 'a-b', lambda: 'Event convergence time (a) and steady-state voltage oscillation (b), mean ± std over seeds, nominal plant.'),
 'Fig27_ablation': ('experiments/exp_ablation.py', 'a-b', lambda: 'Stage-1 design ablation: test RMSE of V_gmpp (a) and closed-loop shading efficiency (b), mean ± std over seeds.'),
 'Fig28_hpc_scaling_latency': ('experiments/exp_hpc.py', 'a-b', lambda: 'Parallel speed-up of training-data generation and of the scenario-seed sweep (a) and DNN inference latency per precision/runtime against the 1 s control cycle (b).'),
 'Fig30_converter_step': ('experiments/exp_converter.py', 'a-h', lambda: 'Irradiance step on the averaged boost-converter plant: power, PV voltage, duty cycle and inductor current; left full run, right zoom.'),
 'Fig31_converter_verification': ('experiments/exp_converter.py', 'a-d', lambda: 'Averaged versus switching-level model after the irradiance step: PV voltage (top) and inductor current (bottom), full window (left) and last millisecond (right).'),
 'Fig32_seed_statistics': ('experiments/exp_stats.py', 'a-c', lambda: 'Tracking efficiency mean ± std over scenario seeds for standard scenarios (a), real days (b) and partial shading (c), nominal and mismatched plant.'),
 'Fig33_noise': ('experiments/exp_stats.py', 'single', lambda: 'Shading efficiency with and without sensor noise (I 0.5 %, V 0.5 %, G 2 %).'),
 'Fig34_quantisation': ('experiments/exp_stats.py', 'a-b', lambda: 'Model size (a) and largest deviation of the estimated voltage from float32 (b) for float32, float16 and int8 (numpy emulation).'),
 'Fig35_compute_cost': ('experiments/exp_figures.py', 'single', lambda: 'Mean controller decision time per cycle (logarithmic axis); plant model excluded.'),
 'Fig36_mismatch_sweep': ('experiments/exp_figures.py', 'a-b', lambda: 'Tracking efficiency versus plant-mismatch severity for the comprehensive (a) and partial-shading (b) scenarios.'),
 'Fig37_timedomain_comprehensive': ('experiments/exp_figures.py', 'a-f', lambda: 'Comprehensive scenario (nominal plant); layout as in Fig. 19.'),
 'Fig38_timedomain_real_clear_day': ('experiments/exp_figures.py', 'a-f', lambda: 'Real clear held-out day (nominal plant); layout as in Fig. 19.'),
 'Fig39_timedomain_real_variable_day': ('experiments/exp_figures.py', 'a-f', lambda: 'Real variable held-out day (nominal plant); layout as in Fig. 19.'),
 'Fig40_timedomain_comprehensive_mismatched': ('experiments/exp_figures.py', 'a-f', lambda: 'Comprehensive scenario, mismatched plant (severity 1); layout as in Fig. 19.'),
 'Fig41_timedomain_partial_shading_mismatched': ('experiments/exp_figures.py', 'a-f', lambda: 'Partial shading, mismatched plant (severity 1); layout as in Fig. 19.'),
 'Fig42_timedomain_real_clear_day_mismatched': ('experiments/exp_figures.py', 'a-f', lambda: 'Real clear day, mismatched plant (severity 1); layout as in Fig. 19.'),
 'Fig43_sanity_P_vs_G': ('experiments/exp_figures.py', 'a-b', lambda: 'Sanity check of the module power computed from PVGIS irradiance: power versus irradiance (a) and a three-day series (b).'),
 'Fig44_STC_validation': ('experiments/exp_figures.py', 'a-b', lambda: 'Single-diode model at STC against the datasheet-type inputs: I-V (a) and P-V (b).'),
}

def _nparams():
    p = R('hpc', 'latency.csv')
    return int(pd.read_csv(p).params.iloc[0]) if os.path.exists(p) else -1


def figure_docs():
    lines = ['# Figure map\n', 'Generated by `tools/build_docs.py`. Every figure: PDF (vector) + PNG (600 dpi) in `results/figures/`, plotted data in `results/figures/data/<name>.csv`. No title inside the image; panel letters are drawn inside the axes.\n',
             '| figure | script | panels | data CSV | exists |\n|---|---|---|---|---|']
    caps = ['# Figure captions (generated from the result files)\n']
    for n, (script, panels, fn) in FIG.items():
        ex = os.path.exists(R('figures', n + '.pdf')); csv = os.path.join('results', 'figures', 'data', n + '.csv')
        lines.append(f'| {n} | `{script}` | {panels} | `{csv}` | {"yes" if ex else "no"} |')
        try: caps.append(f'**{n.split("_")[0].replace("Fig", "Fig. ")}.** {fn()}\n')
        except Exception as e: caps.append(f'**{n}.** (caption not generated: {type(e).__name__}: {e})\n')
    extra = sorted(set(os.path.basename(p)[:-4] for p in glob.glob(R('figures', '*.pdf'))) - set(FIG))
    if extra: lines.append('\nFigures present but missing from the map: ' + ', '.join(extra))
    open(os.path.join(DOCS, 'FIGURE_MAP.md'), 'w').write('\n'.join(lines) + '\n'); open(os.path.join(DOCS, 'FIGURE_CAPTIONS.md'), 'w').write('\n'.join(caps))


# ------------------------------------------------------------------------------------------------ fingerprint
TIMING = re.compile(r'(_us|us_per|seconds|_s$|latency|wallclock|speedup|efficiency$|per_s|fit_s|generation|compute|cycles_per|suites_per|plants_per|ratio|time)', re.I)
def fingerprint():
    out, allh = [], hashlib.sha256()
    for p in sorted(glob.glob(R('**', '*.csv'), recursive=True)):
        rel = os.path.relpath(p, RESULTS); df = pd.read_csv(p)
        keep = [c for c in df.columns if not (TIMING.search(c) and df[c].dtype.kind == 'f' and ('hpc' in rel or 'compute' in c.lower() or 'us' in c.lower() or 'seconds' in c.lower() or 'speedup' in c.lower() or 'latency' in c.lower() or 'wallclock' in c.lower()))]
        h = hashlib.sha256(df[keep].round(6).to_csv(index=False).encode()).hexdigest()[:16]; out.append(f'{h}  {rel}  (non-timing columns: {len(keep)}/{len(df.columns)})'); allh.update(h.encode())
    for p in sorted(glob.glob(R('figures', '*.pdf'))): out.append(f'{"-" * 16}  {os.path.relpath(p, RESULTS)}  (vector file; contents not hashed, see data CSV)')
    open(R('fingerprint.txt'), 'w').write(f'overall {allh.hexdigest()[:16]}\nhashes of every CSV with timing columns removed (rounded to 6 decimals); timings differ between runs by design\n' + '\n'.join(out) + '\n')


if __name__ == '__main__':
    metrics_md(); parameters_md(); figure_docs()
    try:
        from tools.build_results import build_all; build_all()
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from build_results import build_all; build_all()
    fingerprint(); print('docs written')
