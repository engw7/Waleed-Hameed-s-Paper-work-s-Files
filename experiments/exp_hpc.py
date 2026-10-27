"""Editor request: where does this work need parallel compute, and where does it need real-time inference?  Everything below is MEASURED on the machine that runs it
(hardware is recorded in results/hpc/hardware.json); nothing is estimated except where a column is explicitly labelled so.
Outputs: results/hpc/{hardware.json, generation_scaling.csv, sweep_scaling.csv, throughput.csv, latency.csv, scan_overhead.csv, hpc_summary.csv},
results/tables/Table_HPC.{csv,md,tex} (written by the table module), results/figures/Fig28_hpc_scaling_latency.{pdf,png}."""
import os, sys, json, time, platform, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, pandas as pd

CYCLE_BUDGET_S = 1.0                                     # control cycle of the simulation (1 s per cycle, see docs/PARAMETERS.md)


def _hash(d):
    h = hashlib.sha256()
    for k in sorted(d): h.update(k.encode()); h.update(np.ascontiguousarray(d[k]).tobytes())
    return h.hexdigest()[:16]


def hardware():
    cpu = next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), platform.processor()) if os.path.exists('/proc/cpuinfo') else platform.processor()
    return dict(cpu_model=cpu, logical_cpus=os.cpu_count(), python=platform.python_version(), numpy=np.__version__, platform=platform.platform())


# ---------- (1) parallel dataset generation ----------
def generation_scaling(R):
    import pv_hybrid_data as hd
    from mppt.data import get_pv_series
    from mppt.models import hybrid_bases
    from mppt.config import CFG
    pv, _, _ = get_pv_series(); b = hybrid_bases(pv)
    n, chunk, reps = (3000, 500, 2) if CFG['quick'] else (24000, 1500, 3)
    rows, ref = [], None
    for w in (1, 2, 4, 8):
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); d = hd.generate(n, *b['train'], 'dr', 'mix', 7, n_workers=w, chunk=chunk); ts.append(time.perf_counter() - t0)
        hsh = _hash(d); ref = hsh if ref is None else ref
        rows.append(dict(workers=w, samples=n, chunks=-(-n // chunk), wallclock_s=float(np.median(ts)), wallclock_min_s=float(np.min(ts)), repetitions=reps, output_hash=hsh, identical_to_serial=hsh == ref))
    df = pd.DataFrame(rows); t1 = df.wallclock_s.iloc[0]
    df['speedup'] = t1 / df.wallclock_s; df['efficiency'] = df.speedup / df.workers; df['plants_per_s'] = df.samples / df.wallclock_s
    df['oversubscribed'] = df.workers > (os.cpu_count() or 1)
    df.to_csv(os.path.join(R, 'generation_scaling.csv'), index=False); return df


# ---------- (2) batch simulation throughput ----------
_G = {}
def _init_worker():
    from mppt.data import get_pv_series
    _G['pv'], _G['pvm'], _ = get_pv_series()


def _suite_job(args):
    """One unit of the design-space/statistics workload: the 9-pattern shading suite for one scenario seed, both plants, given controllers."""
    seed, mlp, names = args
    from mppt.evaluate import build_controllers, run_suite
    from mppt.scenarios import create_scenarios
    from mppt.nn import ScanEstimator
    sc = {'partial_shading': create_scenarios(_G['pv'], seed)['partial_shading']}
    ctr = build_controllers(dnn_est=ScanEstimator('dnn', mlp), names=names)
    t0 = time.perf_counter(); logs = run_suite(sc, ctr, _G['pvm'], only=['partial_shading']); dt = time.perf_counter() - t0
    n_cycles = sum(len(df) for d1 in logs.values() for d2 in d1.values() for d3 in d2.values() for df in [d3])
    return dict(seed=seed, seconds=dt, cycles=n_cycles, runs=len(ctr) * len(logs))


def throughput_and_sweep(R, mlp):
    from mppt.config import CFG
    from mppt.evaluate import PO, IC, SC, HY, DO, PS
    from mppt.data import get_pv_series
    import multiprocessing as mp
    names = [PO, IC, SC, DO, HY] + ([] if CFG['quick'] else [PS])
    pv, pvm, _ = get_pv_series(); _G['pv'], _G['pvm'] = pv, pvm
    seeds = list(range(2 if CFG['quick'] else 8))
    # serial throughput per controller (one shading suite = 9 patterns x 400 s, 3600 cycles, 2 plants)
    from mppt.evaluate import build_controllers, run_suite
    from mppt.scenarios import create_scenarios
    from mppt.nn import ScanEstimator
    sc = {'partial_shading': create_scenarios(pv, 0)['partial_shading']}; rows = []
    for nm in names:
        ctr = build_controllers(dnn_est=ScanEstimator('dnn', mlp), names=[nm]); ts = []
        for _ in range(1 if CFG['quick'] else 3):
            t0 = time.perf_counter(); run_suite(sc, ctr, pvm, only=['partial_shading']); ts.append(time.perf_counter() - t0)
        t = float(np.median(ts)); cyc = 2 * len(sc['partial_shading']['time'])
        rows.append(dict(controller=nm, workload='9-pattern shading suite, nominal + mismatched plant', cycles=cyc, seconds_serial=t, suites_per_s=1 / t, cycles_per_s=cyc / t, us_per_cycle_incl_plant=1e6 * t / cyc))
    th = pd.DataFrame(rows); th.to_csv(os.path.join(R, 'throughput.csv'), index=False)
    # process-parallel sweep over scenario seeds (what the multi-seed statistics and ablations do)
    srows, t1 = [], None
    for w in ((1, 2) if CFG['quick'] else (1, 2, 4, 8)):
        jobs = [(s, mlp, names) for s in seeds]; t0 = time.perf_counter()
        if w == 1:
            _init_worker() if not _G else None; res = [_suite_job(j) for j in jobs]
        else:
            with mp.get_context('spawn').Pool(w, initializer=_init_worker) as p: res = p.map(_suite_job, jobs, chunksize=1)
        dt = time.perf_counter() - t0; t1 = dt if t1 is None else t1
        srows.append(dict(workers=w, jobs=len(jobs), runs=sum(r['runs'] for r in res), cycles=sum(r['cycles'] for r in res), wallclock_s=dt, speedup=t1 / dt, efficiency=t1 / dt / w, scenarios_per_s=len(jobs) / dt, cycles_per_s=sum(r['cycles'] for r in res) / dt))
    sw = pd.DataFrame(srows); sw['oversubscribed'] = sw.workers > (os.cpu_count() or 1); sw.to_csv(os.path.join(R, 'sweep_scaling.csv'), index=False)
    return th, sw


# ---------- (3) latency vs real-time budget ----------
def _lat(fn, n_warm=200, n=3000):
    for _ in range(n_warm): fn()
    t = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter_ns(); fn(); t[i] = time.perf_counter_ns() - t0
    return t / 1e3                                              # microseconds


def latency(R, trained, X_calib):
    from mppt.nn import QuantMLP
    mlp = trained.mlp; x = np.asarray(X_calib[:1], float); rows = []
    variants = [('float32 (numpy, as used in the closed loop)', lambda: mlp.predict(x), mlp.n_params() * 4),
                ('float16 weights (numpy emulation)', None, None), ('int8 weights/activations (numpy emulation)', None, None)]
    q16, q8 = QuantMLP(mlp, 'fp16'), QuantMLP(mlp, 'int8', X_calib)
    variants[1] = (variants[1][0], lambda: q16.predict(x), q16.n_bytes()); variants[2] = (variants[2][0], lambda: q8.predict(x), q8.n_bytes())
    import subprocess, tempfile, json as _json
    from mppt.config import CACHE
    xs = trained.sx.transform(x).astype('float32'); tmp = tempfile.mkdtemp(); mp_ = os.path.join(tmp, 'm.keras'); trained.model.save(mp_); np.save(os.path.join(tmp, 'x.npy'), xs)
    subprocess.check_call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools', 'bench_keras.py'), mp_, os.path.join(tmp, 'x.npy'), os.path.join(tmp, 'o.json')], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    kb = _json.load(open(os.path.join(tmp, 'o.json'))); KT = {'call': np.array(kb['call']), 'predict': np.array(kb['predict'])}
    variants += [('Keras model(x) call (float32, framework overhead, separate process)', ('K', 'call'), mp_ and mlp.n_params() * 4), ('Keras model.predict (float32, framework overhead, separate process)', ('K', 'predict'), mlp.n_params() * 4)]
    for nm, fn, nbytes in variants:
        if isinstance(fn, tuple): t = KT[fn[1]]; n = len(t)
        else: n = 3000; t = _lat(fn, 200, n)
        rows.append(dict(variant=nm, params=mlp.n_params(), macs=mlp.macs(), model_bytes=nbytes, median_us=float(np.median(t)), p99_us=float(np.percentile(t, 99)), max_us=float(t.max()), n_calls=n,
                         cycle_budget_s=CYCLE_BUDGET_S, fraction_of_cycle_budget_median=float(np.median(t)) * 1e-6 / CYCLE_BUDGET_S, fraction_of_cycle_budget_p99=float(np.percentile(t, 99)) * 1e-6 / CYCLE_BUDGET_S))
    # accuracy cost of the reduced precision on held-out calibration inputs (same units as the estimator: volts)
    ref = mlp.predict(X_calib); acc = pd.DataFrame([dict(variant='float16', max_abs_dv=float(np.abs(q16.predict(X_calib) - ref).max()), mean_abs_dv=float(np.abs(q16.predict(X_calib) - ref).mean())),
                                                     dict(variant='int8', max_abs_dv=float(np.abs(q8.predict(X_calib) - ref).max()), mean_abs_dv=float(np.abs(q8.predict(X_calib) - ref).mean()))])
    acc.to_csv(os.path.join(R, 'precision_deviation_vs_float32.csv'), index=False)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(R, 'latency.csv'), index=False); return df


def scan_overhead(R, mlp, pvm, pv):
    import metrics as M
    from mppt.evaluate import build_controllers, run_suite, HY, SC
    from mppt.scenarios import create_scenarios
    from mppt.nn import ScanEstimator
    sc = {'partial_shading': create_scenarios(pv, None)['partial_shading']}
    logs = run_suite(sc, build_controllers(dnn_est=ScanEstimator('dnn', mlp), names=[SC, HY]), pvm, only=['partial_shading']); rows = []
    for plant, d in logs.items():
        for nm, df in d['partial_shading'].items():
            rows.append(dict(plant=plant, controller=nm, cycles=len(df), scan_cycles=int(df.scan.sum()), scan_cycle_fraction_pct=100 * float(df.scan.mean()), scan_energy_overhead_pct=M.scan_overhead(df.P, df.mpp_power, df.scan),
                             probes_per_scan=14, scan_assumed_instantaneous_one_cycle=True))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(R, 'scan_overhead.csv'), index=False); return df


def run(trained=None, data=None):
    from mppt.config import RESULTS, CFG, CACHE
    from mppt.data import get_pv_series
    from mppt import models as MD
    R = os.path.join(RESULTS, 'hpc'); os.makedirs(R, exist_ok=True)
    json.dump(hardware(), open(os.path.join(R, 'hardware.json'), 'w'), indent=1)
    pv, pvm, _ = get_pv_series()
    if trained is None:
        p = os.path.join(CACHE, 'hybrid_seed42')
        if os.path.exists(p + '.keras'): trained = MD.Trained.load(p)
        else:
            data, _c = MD.hybrid_data(pv); trained = MD.train_hybrid_dnn(data, seed=CFG['seed'], name='Hybrid DNN seed42').trained
    if data is None: data, _c = MD.hybrid_data(pv)
    X_calib = MD.hybrid_features(data['val'])[:2000]
    gen = generation_scaling(R); th, sw = throughput_and_sweep(R, trained.mlp); lat = latency(R, trained, X_calib); so = scan_overhead(R, trained.mlp, pvm, pv)
    ident = bool(gen.identical_to_serial.all())
    f32 = lat.iloc[0]; i8 = lat[lat.variant.str.startswith('int8')].iloc[0]; kp = lat[lat.variant.str.contains('predict')].iloc[0]
    hy = so[(so.controller == 'Hybrid (DNN+P&O)')]
    summ = pd.DataFrame([
        dict(quantity='logical CPUs of this machine', value=os.cpu_count(), unit='-', note='worker counts above this value are oversubscribed'),
        dict(quantity='data generation speed-up at 4 workers', value=float(gen.set_index('workers').speedup.get(4, float('nan'))), unit='x', note=f'serial {gen.wallclock_s.iloc[0]:.1f} s for {int(gen.samples.iloc[0])} plants; includes pool start-up'),
        dict(quantity='data generation efficiency at 4 workers', value=float(gen.set_index('workers').efficiency.get(4, float('nan'))), unit='-', note=''),
        dict(quantity='parallel output bit-identical to serial (1/2/4/8 workers)', value=ident, unit='bool', note='SHA-256 of all arrays'),
        dict(quantity='scenario-seed sweep speed-up at 4 workers', value=float(sw.set_index('workers').speedup.get(4, float('nan'))), unit='x', note='process-parallel 9-pattern suites'),
        dict(quantity='Hybrid controller throughput (serial, incl. plant model)', value=float(th[th.controller == 'Hybrid (DNN+P&O)'].cycles_per_s.iloc[0]), unit='cycles/s', note='simulation speed, not controller speed'),
        dict(quantity='DNN inference latency float32 numpy (median)', value=float(f32.median_us), unit='us', note=f'{f32.fraction_of_cycle_budget_median:.2e} of the {CYCLE_BUDGET_S:.0f} s cycle'),
        dict(quantity='DNN inference latency int8 numpy emulation (median)', value=float(i8.median_us), unit='us', note='emulation; not an MCU measurement'),
        dict(quantity='DNN latency via Keras model.predict (median)', value=float(kp.median_us), unit='us', note=f'{kp.fraction_of_cycle_budget_median:.2e} of the cycle; framework overhead'),
        dict(quantity='Hybrid scan-cycle fraction of all cycles (nominal plant)', value=float(hy[hy.plant == 'nominal'].scan_cycle_fraction_pct.iloc[0]), unit='%', note='9-pattern suite'),
        dict(quantity='Hybrid scan energy overhead (nominal plant)', value=float(hy[hy.plant == 'nominal'].scan_energy_overhead_pct.iloc[0]), unit='% of available energy', note='scan cycle assumed one control cycle')])
    summ.to_csv(os.path.join(R, 'hpc_summary.csv'), index=False)
    figure_and_table(dict(generation=gen, sweep=sw, throughput=th, latency=lat, scan=so, summary=summ))
    return dict(generation=gen, sweep=sw, throughput=th, latency=lat, scan=so, summary=summ)




def figure_and_table(out):
    """Fig_HPC (a) speed-up of data generation and of the seed sweep vs workers (with ideal line), (b) latency per inference vs the 1 s cycle budget; Table_HPC."""
    import matplotlib.pyplot as plt
    import mppt_style as sty
    from mppt.tables import save_table
    from mppt.config import CFG
    sty.apply_style(); gen, sw, lat = out['generation'], out['sweep'], out['latency']
    fig, (a, b) = plt.subplots(1, 2, figsize=(sty.W2, 2.9), gridspec_kw=dict(width_ratios=[1, 1.25]))
    w = gen.workers.to_numpy(); a.plot(w, w, 'k--', lw=1.0, label='Ideal'); a.plot(gen.workers, gen.speedup, 'o-', color=sty.MATLAB[0], label='Training-data generation'); a.plot(sw.workers, sw.speedup, 's-', color=sty.MATLAB[1], label='Scenario-seed sweep')
    ncpu = os.cpu_count() or 1
    if w.max() > ncpu: a.axvspan(ncpu + 0.25, w.max() + 0.5, color='0.9', zorder=0); a.text(ncpu + 0.4, 0.35 * w.max(), f'more workers\nthan {ncpu} CPUs', fontsize=7, va='bottom')
    a.set_xlabel('Worker processes'); a.set_ylabel('Speed-up over serial'); a.set_xticks(w); a.legend(loc='upper left', bbox_to_anchor=(0.0, 0.88)); sty.panel(a, 'a')
    lab = ['float32\nnumpy', 'float16\nnumpy', 'int8\nnumpy', 'Keras\nmodel(x)', 'Keras\npredict']; vals = lat.median_us.to_numpy(); p99 = lat.p99_us.to_numpy()
    x = np.arange(len(lab)); b.bar(x, vals, 0.6, color=[sty.MATLAB[0], sty.MATLAB[4], sty.MATLAB[3], sty.MATLAB[2], sty.MATLAB[6]], edgecolor='k', lw=0.6, label='Median')
    b.plot(x, p99, 'kv', ms=4, label='99th percentile'); b.axhline(CYCLE_BUDGET_S * 1e6, color='k', ls='-', lw=1.2); b.text(len(lab) - 0.5, CYCLE_BUDGET_S * 1e6 * 0.55, 'control cycle = 1 s', ha='right', fontsize=7.5)
    b.set_yscale('log'); b.set_xticks(x); b.set_xticklabels(lab, fontsize=7.5); b.set_ylabel('Latency per inference (\u00b5s)'); b.set_ylim(top=3e6); b.legend(loc='center left', bbox_to_anchor=(0.0, 0.62)); sty.panel(b, 'b')
    for xi, v, q in zip(x, vals, p99): b.text(xi, max(v, q) * 1.45, f'{v:.0f}', ha='center', fontsize=7)
    fig.tight_layout(); fd = pd.concat([gen.assign(series='generation')[['series', 'workers', 'speedup', 'efficiency']], sw.assign(series='seed_sweep')[['series', 'workers', 'speedup', 'efficiency']],
                                         lat.assign(series='latency')[['series', 'variant', 'median_us', 'p99_us']]], ignore_index=True)
    sty.save_figure(fig, 'Fig28_hpc_scaling_latency', fd)
    sc_ = out['scan']; hy = sc_[(sc_.plant == 'nominal') & (sc_.controller == 'Hybrid (DNN+P&O)')].iloc[0]
    s = out['summary']; g = gen.set_index('workers'); q = sw.set_index('workers')
    tb = pd.DataFrame([
        dict(Part=f"Training-data generation (hybrid_n_train = {CFG['hybrid_n_train']:,} plants in the full run)", Need='parallel compute', Measure='speed-up / efficiency at 2, 4, 8 workers', Result='; '.join(f'{int(k)}: {g.speedup[k]:.2f}x / {g.efficiency[k]:.2f}' for k in (2, 4, 8)), Bit_identical='yes' if out['generation'].identical_to_serial.all() else 'NO'),
        dict(Part='Design-space sweeps, multi-seed statistics', Need='parallel compute', Measure='speed-up / efficiency at 2, 4, 8 workers (shading suites)', Result='; '.join(f'{int(k)}: {q.speedup.get(k, float("nan")):.2f}x / {q.efficiency.get(k, float("nan")):.2f}' for k in (2, 4, 8)), Bit_identical='-'),
        dict(Part='Simulation throughput (Hybrid, serial)', Need='-', Measure='controller cycles per second incl. plant model', Result=f"{out['throughput'].set_index('controller').cycles_per_s['Hybrid (DNN+P&O)']:.0f} cycles/s", Bit_identical='-'),
        dict(Part='Online DNN inference (float32, numpy)', Need='real-time, not HPC', Measure='median / p99 latency vs 1 s cycle', Result=f"{lat.median_us.iloc[0]:.0f} / {lat.p99_us.iloc[0]:.0f} us = {lat.fraction_of_cycle_budget_median.iloc[0]:.1e} of the cycle", Bit_identical='-'),
        dict(Part='Online DNN inference (int8, numpy emulation)', Need='real-time, not HPC', Measure='median / p99 latency', Result=f"{lat.median_us.iloc[2]:.0f} / {lat.p99_us.iloc[2]:.0f} us", Bit_identical='-'),
        dict(Part='Online DNN inference (Keras predict)', Need='real-time, not HPC', Measure='median / p99 latency', Result=f"{lat.median_us.iloc[4]:.0f} / {lat.p99_us.iloc[4]:.0f} us = {lat.fraction_of_cycle_budget_median.iloc[4]:.1e} of the cycle", Bit_identical='-'),
        dict(Part='Scan overhead (Hybrid, shading suite, nominal plant)', Need='real-time', Measure='share of cycles / share of available energy', Result=f"{hy.scan_cycle_fraction_pct:.2f} % / {hy.scan_energy_overhead_pct:.2f} %", Bit_identical='-')])
    tb.columns = [c.replace('_', '-') for c in tb.columns]
    save_table('Table_HPC', tb, 'Where parallel compute and where real-time inference are needed (measured on the machine recorded in results/hpc/hardware.json).')
    return tb


def from_csv():
    from mppt.config import RESULTS
    R = os.path.join(RESULTS, 'hpc'); rd = lambda n: pd.read_csv(os.path.join(R, n))
    return dict(generation=rd('generation_scaling.csv'), sweep=rd('sweep_scaling.csv'), throughput=rd('throughput.csv'), latency=rd('latency.csv'), scan=rd('scan_overhead.csv'), summary=rd('hpc_summary.csv'))


if __name__ == '__main__':
    if '--figure-only' in sys.argv: figure_and_table(from_csv())
    else:
        out = run()
        for k, v in out.items(): print('\n==', k); print(v.round(4).to_string())
