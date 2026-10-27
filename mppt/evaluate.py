"""Controller factory and the evaluation suite (scenarios x plants x controllers) producing run logs and metric tables with metrics.py."""
import numpy as np, pandas as pd
import metrics as M
from .config import CFG
from .controllers import (PerturbAndObserveMPPT, IncrementalConductanceMPPT, PolyMPPT, ANNMPPT, ScanPOMPPT, PSOMPPT)
from .plants import make_plant_and_truth, new_plant
from .sim import run_algorithm
from .scenarios import STD4, REAL

HY, PO, IC, P2, AN, SC, DO, PS, TR = 'Hybrid (DNN+P&O)', 'P&O', 'IncCond', 'Poly2(G,T)-MPPT', 'ANN-MPPT', 'Scan+P&O (no ML)', 'DNN scan-only (no P&O)', 'PSO', 'Tree scan+P&O'
CORE = [PO, IC, P2, AN, SC, DO, HY]
ALL_CTRL = CORE + [PS, TR]


def build_controllers(ann=None, poly=None, dnn_est=None, tree_est=None, pso_seed=0, names=ALL_CTRL):
    """name -> factory(v_start=None). Predictors/estimators are passed in (trained elsewhere)."""
    f = {PO: lambda v=None: PerturbAndObserveMPPT(v_start=v), IC: lambda v=None: IncrementalConductanceMPPT(v_start=v), SC: lambda v=None: ScanPOMPPT(SC, 'argmax', v_start=v),
         PS: lambda v=None: PSOMPPT(seed=pso_seed, v_start=v)}
    if poly is not None: f[P2] = lambda v=None: PolyMPPT(poly, v_start=v)
    if ann is not None: f[AN] = lambda v=None: ANNMPPT(ann, v_start=v)
    if dnn_est is not None:
        f[DO] = lambda v=None: ScanPOMPPT(DO, 'dnn_only', dnn_est, v_start=v); f[HY] = lambda v=None: ScanPOMPPT(HY, 'dnn', dnn_est, v_start=v)
    if tree_est is not None: f[TR] = lambda v=None: ScanPOMPPT(TR, 'tree', tree_est, v_start=v)
    return {n: f[n] for n in names if n in f}


def run_suite(scenarios, controllers, pvm, plants=(('nominal', 0.0), ('mismatched', None)), only=None, noise_seed=None, plant_factory=None):
    """logs[plant][scenario][controller] = DataFrame. plants: (label, severity) with None -> CFG['mismatch_default']."""
    logs = {}
    for label, sev in plants:
        sev = CFG['mismatch_default'] if sev is None else sev
        model = pvm.perturbed(sev); logs[label] = {}
        for sname, scn in scenarios.items():
            if only and sname not in only: continue
            _, truth = make_plant_and_truth(scn, model)
            logs[label][sname] = {}
            for cname, fac in controllers.items():
                plant = new_plant(scn, model) if plant_factory is None else plant_factory(scn, model)
                rng = None if noise_seed is None else np.random.RandomState(noise_seed)
                logs[label][sname][cname] = run_algorithm(fac(), plant, scn, truth, noise_rng=rng)
    return logs


def metrics_table(logs, scenarios):
    rows = []
    for plant, d1 in logs.items():
        for sname, d2 in d1.items():
            ev = scenarios[sname]['events']
            for cname, df in d2.items():
                r = {'plant': plant, 'scenario': sname, 'controller': cname, **M.summarise_run(df, ev)}
                r['gmpp_success_%'] = M.gmpp_success(df.P, df.mpp_power, df.scan, ev)[0] if sname == 'partial_shading' else np.nan
                rows.append(r)
    return pd.DataFrame(rows)


def cold_start(pvm, controllers_fn, voc_fractions=None, v0_list=None):
    """Cold start at (G,T) = CFG['startup_cond']; protocol A: fixed list, B: fractions of the actual Voc. controllers_fn(v0) -> dict name -> controller."""
    from .scenarios import startup_scenario
    from .plants import ModulePlant, truth_module
    scn = startup_scenario(); truth = truth_module(scn, pvm); plant = ModulePlant(pvm)
    voc = float(pvm.voc(scn['G'][0], pvm.cell_temperature(scn['G'][0], scn['T'][0])))
    starts = [('A', v) for v in (CFG['startup_v0_list'] if v0_list is None else v0_list)] + [('B', round(f * voc, 3)) for f in (CFG['startup_voc_fractions'] if voc_fractions is None else voc_fractions)]
    rows, logs = [], {}
    for pr, v0 in starts:
        for cname, ctrl in controllers_fn(v0).items():
            df = run_algorithm(ctrl, ModulePlant(pvm), scn, truth); logs[(pr, v0, cname)] = df
            rows.append({'protocol': pr, 'V0': v0, 'controller': cname, 'convergence_cycles': M.convergence_cycles(df.efficiency.to_numpy(), [0])[0]})
    return pd.DataFrame(rows), logs
