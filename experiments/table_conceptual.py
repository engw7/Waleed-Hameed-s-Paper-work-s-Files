"""Conceptual comparison table (Reviewer 1): design facts (what each controller needs: sensors, training - properties of the implementations in mppt/controllers.py)
next to MEASURED columns read from the evaluation CSV (shading efficiency, GMPP success, mismatch robustness, compute per cycle).
'Hybrid P&O' from the literature is NOT implemented in this study, so it has no measured entries; Scan+P&O is our own scan/local-P&O hybrid.
No literature numbers are entered here: the literature comparison table (Table_Literature) is filled only from papers that were actually read."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, pandas as pd

DESIGN = {   # controller -> (sensors, training need, global search)   [implementation facts]
    'P&O': ('V, I', 'none', 'no (local)'),
    'IncCond': ('V, I', 'none', 'no (local)'),
    'Hybrid P&O (literature)': ('not implemented here', 'not implemented here', 'not implemented here'),
    'ANN-MPPT': ('G, T, V, I', 'offline, simulated/measured (G,T,V,I) data', 'no (maps the inputs to one V_ref)'),
    'Scan+P&O (no ML)': ('V, I, I-V scan (14 probe voltages), G/T not needed', 'none', 'yes (scan + argmax, then local P&O)'),
    'DNN scan-only (no P&O)': ('V, I scan, G (mean irradiance), T', 'offline, domain-randomised plant simulations', 'yes (DNN picks V_gmpp from the scan)'),
    'Hybrid (DNN+P&O)': ('V, I scan, G (mean irradiance), T', 'offline, domain-randomised plant simulations', 'yes (scan, DNN hand-off, local P&O)'),
}
ORDER = list(DESIGN)


def build(m, std4=('comprehensive', 'step_change', 'cloud_passing', 'ramp_change')):
    """m: DataFrame with columns plant, scenario, controller, tracking_efficiency_%, gmpp_success_%, compute_us_per_cycle (metrics table of the evaluation)."""
    from mppt.tables import save_table
    g = lambda plant, sc, col, c: (lambda x: float(x.mean()) if len(x) else np.nan)(m[(m.plant == plant) & m.scenario.isin(sc) & (m.controller == c)][col])
    rows = []
    for c in ORDER:
        sens, train, glob = DESIGN[c]; k = c if c in set(m.controller) else None
        meas = (lambda plant, sc, col: g(plant, sc, col, k)) if k else (lambda *a: np.nan)
        rows.append({'Controller': c, 'Sensors / inputs': sens, 'Training need': train, 'Global search (partial shading)': glob,
                     'Shading eff. nominal (%)': meas('nominal', ['partial_shading'], 'tracking_efficiency_%'), 'Shading eff. mismatched (%)': meas('mismatched', ['partial_shading'], 'tracking_efficiency_%'),
                     'GMPP success nominal (%)': meas('nominal', ['partial_shading'], 'gmpp_success_%'),
                     'Std. scenarios eff. nominal (%)': meas('nominal', list(std4), 'tracking_efficiency_%'), 'Std. scenarios eff. mismatched (%)': meas('mismatched', list(std4), 'tracking_efficiency_%'),
                     'Compute per cycle (us, measured)': meas('nominal', list(std4) + ['partial_shading'], 'compute_us_per_cycle')})
    df = pd.DataFrame(rows)
    save_table('Table_conceptual_comparison', df, 'Conceptual comparison of the controllers. Design columns are properties of the implementations; numeric columns are measured in this study (nominal = model-matched plant, mismatched = perturbed plant). Hybrid P&O from the literature was not implemented.', digits=2)
    return df


if __name__ == '__main__':      # smoke test on the frozen v3 numbers (column names mapped), written to results_quick via MPPT_QUICK=1
    from mppt.config import ROOT
    m = pd.read_csv(os.path.join(ROOT, 'reference', 'v3', 'metrics_by_scenario.csv')).rename(columns={'algorithm': 'controller', 'shading_success_%': 'gmpp_success_%'})
    print(build(m).round(2).to_string())
