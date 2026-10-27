"""Experiment configuration: inputs only, never results. QUICK mode (env MPPT_QUICK=1) shrinks data/epochs/scenarios for a ~5-minute smoke test."""
import os
from . import ROOT
QUICK = os.environ.get('MPPT_QUICK', '0') == '1'
RESULTS = os.path.join(ROOT, 'results' if not QUICK else os.path.join('results_quick'))
CACHE = os.path.join(ROOT, 'cache' if not QUICK else 'cache_quick')
for _d in (RESULTS, CACHE, os.path.join(RESULTS, 'tables'), os.path.join(RESULTS, 'figures', 'data'), *[os.path.join(RESULTS, s) for s in ('ablation', 'baselines', 'converter', 'quantisation', 'noise', 'hpc', 'models')]):
    os.makedirs(_d, exist_ok=True)

CFG = {
    'quick': QUICK, 'data_file': os.path.join(ROOT, 'PVGIS_Ramadi_2005-2020.csv'), 'seed': 42, 'g_min': 50.0,
    'v_start': 26.0, 'v_min_op': 8.0, 'v_max_op': 35.0, 'step_size': 0.5, 'ref_slew': 0.5, 'v_bus': 48.0,
    'n_v_per_row': 2, 'chrono': dict(train_end=2014, val_end=2016, calib_end=2018),
    'hot_split': dict(train_below_T=30.0, test_above_T=38.0), 'irr_split': dict(train_below_G=600.0, test_above_G=850.0),
    'nn_hidden': (64, 64, 32), 'nn_epochs': 3 if QUICK else 150, 'nn_batch': 512, 'nn_patience': 15,
    'conformal_alpha': [0.10, 0.05], 'mismatch_default': 1.0, 'mismatch_sweep': [0.0, 1.0] if QUICK else [0.0, 0.5, 1.0, 1.5, 2.0],
    'shade_mismatch_sweep': [0.0, 2.0] if QUICK else [0.0, 1.0, 2.0],
    'startup_v0_list': [15.0, 20.0, 26.0, 30.0, 35.0], 'startup_voc_fractions': [0.45, 0.60, 0.75, 0.90, 0.97], 'startup_cond': (800.0, 25.0), 'startup_steps': 300,
    # hybrid
    'scan_period': 300, 'scan_trigger': 0.10, 'scan_min_interval': 30,
    'n_workers': max(1, min(4, os.cpu_count() or 1)),
    'hybrid_n_train': 8000 if QUICK else 200000, 'hybrid_n_val': 2000 if QUICK else 15000, 'hybrid_n_calib': 2000 if QUICK else 15000, 'hybrid_n_test': 1000 if QUICK else 8000,
    'hybrid_hidden': (128, 128, 64), 'hybrid_epochs': 3 if QUICK else 60, 'hybrid_patience': 8, 'hybrid_batch': 1024,
    'quick_steps': 1900,
    # round-4 experiments
    'dnn_seeds': [0, 1] if QUICK else [0, 1, 2, 3, 4],
    'scenario_seeds': [0, 1] if QUICK else list(range(10)),
    'noise': dict(sigma_i=0.005, sigma_v=0.005, sigma_g=0.02),
}
