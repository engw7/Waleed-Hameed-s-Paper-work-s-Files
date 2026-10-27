"""Consistency tests: architecture numbers, paper numbers equal CSV values, forbidden patterns absent, evidence paths exist, models agree with their emulations."""
import os, sys, json, glob
import numpy as np, pandas as pd, pytest
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT)
from mppt.config import RESULTS, CFG

FORBIDDEN = ['mock', 'np.random.uniform(0.7', 'ensure_complex', 'force significance']


def _keras_model(n_in=16, hidden=(128, 128, 64)):
    from tensorflow.keras import layers, Sequential
    return Sequential([layers.Input((n_in,))] + [layers.Dense(h, activation='relu') for h in hidden] + [layers.Dense(1)])


def test_parameter_count_matches_keras_and_results():
    m = _keras_model(); assert m.count_params() == 27009
    p = os.path.join(RESULTS, 'hpc', 'latency.csv')
    if os.path.exists(p): assert int(pd.read_csv(p).params.iloc[0]) == 27009
    from mppt.nn import NumpyMLP
    assert 2 + 14 == 16 and NumpyMLP(m.get_weights(), np.zeros(16), np.ones(16), 0.0, 1.0).n_params() == 27009


def test_numpy_forward_equals_keras():
    from mppt.nn import NumpyMLP
    m = _keras_model(); mlp = NumpyMLP(m.get_weights(), np.zeros(16), np.ones(16), 0.0, 1.0); X = np.random.RandomState(0).normal(size=(64, 16)).astype('float32')
    assert np.max(np.abs(m(X, training=False).numpy().ravel() - mlp.predict(X))) < 1e-4


def test_int8_and_fp16_emulation_are_close_to_float32():
    from mppt.nn import NumpyMLP, QuantMLP
    m = _keras_model(); X = np.random.RandomState(1).normal(size=(500, 16)); mlp = NumpyMLP(m.get_weights(), np.zeros(16), np.ones(16), 0.0, 1.0); ref = mlp.predict(X)
    assert np.max(np.abs(QuantMLP(mlp, 'fp16').predict(X) - ref)) < 0.02 * (np.abs(ref).max() + 1e-6) + 1e-3
    assert np.max(np.abs(QuantMLP(mlp, 'int8', X).predict(X) - ref)) < 0.15 * (np.abs(ref).max() + 1e-6) + 1e-2


def test_converter_averaged_matches_switching_in_steady_state():
    from mppt.converter import AveragedBoostPlant, SwitchingBoostPlant
    from pv_single_diode import SingleDiodePV
    pvm = SingleDiodePV(); a, s = AveragedBoostPlant(pvm, v0=26.0), SwitchingBoostPlant(pvm, v0=26.0)
    for vc in (26.0, 25.0):
        va, ia, pa = a.calculate_pv_characteristics(800, 25, vc); vs, i_s, ps = s.calculate_pv_characteristics(800, 25, vc)
        assert abs(va - vc) < 0.05 and abs(pa - ps) / ps < 0.005


def test_paper_numbers_equal_csv_values():
    p = os.path.join(RESULTS, 'PAPER_NUMBERS.json')
    if not os.path.exists(p): pytest.skip('PAPER_NUMBERS.json not generated yet')
    ent = json.load(open(p)); assert len(ent) > 20
    for e in ent:
        df = pd.read_csv(os.path.join(ROOT, e['source_csv'])); q = df
        for k, v in e['filter'].items(): q = q[q[k] == v]
        assert len(q) >= 1, e['key']
        if e['value'] is None: continue
        assert np.isclose(float(q[e['column']].iloc[0]), float(e['value']), rtol=1e-9, atol=1e-12), e['key']


def test_no_forbidden_patterns_in_code():
    files = [f for pat in ('*.py', 'mppt/*.py', 'experiments/*.py', 'tools/*.py') for f in glob.glob(os.path.join(ROOT, pat))]
    for f in files:
        if os.path.basename(f) == 'test_consistency.py': continue
        txt = open(f).read().lower()
        for pat in FORBIDDEN: assert pat.lower() not in txt, (f, pat)
    nb = json.load(open(os.path.join(ROOT, 'archive', '20236359-code-new.ipynb')))
    for c in nb['cells']:
        s = ''.join(c['source'])
        if c['cell_type'] == 'code' and 'forb =' not in s:
            for pat in FORBIDDEN: assert pat.lower() not in s.lower(), pat


def test_every_figure_has_pdf_png_and_data():
    d = os.path.join(RESULTS, 'figures'); figs = sorted(set(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(d, 'Fig*.pdf'))))
    if not figs: pytest.skip('no figures yet')
    for f in figs:
        assert os.path.exists(os.path.join(d, f + '.png')) and os.path.exists(os.path.join(d, 'data', f + '.csv')), f


def test_figure_map_lists_every_figure():
    sys.path.insert(0, os.path.join(ROOT, 'tools')); import build_docs as B
    d = os.path.join(RESULTS, 'figures'); figs = set(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(d, 'Fig*.pdf')))
    assert figs <= set(B.FIG), sorted(figs - set(B.FIG))


