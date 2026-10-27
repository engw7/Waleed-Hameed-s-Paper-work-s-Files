"""Training of every learned estimator: v2 networks (simple / complex / Poly2 on the S1-S3 protocols), the hybrid stage-1 DNN (parameterised by
architecture, regularisation, inputs and seed for the ablation), a ridge model, and a gradient-boosted-tree estimator.
Decisions (architecture, early stopping) use the train/validation years only; test years are touched only by evaluate_* functions."""
import os, time, hashlib, json
import numpy as np, pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential, layers, callbacks, optimizers, regularizers
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from .config import CFG, CACHE
from .nn import NumpyMLP, VrefPredictor, PolyPredictor, ScanEstimator
import pv_hybrid_data as hd

tf.get_logger().setLevel('ERROR')
tf.config.experimental.enable_op_determinism()      # reproducible weight init / training on CPU
FEATS = {'simple': ['G', 'T'], 'complex': ['G', 'T', 'V', 'I']}


def set_determinism(seed):
    tf.keras.utils.set_random_seed(seed)


# ------------------------------------------------------------------------------------------- v2 networks (uniform irradiance, single nominal plant)
def make_rows(day, pvm, n_v, rng):
    d = day.loc[day.index.repeat(n_v)].reset_index(drop=True)
    voc = pvm.voc(d['G(i)'].to_numpy(), d['Tcell'].to_numpy())
    hi = np.minimum(CFG['v_max_op'], 0.95 * voc)
    d['V'] = CFG['v_min_op'] + (hi - CFG['v_min_op']) * rng.uniform(size=len(d))
    d['I'] = pvm.current(d['G(i)'].to_numpy(), d['Tcell'].to_numpy(), d['V'].to_numpy())
    d = d.rename(columns={'G(i)': 'G', 'T2m': 'T', 'Vm': 'V_mpp', 'P': 'P_mpp'})
    return d[['t', 'year', 'month', 'G', 'T', 'Tcell', 'V', 'I', 'V_mpp', 'P_mpp']]


def v2_protocols(pv, pvm):
    day = pv[pv['G(i)'] >= CFG['g_min']].reset_index(drop=True)
    rng = np.random.RandomState(CFG['seed'])
    rows = make_rows(day.iloc[::10] if CFG['quick'] else day, pvm, CFG['n_v_per_row'], rng)
    c = CFG['chrono']
    S1 = dict(train=rows[rows.year <= c['train_end']], val=rows[(rows.year > c['train_end']) & (rows.year <= c['val_end'])],
              calib=rows[(rows.year > c['val_end']) & (rows.year <= c['calib_end'])], test=rows[rows.year > c['calib_end']])
    assert S1['train'].t.max() < S1['val'].t.min() < S1['calib'].t.min() < S1['test'].t.min()
    def split_range(col, lo, hi):
        tr, te = rows[rows[col] < lo], rows[rows[col] >= hi]
        hold = np.random.RandomState(CFG['seed'] + 7).uniform(size=len(tr)) < 0.15
        return dict(train=tr[~hold], val=tr[hold], calib=tr[hold], test=te)
    return {'S1': S1, 'S2': split_range('T', CFG['hot_split']['train_below_T'], CFG['hot_split']['test_above_T']),
            'S3': split_range('G', CFG['irr_split']['train_below_G'], CFG['irr_split']['test_above_G'])}, rows


class Trained:
    """Keras model + scalers + history; .mlp is the numpy forward used in closed loop."""
    def __init__(self, name, model, sx, sy, features, history, seconds):
        self.name, self.model, self.sx, self.sy, self.features, self.history, self.seconds = name, model, sx, sy, features, history, seconds
        self.mlp = NumpyMLP.from_keras(model, sx, sy); self.epochs = len(history['loss'])
        self.best_val = float(min(history['val_loss']))

    def dump(self, prefix):
        import pickle
        self.model.save(prefix + '.keras')
        with open(prefix + '.pkl', 'wb') as f: pickle.dump(dict(name=self.name, sx=self.sx, sy=self.sy, features=self.features, history=self.history, seconds=self.seconds), f)

    @classmethod
    def load(cls, prefix):
        import pickle
        d = pickle.load(open(prefix + '.pkl', 'rb')); model = tf.keras.models.load_model(prefix + '.keras')
        t = cls(d['name'], model, d['sx'], d['sy'], d['features'], d['history'], d['seconds']); return t

    def predict_keras(self, X):
        return self.model.predict(self.sx.transform(np.asarray(X, float)).astype('float32'), batch_size=8192, verbose=0).ravel() * self.sy.scale_[0] + self.sy.mean_[0]


def fit_mlp(name, X_tr, y_tr, X_va, y_va, hidden, seed, epochs, batch, patience, dropout=0.0, l2=0.0, lr_patience=6):
    set_determinism(seed)
    sx = StandardScaler().fit(X_tr); sy = StandardScaler().fit(np.asarray(y_tr, float)[:, None])
    f = lambda X: sx.transform(X).astype('float32'); g = lambda y: sy.transform(np.asarray(y, float)[:, None]).astype('float32')
    reg = regularizers.l2(l2) if l2 > 0 else None
    ls = [layers.Input((X_tr.shape[1],))]
    for h in hidden:
        ls.append(layers.Dense(h, activation='relu', kernel_regularizer=reg))
        if dropout > 0: ls.append(layers.Dropout(dropout))
    ls.append(layers.Dense(1))
    m = Sequential(ls); m.compile(optimizer=optimizers.Adam(1e-3), loss='mse')
    t0 = time.perf_counter()
    h = m.fit(f(X_tr), g(y_tr), validation_data=(f(X_va), g(y_va)), epochs=epochs, batch_size=batch, verbose=0,
              callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
                         callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=lr_patience, min_lr=1e-6)])
    return Trained(name, m, sx, sy, None, h.history, time.perf_counter() - t0)


def train_v2(pv, pvm, seed=None, protocols=None):
    """Simple (G,T), complex (G,T,V,I) and Poly2 on S1; complex + Poly2 on S2/S3. Same settings for every network."""
    seed = CFG['seed'] if seed is None else seed
    protocols, rows = protocols or v2_protocols(pv, pvm)
    out = {}
    for key, sp in protocols.items():
        out[key] = {}
        pf = make_pipeline(StandardScaler(), PolynomialFeatures(2), Ridge(alpha=1e-6)).fit(sp['train'][FEATS['simple']].to_numpy(float), sp['train'].V_mpp.to_numpy(float))
        out[key]['poly2'] = PolyPredictor(f'Poly2(G,T) {key}', pf)
        kinds = ['complex'] + (['simple'] if key == 'S1' else [])
        for kind in kinds:
            tr = fit_mlp(f'ANN {kind} {key}', sp['train'][FEATS[kind]].to_numpy(float), sp['train'].V_mpp.to_numpy(float), sp['val'][FEATS[kind]].to_numpy(float), sp['val'].V_mpp.to_numpy(float),
                         CFG['nn_hidden'], seed, CFG['nn_epochs'], CFG['nn_batch'], CFG['nn_patience'], lr_patience=10)
            tr.features = FEATS[kind]; out[key][kind] = tr
    return out, protocols


def v2_predictor(tr, name):
    return VrefPredictor(name, tr.mlp, tr.features)


def eval_v2(pred_fn, df, pvm):
    y, yh = df.V_mpp.to_numpy(), pred_fn(df)
    vref = np.clip(yh, CFG['v_min_op'], CFG['v_max_op'])
    I = pvm.current(df.G.to_numpy(), df.Tcell.to_numpy(), vref)
    return {'R2': r2_score(y, yh), 'RMSE_V': float(np.sqrt(mean_squared_error(y, yh))), 'MAE_V': mean_absolute_error(y, yh), 'max_abs_err_V': float(np.max(np.abs(y - yh))),
            'static_P_ratio_%': float(100 * np.mean(vref * I / df.P_mpp.to_numpy())), 'n': len(df)}, yh


# --------------------------------------------------------------------------------------------- hybrid stage-1 datasets and estimators
def hybrid_bases(pv):
    day = pv[pv['G(i)'] >= CFG['g_min']].reset_index(drop=True)
    c = CFG['chrono']; yr = day.year
    sel = {'train': yr <= c['train_end'], 'val': (yr > c['train_end']) & (yr <= c['val_end']), 'calib': (yr > c['val_end']) & (yr <= c['calib_end']), 'test': yr > c['calib_end']}
    return {k: (day.loc[m, 'G(i)'].to_numpy(), day.loc[m, 'T2m'].to_numpy()) for k, m in sel.items()}


def hybrid_features(d, scan=True):
    cols = [d['G_in'], d['T']] + ([d['I_probe']] if scan else [])
    return np.column_stack([d['G_in'], d['T'], d['I_probe']]) if scan else np.column_stack([d['G_in'], d['T']])


def hybrid_data(pv, use_cache=True):
    """Train/val/calibration sets and the four held-out test sets (T1-T4); cached on disk keyed by the data-generation config."""
    key = hashlib.md5(json.dumps([CFG['hybrid_n_train'], CFG['hybrid_n_val'], CFG['hybrid_n_calib'], CFG['hybrid_n_test'], CFG['quick'], hd.DR_BOX, hd.OOR_BOX], sort_keys=True, default=str).encode()).hexdigest()[:10]
    cp = os.path.join(CACHE, f'hybrid_data_{key}.npz')
    names = ['train', 'val', 'calib', 'T1', 'T2', 'T3', 'T4']
    if use_cache and os.path.exists(cp):
        z = np.load(cp, allow_pickle=True); return {n: {k.split('|', 1)[1]: z[k] for k in z.files if k.split('|', 1)[0] == n} for n in names}, json.loads(str(z['_cost']))
    b = hybrid_bases(pv); nw = CFG['n_workers']; cost = {}
    bn = 600 if CFG['quick'] else 12000
    t0 = time.perf_counter(); hd.generate(bn, *b['train'], 'dr', 'mix', 1, n_workers=1); ts = time.perf_counter() - t0
    t0 = time.perf_counter(); hd.generate(bn, *b['train'], 'dr', 'mix', 1, n_workers=nw); tp = time.perf_counter() - t0
    a_ = hd.generate(3000, *b['train'], 'dr', 'mix', 5, n_workers=1); b_ = hd.generate(3000, *b['train'], 'dr', 'mix', 5, n_workers=nw)
    t0 = time.perf_counter(); tr = hd.generate(CFG['hybrid_n_train'], *b['train'], 'dr', 'mix', 11, n_workers=nw); tg = time.perf_counter() - t0
    cost.update(cpu_cores_used=nw, benchmark_samples=bn, serial_seconds=ts, parallel_seconds=tp, measured_speedup=ts / tp, serial_ms_per_plant_sample=1e3 * ts / bn,
                parallel_identical_to_serial=all(np.array_equal(a_[k], b_[k]) for k in a_), train_samples=CFG['hybrid_n_train'], train_generation_wallclock_s=tg,
                train_generation_serial_equivalent_s=CFG['hybrid_n_train'] * ts / bn, distinct_plants_in_training_set=int(len(np.unique(np.round(tr['rs'], 12)))))
    sets = {'train': tr, 'val': hd.generate(CFG['hybrid_n_val'], *b['val'], 'dr', 'mix', 12, n_workers=nw),
            'calib': hd.generate(CFG['hybrid_n_calib'], *b['calib'], 'dr', 'mix', 13, n_workers=nw, store_pv=True),
            'T1': hd.generate(CFG['hybrid_n_test'] * 2, *b['test'], 'dr', 'mix', 21, n_workers=nw, store_pv=True),
            'T2': hd.generate(CFG['hybrid_n_test'], *b['test'], 'nominal', 'none', 22, n_workers=nw, store_pv=True),
            'T3': hd.generate(CFG['hybrid_n_test'], *b['test'], 'nominal', 'always', 23, n_workers=nw, store_pv=True),
            'T4': hd.generate(CFG['hybrid_n_test'], *b['test'], 'oor', 'mix', 24, n_workers=nw, store_pv=True)}
    np.savez_compressed(cp, _cost=json.dumps(cost), **{f'{n}|{k}': v for n, d in sets.items() for k, v in d.items()})
    return sets, cost


TEST_NAMES = {'T1': 'T1 in-distribution (DR plants, mixed shading)', 'T2': 'T2 nominal plant, uniform irradiance', 'T3': 'T3 nominal plant, partial shading', 'T4': 'T4 OUT-OF-RANGE plants, mixed shading'}


class HybridModel:
    """A trained stage-1 estimator (any family) with a common interface for the ablation."""
    def __init__(self, name, kind, predict, estimator, n_params, macs, seconds, epochs=None, history=None, trained=None, scan=True):
        self.name, self.kind, self.predict, self.estimator, self.n_params, self.macs, self.seconds, self.epochs, self.history, self.trained, self.scan = name, kind, predict, estimator, n_params, macs, seconds, epochs, history, trained, scan


def train_hybrid_dnn(data, hidden=None, seed=0, dropout=0.0, l2=0.0, scan=True, name=None):
    hidden = CFG['hybrid_hidden'] if hidden is None else hidden
    Xtr, Xva = hybrid_features(data['train'], scan), hybrid_features(data['val'], scan)
    tr = fit_mlp(name or f'DNN {hidden}', Xtr, data['train']['V_gmpp'], Xva, data['val']['V_gmpp'], hidden, seed, CFG['hybrid_epochs'], CFG['hybrid_batch'], CFG['hybrid_patience'], dropout, l2, lr_patience=4)
    mlp = tr.mlp
    if scan:
        pred = lambda X: mlp.predict(X)
        est = ScanEstimator(tr.name, mlp)
    else:                                                      # G,T-only network inside the scan structure: ignores the scan currents
        class _GT:
            def predict(self, X): return mlp.predict(np.asarray(X)[:, :2])
        pred = lambda X: mlp.predict(np.asarray(X)[:, :2]); est = ScanEstimator(tr.name, _GT())
    return HybridModel(tr.name, 'mlp', pred, est, mlp.n_params(), mlp.macs(), tr.seconds, tr.epochs, tr.history, tr, scan)


def train_ridge(data, scan=True, name='Ridge (linear)'):
    t0 = time.perf_counter(); sx = StandardScaler().fit(hybrid_features(data['train'], scan))
    r = Ridge(alpha=1.0).fit(sx.transform(hybrid_features(data['train'], scan)), data['train']['V_gmpp']); sec = time.perf_counter() - t0
    pred = lambda X: r.predict(sx.transform(np.asarray(X, float))[:, :(16 if scan else 2)])
    class _M:
        def predict(self, X): return pred(X)
    return HybridModel(name, 'ridge', pred, ScanEstimator(name, _M()), int(r.coef_.size + 1), int(r.coef_.size), sec)


def train_trees(data, seed=0, name='Gradient-boosted trees'):
    t0 = time.perf_counter()
    gb = HistGradientBoostingRegressor(max_iter=400 if not CFG['quick'] else 40, learning_rate=0.1, max_leaf_nodes=31, early_stopping=False, random_state=seed)
    Xtr, ytr = hybrid_features(data['train']), data['train']['V_gmpp']
    # early-stopping iteration chosen on the VALIDATION years only (staged predictions), then the model is refit once with that many iterations
    gb.fit(Xtr, ytr); Xva, yva = hybrid_features(data['val']), data['val']['V_gmpp']
    errs = [np.mean((p - yva) ** 2) for p in gb.staged_predict(Xva)]; best = int(np.argmin(errs)) + 1
    gb = HistGradientBoostingRegressor(max_iter=best, learning_rate=0.1, max_leaf_nodes=31, early_stopping=False, random_state=seed).fit(Xtr, ytr)
    sec = time.perf_counter() - t0
    n_nodes = int(sum(len(p.nodes) for p in gb._predictors for p in p))
    return HybridModel(name, 'trees', lambda X: gb.predict(np.asarray(X, float)), ScanEstimator(name, gb), n_nodes, int(best * 5), sec, epochs=best)


def eval_hybrid(model, data, pvm=None):
    """Offline accuracy on the held-out sets: RMSE/MAE of V_gmpp, wrong-peak rate (|err| > 2 V) and static power ratio P(V_est)/P_gmpp."""
    rows = []
    for k in ('T1', 'T2', 'T3', 'T4'):
        d = data[k]; yh = np.clip(model.predict(hybrid_features(d, True)), CFG['v_min_op'], CFG['v_max_op'])
        ratio = 100 * np.mean([np.interp(v, d['V_eval'][i], d['P_eval'][i]) / d['P_gmpp'][i] for i, v in enumerate(yh)])
        err = yh - d['V_gmpp']
        rows.append({'test_set': TEST_NAMES[k], 'RMSE_V': float(np.sqrt(np.mean(err ** 2))), 'MAE_V': float(np.mean(np.abs(err))), 'wrong_peak_%': float(100 * np.mean(np.abs(err) > 2.0)),
                     'P_ratio_%': float(ratio), 'R2': float(r2_score(d['V_gmpp'], yh)), 'n': len(yh)})
    return pd.DataFrame(rows)
