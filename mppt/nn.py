"""Numpy forward pass of the trained Keras MLPs (identical weights; verified against Keras in tests) and thin predictor wrappers.
Closed-loop runs use the numpy forward so that experiments are fast and process-parallel; framework-level latency is benchmarked separately."""
import numpy as np


class NumpyMLP:
    def __init__(self, weights, mean_x, scale_x, mean_y, scale_y, activation='relu'):
        self.W = [np.asarray(w, np.float32) for w in weights[0::2]]; self.b = [np.asarray(b, np.float32) for b in weights[1::2]]
        self.mx, self.sx, self.my, self.sy = np.asarray(mean_x, np.float64), np.asarray(scale_x, np.float64), float(mean_y), float(scale_y)
        self.act = activation

    @classmethod
    def from_keras(cls, model, scaler_x, scaler_y):
        return cls(model.get_weights(), scaler_x.mean_, scaler_x.scale_, scaler_y.mean_[0], scaler_y.scale_[0])

    def n_params(self):
        return int(sum(w.size + b.size for w, b in zip(self.W, self.b)))

    def macs(self):
        return int(sum(w.size for w in self.W))

    def predict_scaled(self, Xs):
        h = np.asarray(Xs, np.float32)
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            h = h @ w + b
            if i < len(self.W) - 1:
                h = np.maximum(h, 0.0)
        return h.ravel()

    def predict(self, X):
        return self.predict_scaled((np.asarray(X, float) - self.mx) / self.sx).astype(np.float64) * self.sy + self.my


class VrefPredictor:
    """(G,T,V,I) or (G,T) -> V_ref for the plain sensor-based controllers (ANN-MPPT, Poly2)."""
    def __init__(self, name, mlp, features):
        self.name, self.mlp, self.features = name, mlp, features

    def vref(self, G, T, V, I):
        d = {'G': G, 'T': T, 'V': V, 'I': I}
        return float(self.mlp.predict(np.array([[d[f] for f in self.features]], float))[0])

    def predict_batch(self, df):
        return self.mlp.predict(df[self.features].to_numpy(float))


class PolyPredictor:
    """Quadratic polynomial of (G,T) -> V_mpp (numpy re-implementation of StandardScaler + PolynomialFeatures(2) + Ridge, coefficients extracted from the fitted sklearn pipeline)."""
    features = ['G', 'T']

    def __init__(self, name, pipe):
        self.name = name
        sc, pf, rg = pipe.steps[0][1], pipe.steps[1][1], pipe.steps[2][1]
        self.mean, self.scale, self.coef, self.icpt = sc.mean_, sc.scale_, rg.coef_, float(rg.intercept_)
        assert pf.degree == 2 and pf.include_bias

    def _feat(self, X):
        z = (np.asarray(X, float) - self.mean) / self.scale
        return np.column_stack([np.ones(len(z)), z[:, 0], z[:, 1], z[:, 0] ** 2, z[:, 0] * z[:, 1], z[:, 1] ** 2])

    def predict_batch(self, df):
        return self._feat(df[self.features].to_numpy(float)) @ self.coef + self.icpt

    def vref(self, G, T, V, I):
        return float((self._feat(np.array([[G, T]])) @ self.coef)[0] + self.icpt)

    def macs(self):
        return 6


class ScanEstimator:
    """Stage-1 estimator of the scan-based controllers: (G_in, T, 14 scan currents) -> V_gmpp. `model` is a NumpyMLP, an sklearn regressor or any object with .predict(X)."""
    def __init__(self, name, model):
        self.name, self.model = name, model

    def vest(self, G_in, T, I_probe):
        x = np.concatenate([[G_in, T], I_probe])[None, :]
        return float(self.model.predict(x)[0])

    def predict_batch(self, X):
        return np.asarray(self.model.predict(X), float)


class QuantMLP:
    """Post-training quantised copy of a NumpyMLP (numpy emulation, so measured latency is that of the emulation, not of an MCU).
    mode 'fp16': weights/biases rounded to float16, arithmetic in float32.  mode 'int8': symmetric per-tensor int8 weights, static per-layer int8 activations
    (scales from calibration inputs), int32 accumulation, float rescale between layers (as in common integer-inference kernels)."""
    def __init__(self, mlp, mode, X_calib=None):
        assert mode in ('fp16', 'int8')
        self.mode, self.mx, self.sx, self.my, self.sy = mode, mlp.mx, mlp.sx, mlp.my, mlp.sy
        if mode == 'fp16':
            self.W = [w.astype(np.float16).astype(np.float32) for w in mlp.W]; self.b = [b.astype(np.float16).astype(np.float32) for b in mlp.b]
        else:
            assert X_calib is not None
            self.sw = [float(np.abs(w).max()) / 127.0 for w in mlp.W]
            self.Wq = [np.clip(np.round(w / s), -127, 127).astype(np.int32) for w, s in zip(mlp.W, self.sw)]
            self.b = [b.astype(np.float32) for b in mlp.b]
            h = ((np.asarray(X_calib, float) - mlp.mx) / mlp.sx).astype(np.float32); self.sa = []
            for i, (w, b) in enumerate(zip(mlp.W, mlp.b)):
                self.sa.append(float(np.abs(h).max()) / 127.0)
                h = h @ w + b
                if i < len(mlp.W) - 1: h = np.maximum(h, 0.0)
        self.n_layers = len(mlp.W)

    def n_bytes(self):
        return int(sum(w.size for w in (self.W if self.mode == 'fp16' else self.Wq)) * (2 if self.mode == 'fp16' else 1) + sum(b.size for b in self.b) * 4)

    def predict_scaled(self, Xs):
        h = np.asarray(Xs, np.float32)
        for i in range(self.n_layers):
            if self.mode == 'fp16':
                h = h @ self.W[i] + self.b[i]
            else:
                q = np.clip(np.round(h / self.sa[i]), -127, 127).astype(np.int32)
                h = (q @ self.Wq[i]).astype(np.float32) * np.float32(self.sa[i] * self.sw[i]) + self.b[i]
            if i < self.n_layers - 1: h = np.maximum(h, 0.0)
        return h.ravel()

    def predict(self, X):
        return self.predict_scaled((np.asarray(X, float) - self.mx) / self.sx).astype(np.float64) * self.sy + self.my
