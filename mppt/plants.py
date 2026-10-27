"""Quasi-static plants (module, bypass-diode string, sensor-noise wrapper) and ground-truth MPP series (used for scoring only)."""
import numpy as np
from .config import CFG
from pv_single_diode import ShadedString, mpp_batch


class ModulePlant:
    """Whole module (single-diode, uniform irradiance). `model` may be the nominal or a perturbed (mismatched) plant."""
    def __init__(self, model): self.m = model
    def calculate_pv_characteristics(self, G, T, V):
        I = float(self.m.current(G, self.m.cell_temperature(G, T), V)); return V, I, V * I


class ShadedPlant:
    """3-substring bypass-diode string; irradiance per substring is set by the simulator each cycle."""
    def __init__(self, s): self.s, self.G_sub = s, None
    def set_shading(self, g): self.G_sub = g
    def calculate_pv_characteristics(self, G, T, V):
        I = self.s.i_at_v(self.G_sub, T, V); return V, I, V * I


class SensorNoisePlant:
    """Wraps a plant: the controller receives noisy measurements (relative Gaussian noise on V, I and on the irradiance reading), the physics
    and the scoring use the true values. `last_true` accumulates the true (V, I, P) of every call of the current control cycle."""
    def __init__(self, plant, rng, sigma_v, sigma_i, sigma_g):
        self.plant, self.rng, self.sv, self.si, self.sg = plant, rng, sigma_v, sigma_i, sigma_g
        self.G_true = self.T_true = None; self.buf = []
    def set_shading(self, g):
        if hasattr(self.plant, 'set_shading'): self.plant.set_shading(g)
    def begin_cycle(self, G, T):
        self.G_true, self.T_true, self.buf = G, T, []
    def sense_G(self, G): return G * (1.0 + self.sg * self.rng.normal())
    def calculate_pv_characteristics(self, G_ctrl, T_ctrl, V):
        Vt, It, Pt = self.plant.calculate_pv_characteristics(self.G_true, self.T_true, V)
        self.buf.append((Vt, It, Pt))
        Vm, Im = Vt * (1 + self.sv * self.rng.normal()), It * (1 + self.si * self.rng.normal())
        return Vm, Im, Vm * Im
    def true_cycle(self):
        b = np.array(self.buf); return b[:, 0].mean(), b[:, 1].mean(), b[:, 2].mean()


def truth_module(scn, model):
    Vm, Im, Pm, _ = mpp_batch(model, scn['G'], scn['T'])
    assert Vm.min() >= CFG['v_min_op'] and Vm.max() <= CFG['v_max_op'], 'true MPP outside converter window'
    return np.column_stack([Vm, Pm])


def truth_shaded(scn, sh_model):
    cache, out = {}, []
    for g, t in zip(scn['G_sub'], scn['T']):
        k = (tuple(g), round(float(t), 6))
        if k not in cache:
            v, p, _ = sh_model.mpp(g, t); cache[k] = (v, p)
        out.append(cache[k])
    return np.array(out)


def make_plant_and_truth(scn, model):
    """Plant + truth for a scenario and a (nominal or perturbed) single-diode model."""
    win = (CFG['v_min_op'], CFG['v_max_op'])
    if 'G_sub' in scn:
        sh = ShadedString(model, win); return ShadedPlant(sh), truth_shaded(scn, sh)
    return ModulePlant(model), truth_module(scn, model)


def new_plant(scn, model):
    """Fresh plant object (stateful shading set-point) for a scenario/model; no truth computed."""
    win = (CFG['v_min_op'], CFG['v_max_op'])
    return ShadedPlant(ShadedString(model, win)) if 'G_sub' in scn else ModulePlant(model)
