"""DC-DC boost converter between the PV source and a constant-voltage bus: averaged continuous-conduction (CCM) model and a switching-level (PWM) model.

Topology: PV (source i_pv(v_pv)) -> input capacitor C_in -> inductor L (series resistance R_L) -> low-side switch (R_on) / diode (V_f) -> bus V_bus (battery or regulated DC link).
Averaged model (state: v_pv, i_L; control: duty d):
    C_in dv_pv/dt = i_pv(v_pv) - i_L
    L    di_L/dt  = v_pv - i_L (R_L + d R_on) - (1 - d) (V_bus + V_f)
Inner loop: PI voltage controller on v_pv with feed-forward  d = d_ff + Kp e + Ki int(e),  e = v_pv - V_cmd,  d_ff = 1 - v_pv/(V_bus + V_f), d limited to [D_MIN, D_MAX].
The MPPT algorithm only chooses V_cmd; the converter plant interface is the same as the quasi-static plants:  calculate_pv_characteristics(G, T, V_cmd) -> (V, I, P)
where V, I, P are the means over the last 25 % of a W_MS settling window (several inductor/capacitor time constants) of the instantaneous v_pv, i_pv, v_pv*i_pv.
The averaged model is valid in CCM (i_L above half the ripple); the switching model includes DCM (diode blocks at i_L = 0)."""
import numpy as np

P = dict(L=1.0e-3, R_L=0.02, C_in=470e-6, f_sw=20e3, V_bus=48.0, V_f=0.5, R_on=0.015, Kp=0.05, Ki=10.0, D_MIN=0.05, D_MAX=0.95, W_MS=20.0, DT_AVG=20e-6, STEPS_PER_PERIOD=100)
PARAM_NOTES = {
    'L': 'chosen so that the ripple at the nominal operating point (V_pv = 26 V, d = 0.46, f_sw = 20 kHz) is dI = V_pv d / (f_sw L) = 0.6 A (about 7 % of 9 A); CCM holds above 0.3 A',
    'C_in': 'input ripple dV = dI / (8 f_sw C_in) = 0.08 V; also fixes the input-capacitor time constant of the voltage loop',
    'f_sw': 'typical for a 250 W module-level converter', 'V_bus': 'constant 48 V bus (battery or regulated DC link), the same value used for the display-only duty of the quasi-static results',
    'V_f': 'diode forward drop', 'R_on': 'switch on-resistance', 'R_L': 'inductor series resistance',
    'Kp': 'PI proportional gain [1/V], fixed once on the step response at G = 800 W/m2, T = 25 C (tools in this module), before any scenario was simulated',
    'Ki': 'PI integral gain [1/(V s)]', 'W_MS': 'settling window per measurement [ms] (3x for command jumps larger than 2 V); one MPPT control cycle (1 s) holds at least one window; a 14-point scan needs at most 14 x 60 ms = 0.84 s',
    'DT_AVG': 'integration step of the averaged model [s]', 'STEPS_PER_PERIOD': 'switching-level steps per PWM period'}


class IVTable:
    """Uniform-grid I(V) table of the PV source for the current operating condition."""
    N = 801

    def __init__(self, v_max, i_of_v):
        self.dv = v_max / (self.N - 1); self.I = np.maximum(np.asarray(i_of_v(np.linspace(0, v_max, self.N)), float), 0.0); self.vmax = v_max

    def i(self, v):
        if v <= 0.0: return float(self.I[0])
        x = v / self.dv; k = int(x)
        if k >= self.N - 1: return 0.0
        return float(self.I[k] + (x - k) * (self.I[k + 1] - self.I[k]))


def _module_table(pvm, G, T):
    Tc = float(pvm.cell_temperature(G, T)); v_max = float(pvm.voc(G, Tc)) * 1.02
    return IVTable(v_max, lambda v: pvm.current(G, Tc, v))


def _shaded_table(sh, G_sub, T):
    pre = sh._pre(np.asarray(G_sub, float), T); v_max = 36.0
    return IVTable(v_max, lambda v: np.array([sh.i_at_v(G_sub, T, x) for x in v]))


class _Base:
    def __init__(self, model, shaded=False, v0=26.0, params=None):
        self.p = dict(P if params is None else params); self.model, self.shaded = model, shaded
        self.G_sub = None; self._key = None; self.table = None
        self.v, self.iL, self.integ = float(v0), 0.0, 0.0; self.calls = []; self.outs = []; self.last = dict(D=np.nan, I_L=np.nan)
        self._init_pending = True

    def set_shading(self, g): self.G_sub = np.asarray(g, float)

    def _ensure_table(self, G, T):
        key = (tuple(self.G_sub.round(3)), round(float(T), 3)) if self.shaded else (round(float(G), 4), round(float(T), 4))
        if key != self._key:
            self.table = _shaded_table(self.model, self.G_sub, T) if self.shaded else _module_table(self.model, G, T); self._key = key
        if self._init_pending:                                        # start at the steady state of the first command: i_L from the power balance is unknown, use i_pv
            self.iL = self.table.i(self.v); self._init_pending = False

    def _window_ms(self, V_cmd):
        return self.p['W_MS'] * (3 if abs(V_cmd - self.v) > 2.0 else 1)          # large command jumps (scan probes, cold start) get a 3x longer settling window

    def state(self): return (self.v, self.iL, self.integ)

    def set_state(self, s): self.v, self.iL, self.integ = s; self._init_pending = False


class AveragedBoostPlant(_Base):
    """Averaged CCM model, explicit integration with step DT_AVG (semi-implicit for the capacitor voltage)."""
    def extras(self): return dict(D_conv=self.last['D'], I_L=self.last['I_L'])

    def calculate_pv_characteristics(self, G, T, V_cmd, trace=None):
        self._ensure_table(G, T); p = self.p; tb = self.table
        n = int(round(self._window_ms(V_cmd) * 1e-3 / p['DT_AVG'])); n_meas = int(round(p['W_MS'] * 1e-3 / p['DT_AVG'])) // 4; dt = p['DT_AVG']
        L, RL, C, Vb, Vf, Ron, Kp, Ki, dmin, dmax = p['L'], p['R_L'], p['C_in'], p['V_bus'], p['V_f'], p['R_on'], p['Kp'], p['Ki'], p['D_MIN'], p['D_MAX']
        v, iL, integ = self.v, self.iL, self.integ; sv = si = sp = sd = sil = 0.0; i_of = tb.i
        self.calls.append((G, T, None if self.G_sub is None else self.G_sub.copy(), V_cmd, (v, iL, integ)))
        for k in range(n):
            e = v - V_cmd; integ += Ki * e * dt
            lim = dmax - dmin; integ = -0.15 if integ < -0.15 else (0.15 if integ > 0.15 else integ)
            d = 1.0 - v / (Vb + Vf) + Kp * e + integ; d = dmin if d < dmin else (dmax if d > dmax else d)
            ipv = i_of(v)
            iL += (v - iL * (RL + d * Ron) - (1.0 - d) * (Vb + Vf)) / L * dt
            if iL < 0.0: iL = 0.0
            v += (ipv - iL) / C * dt
            if v < 0.0: v = 0.0
            if k >= n - n_meas: sv += v; si += ipv; sp += v * ipv; sd += d; sil += iL
            if trace is not None: trace.append((k * dt, v, iL, ipv))
        self.v, self.iL, self.integ = v, iL, integ; m = float(n_meas)
        self.last = dict(D=sd / m, I_L=sil / m); self.outs.append((sv / m, si / m, sp / m))
        return sv / m, si / m, sp / m


class SwitchingBoostPlant(_Base):
    """Switching-level model: PWM with sawtooth carrier, duty updated once per period by the same PI law (sampled at the period start), diode with DCM clamp."""
    def extras(self): return dict(D_conv=self.last['D'], I_L=self.last['I_L'])

    def calculate_pv_characteristics(self, G, T, V_cmd, trace=None):
        self._ensure_table(G, T); p = self.p; tb = self.table
        self.calls.append((G, T, None if self.G_sub is None else self.G_sub.copy(), V_cmd, (self.v, self.iL, self.integ)))
        Tp = 1.0 / p['f_sw']; nps = p['STEPS_PER_PERIOD']; dt = Tp / nps; periods = int(round(self._window_ms(V_cmd) * 1e-3 / Tp)); meas_from = periods - int(round(p['W_MS'] * 1e-3 / Tp)) // 4
        L, RL, C, Vb, Vf, Ron, Kp, Ki, dmin, dmax = p['L'], p['R_L'], p['C_in'], p['V_bus'], p['V_f'], p['R_on'], p['Kp'], p['Ki'], p['D_MIN'], p['D_MAX']
        v, iL, integ = self.v, self.iL, self.integ; sv = si = sp = sd = sil = 0.0; cnt = 0; i_of = tb.i
        vmin = vmax = v; ilmin = ilmax = iL
        for j in range(periods):
            e = v - V_cmd; integ += Ki * e * Tp; lim = dmax - dmin; integ = -0.15 if integ < -0.15 else (0.15 if integ > 0.15 else integ)
            d = 1.0 - v / (Vb + Vf) + Kp * e + integ; d = dmin if d < dmin else (dmax if d > dmax else d); n_on = int(round(d * nps))
            for k in range(nps):
                ipv = i_of(v)
                if k < n_on: iL += (v - iL * (RL + Ron)) / L * dt
                else:
                    iL += (v - iL * RL - (Vb + Vf)) / L * dt
                    if iL < 0.0: iL = 0.0
                v += (ipv - iL) / C * dt
                if v < 0.0: v = 0.0
                if j >= meas_from:
                    sv += v; si += ipv; sp += v * ipv; sd += d; sil += iL; cnt += 1
                    if v < vmin: vmin = v
                    if v > vmax: vmax = v
                    if iL < ilmin: ilmin = iL
                    if iL > ilmax: ilmax = iL
                if trace is not None and k % 5 == 0: trace.append((len(trace) * dt * 5, v, iL, ipv))
        self.v, self.iL, self.integ = v, iL, integ; m = float(cnt)
        self.last = dict(D=sd / m, I_L=sil / m, ripple_v=vmax - vmin, ripple_i=ilmax - ilmin); self.outs.append((sv / m, si / m, sp / m, ilmax - ilmin, vmax - vmin))
        return sv / m, si / m, sp / m
