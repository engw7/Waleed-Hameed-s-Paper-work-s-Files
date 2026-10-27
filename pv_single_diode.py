"""Five-parameter single-diode PV module model with series/shunt resistance (explicit Lambert-W solution).

Equations (standard PV literature):
  [DS06] W. De Soto, S.A. Klein, W.A. Beckman, "Improvement and validation of a model for photovoltaic array
         performance", Solar Energy 80 (2006) 78-88  -- Eq. 1-5 (single-diode), parameter extraction from datasheet.
  [DB13] Duffie & Beckman, "Solar Engineering of Thermal Processes", 4th ed. (2013), ch. 23.
  [JA00] Jain & Kapoor (2004) / Jain, Kapoor: explicit Lambert-W solution of the single-diode equation.

  I = Iph - I0 [exp((V + I Rs)/a) - 1] - (V + I Rs)/Rsh,         a = n Ns k Tc / q  (modified ideality factor)
  Iph = (G/Gref) [Iph_ref + muIsc (Tc - Tref)]                                           [DS06 Eq. 2]
  I0  = I0_ref (Tc/Tref)^3 exp[(Eg q /(n_ref k)) (1/Tref - 1/Tc)]  (Eg = 1.121 eV Si)    [DS06 Eq. 3]
  Rsh = Rsh_ref (Gref/G)      Rs = Rs_ref   a = a_ref Tc/Tref                            [DS06 Eq. 4-5]
  Explicit current:  I = (Rsh (Iph+I0) - V)/(Rs+Rsh) - (a/Rs) W( Rs Rsh I0 /(a (Rs+Rsh))
                         * exp( Rsh (Rs (Iph+I0) + V) / (a (Rs+Rsh)) ) )                [JA00]
Voc depends on G logarithmically (Voc ~ a ln(Iph/I0)); this is what gives V_mpp its weak irradiance dependence.
Cell temperature = ambient temperature + (NOCT-20)/800 * G  (standard NOCT model, [DB13] eq. 23.x).
"""
import numpy as np
from scipy.special import lambertw

K_B, Q_E, T_REF, G_REF, EG = 1.380649e-23, 1.602176634e-19, 298.15, 1000.0, 1.121 * 1.602176634e-19


def w_exp(x):
    """Lambert W of exp(x), stable for large x (asymptotic start + Newton on w + ln w = x)."""
    x = np.asarray(x, float)
    small = x <= 30.0
    w = np.real(lambertw(np.exp(np.minimum(x, 30.0))))
    big = np.where(small, 40.0, x)
    wb = big - np.log(big)
    for _ in range(12):
        wb = wb - (wb + np.log(wb) - big) / (1.0 + 1.0 / wb)
    return np.where(small, w, wb)


class SingleDiodePV:
    # Datasheet (STC). Isc and Voc are the values the old Cell-4 model actually used. The old nameplate Impp = 11.4 A
    # exceeded Isc = 9.87 A (impossible), so Pmax = 244.6 W (what the old model really delivered at STC, FF ~ 0.75) is used.
    DATASHEET = dict(Isc=9.87, Voc=32.9, Pmax=244.6, mu_isc=0.0006 * 9.87, Ns=54, NOCT=45.0)

    def __init__(self, ds=None):
        self.ds = dict(self.DATASHEET if ds is None else ds)
        self.Ns = self.ds['Ns']
        self._extract()

    # ---- parameter extraction (Villalva-style, [DS06] Sec. 3): n and Rsh_ref are fixed at typical mono-Si values; for a
    #      trial Rs, Iph and I0 follow in closed form from the Isc and Voc conditions; Rs is then found (Brent) so that the
    #      model's TRUE maximum power equals the datasheet Pmax. Vmpp, Impp and the Voc temperature coefficient are OUTPUTS,
    #      reported and compared with the nameplate rather than forced (the old nameplate Vmpp*Impp > Voc*Isc*FF was
    #      infeasible and its -0.31 %/K Voc coefficient needs n < 0.7).
    N_DIODE, RSH_REF = 1.2, 400.0

    def _extract(self):
        from scipy.optimize import brentq
        d, Ns, n, Rsh = self.ds, self.Ns, self.N_DIODE, self.RSH_REF
        a = n * Ns * K_B * T_REF / Q_E

        def pars(Rs):
            I0 = (d['Isc'] * (1 + Rs / Rsh) - d['Voc'] / Rsh) / (np.expm1(d['Voc'] / a) - np.expm1(d['Isc'] * Rs / a))
            Iph = I0 * np.expm1(d['Voc'] / a) + d['Voc'] / Rsh
            return Iph, I0

        def pmax(Rs):
            Iph, I0 = pars(Rs)
            V = np.linspace(0.5, d['Voc'], 4000)
            arg = np.log(Rs * Rsh * I0 / (a * (Rs + Rsh))) + Rsh * (Rs * (Iph + I0) + V) / (a * (Rs + Rsh))
            I = (Rsh * (Iph + I0) - V) / (Rs + Rsh) - a / Rs * np.real(lambertw(np.exp(arg)))
            return float(np.max(V * I))

        target = d['Pmax']
        Rs = brentq(lambda r: pmax(r) - target, 1e-4, 1.5, xtol=1e-12)
        Iph, I0 = pars(Rs)
        self.p = dict(Iph_ref=Iph, I0_ref=I0, Rs=Rs, Rsh_ref=Rsh, n=n)

    def perturbed(self, severity):
        """Plant whose parameters differ from the nominal ones the controllers were trained on (aging / soiling / unknown
        module): Iph x(1-0.08 s), Rs x(1+0.8 s), Rsh x(1-0.4 s), n x(1+0.05 s). severity=0 returns the nominal plant."""
        import copy
        m = copy.copy(self)
        m.p = dict(self.p)
        m.p['Iph_ref'] *= 1 - 0.08 * severity
        m.p['Rs'] *= 1 + 0.8 * severity
        m.p['Rsh_ref'] *= 1 - 0.4 * severity
        m.p['n'] *= 1 + 0.05 * severity
        return m

    # ---- operating conditions
    def cell_temperature(self, G, T_amb):
        return np.asarray(T_amb, float) + (self.ds['NOCT'] - 20.0) / 800.0 * np.asarray(G, float)

    def _params(self, G, Tc):
        G, Tc = np.broadcast_arrays(np.asarray(G, float), np.asarray(Tc, float))
        Tk, p, d = Tc + 273.15, self.p, self.ds
        Gs = np.maximum(G, 1e-6)
        Iph = (Gs / G_REF) * (p['Iph_ref'] + d['mu_isc'] * (Tk - T_REF))
        I0 = p['I0_ref'] * (Tk / T_REF) ** 3 * np.exp(EG / (p['n'] * K_B) * (1 / T_REF - 1 / Tk))
        a = p['n'] * self.Ns * K_B * Tk / Q_E
        Rsh = p['Rsh_ref'] * G_REF / Gs
        return Iph, I0, a, Rsh

    def current(self, G, Tc, V):
        """Terminal current (A) at voltage V; vectorised; clipped at >= 0 (no reverse conduction modelled)."""
        Iph, I0, a, Rsh = self._params(G, Tc)
        Rs = self.p['Rs']
        V = np.asarray(V, float)
        arg = np.log(Rs * Rsh * I0 / (a * (Rs + Rsh))) + Rsh * (Rs * (Iph + I0) + V) / (a * (Rs + Rsh))
        w = w_exp(arg)
        I = (Rsh * (Iph + I0) - V) / (Rs + Rsh) - a / Rs * w
        return np.maximum(I, 0.0)

    def voc(self, G, Tc):
        Iph, I0, a, Rsh = self._params(G, Tc)
        lo, hi = np.zeros_like(Iph), np.full_like(Iph, 80.0)
        for _ in range(60):                                  # bisection on I(V)=0
            mid = 0.5 * (lo + hi)
            pos = self.current(G, Tc, mid) > 1e-9
            lo, hi = np.where(pos, mid, lo), np.where(pos, hi, mid)
        return 0.5 * (lo + hi)

    def mpp(self, G, T_amb, v_window=None, n_grid=3001):
        """True (global) MPP for scalar G, T_amb (ambient). Returns (V_mpp, P_mpp, I_mpp). Grid + golden refinement."""
        Tc = float(self.cell_temperature(G, T_amb))
        voc = float(self.voc(G, Tc))
        lo, hi = (0.05, voc) if v_window is None else (max(0.05, v_window[0]), min(voc, v_window[1]))
        V = np.linspace(lo, hi, n_grid)
        P = V * self.current(G, Tc, V)
        i = int(np.argmax(P))
        a, b = V[max(i - 1, 0)], V[min(i + 1, n_grid - 1)]
        for _ in range(60):                                  # golden-section refinement on the unimodal bracket
            c, d_ = b - 0.618034 * (b - a), a + 0.618034 * (b - a)
            if c * self.current(G, Tc, c) > d_ * self.current(G, Tc, d_):
                b = d_
            else:
                a = c
        v = 0.5 * (a + b)
        return v, v * float(self.current(G, Tc, v)), float(self.current(G, Tc, v))


def mpp_batch(model, G, T_amb, n_grid=1201, chunk=1500):
    """Vectorised true MPP for arrays of (G, ambient T). Returns V_mpp, I_mpp, P_mpp, T_cell (zeros where G < 1 W/m2).
    Grid search on [0, Voc] followed by a parabolic refinement of the maximum (agrees with MPP golden search to <1e-4 V)."""
    G, T_amb = np.asarray(G, float), np.asarray(T_amb, float)
    Vm, Im, Pm = np.zeros_like(G), np.zeros_like(G), np.zeros_like(G)
    Tc_all = model.cell_temperature(G, T_amb)
    live = np.where(G >= 1.0)[0]
    for s in range(0, len(live), chunk):
        idx = live[s:s + chunk]
        g, tc = G[idx][:, None], Tc_all[idx][:, None]
        voc = model.voc(G[idx], Tc_all[idx])[:, None]
        V = voc * np.linspace(0.0, 1.0, n_grid)[None, :]
        I = model.current(g, tc, V)
        P = V * I
        k = np.clip(P.argmax(axis=1), 1, n_grid - 2)
        r = np.arange(len(idx))
        y0, y1, y2 = P[r, k - 1], P[r, k], P[r, k + 1]
        den = y0 - 2 * y1 + y2
        off = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / den, 0.0)
        dv = (V[r, k + 1] - V[r, k - 1]) / 2
        v = V[r, k] + np.clip(off, -1, 1) * dv
        i = model.current(G[idx], Tc_all[idx], v)
        Vm[idx], Im[idx], Pm[idx] = v, i, v * i
    return Vm, Im, Pm, Tc_all


class ShadedString:
    """Module with N_SUB bypass-diode substrings (Ns/N_SUB cells each), each with its own irradiance.
    Substring: same single-diode law with Ns/N_SUB cells (a/N, Rs/N, Rsh/N); explicit INVERSE (Jain & Kapoor):
      V_k(I) = Rsh (Iph+I0) - I (Rs+Rsh) - a W( I0 Rsh / a * exp( Rsh (Iph + I0 - I) / a ) ),   V_k >= -0.7 V (bypass diode).
    String voltage = sum_k V_k(I). With equal irradiance it equals SingleDiodePV (verified in the notebook)."""
    N_SUB, V_BYPASS = 3, 0.7

    def __init__(self, model, v_window):
        self.m, self.v_window = model, v_window

    def _pre(self, G_sub, T_amb):
        """Per-substring parameters (computed once per operating condition)."""
        m, N = self.m, self.N_SUB
        Tc = m.cell_temperature(G_sub, T_amb)
        Iph, I0, a, Rsh = m._params(G_sub, Tc)
        return Iph, I0, a / N, Rsh / N, m.p['Rs'] / N

    def _v_from_pre(self, I, pre):
        Iph, I0, a, Rsh, Rs = pre
        I = np.asarray(I, float)[:, None]
        arg = np.log(I0 * Rsh / a) + Rsh * (Iph + I0 - I) / a
        V = Rsh * (Iph + I0) - I * (Rs + Rsh) - a * w_exp(arg)
        return np.maximum(V, -self.V_BYPASS).sum(axis=1)

    def string_voltage(self, I, G_sub, T_amb):
        return self._v_from_pre(I, self._pre(np.asarray(G_sub, float), T_amb))

    def i_at_v(self, G_sub, T_amb, V):
        """Current at terminal voltage V: coarse grid bracket, then bisection on the bracket (V(I) is monotone decreasing)."""
        pre = self._pre(np.asarray(G_sub, float), T_amb)
        hi = float(pre[0].max()) * 1.02
        grid = np.linspace(0.0, hi, 129)
        Vg = self._v_from_pre(grid, pre)
        if V >= Vg[0]:
            return 0.0
        j = int(np.searchsorted(-Vg, -V))                       # first index with Vg[j] <= V
        j = min(max(j, 1), len(grid) - 1)
        lo, hi = grid[j - 1], grid[j]
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if self._v_from_pre([mid], pre)[0] > V else (lo, mid)
        return 0.5 * (lo + hi)

    def mpp(self, G_sub, T_amb):
        """Global MPP inside the converter window."""
        G_sub = np.asarray(G_sub, float)
        pre = self._pre(G_sub, T_amb)
        I = np.linspace(0.0, float(pre[0].max()) * 1.02, 20001)
        V = self._v_from_pre(I, pre)
        P = np.where((V >= self.v_window[0]) & (V <= self.v_window[1]), V * I, -np.inf)
        k = int(np.argmax(P))
        return V[k], P[k], I[k]
