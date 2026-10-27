"""MPPT controllers. All controllers share one interface:  rec = ctrl.mppt_step(plant, G, T, time_step)  with
rec = dict(V, I, P (measured at the applied command), V_cmd (next command), step, compute_ns [, scan, V_ref]).
Controllers never see the ground-truth MPP. SENSOR USE: P&O / IncCond / PSO need only V and I; Poly2 / ANN additionally use G and T;
scan-based controllers use G, T and a fast I-V scan (energy cost of the scan is charged in the logs)."""
import time
import numpy as np
from .config import CFG
import pv_hybrid_data as hd


class MPPTAlgorithm:
    def __init__(self, name, v_start=None):
        self.algorithm_name = name
        self.V_operating = CFG['v_start'] if v_start is None else v_start     # = the COMMAND currently applied to the converter
        self.I_operating = self.P_operating = 0.0
        self.V_previous, self.I_previous, self.P_previous = self.V_operating, 0.0, 0.0
        self.max_voltage, self.min_voltage = CFG['v_max_op'], CFG['v_min_op']


class PerturbAndObserveMPPT(MPPTAlgorithm):
    """P&O, decision rules identical to the original Cell 4 (see docs/PARAMETERS.md and Fig. flowchart). The previous MEASURED sample is stored after the decision.
    The next command is derived from the applied command (not from the noisy measurement). legacy_bookkeeping=True reproduces the original defect (dV = 0 always)."""
    def __init__(self, step_size=None, v_start=None, legacy_bookkeeping=False):
        super().__init__('P&O', v_start)
        self.step_size = CFG['step_size'] if step_size is None else step_size
        self.perturbation_direction, self.legacy = 1, legacy_bookkeeping

    def mppt_step(self, plant, G, T, time_step):
        if self.legacy:
            self.V_previous, self.P_previous = self.V_operating, self.P_operating
        Vc = self.V_operating
        V_meas, I_meas, P_meas = plant.calculate_pv_characteristics(G, T, Vc)
        self.I_operating, self.P_operating = I_meas, P_meas
        t0 = time.perf_counter_ns()
        voltage_step = 0
        if time_step > 0:
            dP = P_meas - self.P_previous
            dV = V_meas - self.V_previous
            if abs(dP) > 0.01:                       # minimum power change threshold (original)
                if dP > 0:
                    self.perturbation_direction = 1 if dV > 0 else -1
                else:
                    self.perturbation_direction = -1 if dV > 0 else 1
            voltage_step = self.perturbation_direction * self.step_size
            self.V_operating = float(np.clip(Vc + voltage_step, self.min_voltage, self.max_voltage))
        cost = time.perf_counter_ns() - t0
        if not self.legacy:
            self.V_previous, self.P_previous = V_meas, P_meas
        return dict(V=V_meas, I=I_meas, P=P_meas, V_cmd=self.V_operating, step=voltage_step, compute_ns=cost)


class IncrementalConductanceMPPT(MPPTAlgorithm):
    """IncCond, decision rules identical to the original Cell 4; explicit branch for |dV| <= voltage_tolerance (dI-only logic)."""
    def __init__(self, step_size=None, v_start=None, legacy_bookkeeping=False):
        super().__init__('IncCond', v_start)
        self.step_size = CFG['step_size'] if step_size is None else step_size
        self.conductance_tolerance, self.voltage_tolerance, self.legacy = 0.001, 0.1, legacy_bookkeeping

    def mppt_step(self, plant, G, T, time_step):
        if self.legacy:
            self.V_previous, self.I_previous, self.P_previous = self.V_operating, self.I_operating, self.P_operating
        Vc = self.V_operating
        V_meas, I_meas, P_meas = plant.calculate_pv_characteristics(G, T, Vc)
        self.I_operating, self.P_operating = I_meas, P_meas
        t0 = time.perf_counter_ns()
        voltage_step = 0
        if time_step > 0:
            dI = I_meas - self.I_previous
            dV = V_meas - self.V_previous
            if abs(dV) > self.voltage_tolerance:
                conductance_difference = dI / dV + (I_meas / V_meas if abs(V_meas) > self.voltage_tolerance else 0)
                if abs(conductance_difference) <= self.conductance_tolerance:
                    voltage_step = 0
                elif conductance_difference > self.conductance_tolerance:
                    voltage_step = self.step_size
                else:
                    voltage_step = -self.step_size
            elif abs(dI) > 0.001:
                voltage_step = self.step_size if dI > 0 else -self.step_size
        else:
            voltage_step = self.step_size
        self.V_operating = float(np.clip(Vc + voltage_step, self.min_voltage, self.max_voltage))
        cost = time.perf_counter_ns() - t0
        if not self.legacy:
            self.V_previous, self.I_previous, self.P_previous = V_meas, I_meas, P_meas
        return dict(V=V_meas, I=I_meas, P=P_meas, V_cmd=self.V_operating, step=voltage_step, compute_ns=cost)


class VrefMPPT(MPPTAlgorithm):
    """Sensor-based reference-voltage controller: V_ref = predictor(G,T,V,I); move the command toward it under the actuator slew limit."""
    def __init__(self, predictor, name, slew=None, v_start=None):
        if predictor is None or not hasattr(predictor, 'vref'):
            raise ValueError('a trained predictor is required')
        super().__init__(name, v_start)
        self.predictor, self.slew = predictor, CFG['ref_slew'] if slew is None else slew

    def mppt_step(self, plant, G, T, time_step):
        Vc = self.V_operating
        V_meas, I_meas, P_meas = plant.calculate_pv_characteristics(G, T, Vc)
        t0 = time.perf_counter_ns()
        v_ref = float(np.clip(self.predictor.vref(G, T, V_meas, I_meas), self.min_voltage, self.max_voltage))
        voltage_step = float(np.clip(v_ref - Vc, -self.slew, self.slew))
        self.V_operating = float(np.clip(Vc + voltage_step, self.min_voltage, self.max_voltage))
        cost = time.perf_counter_ns() - t0
        return dict(V=V_meas, I=I_meas, P=P_meas, V_cmd=self.V_operating, step=voltage_step, compute_ns=cost, V_ref=v_ref)


class ANNMPPT(VrefMPPT):
    """Plain (G,T,V,I) network -> V_ref (the 'ANN-MPPT' baseline). Deterministic given its inputs."""
    def __init__(self, predictor, slew=None, name='ANN-MPPT', v_start=None):
        super().__init__(predictor, name, slew, v_start)


class PolyMPPT(VrefMPPT):
    """Quadratic polynomial (G,T) -> V_ref: the equal-sensor non-neural baseline."""
    def __init__(self, predictor, slew=None, name='Poly2(G,T)-MPPT', v_start=None):
        super().__init__(predictor, name, slew, v_start)


class ScanPOMPPT(MPPTAlgorithm):
    """TWO-STAGE controller (Scan -> estimate -> P&O).
       Stage 1: a coarse I-V scan at the K fixed probe voltages hd.PROBES within ONE control cycle (fast converter sweep, not subject to the slew limit);
         the energy of that cycle is the MEAN power over the probes. The estimator returns the voltage of the global maximum:
           mode 'dnn' / 'tree': learned estimator (DNN / gradient-boosted trees) -> Hybrid / tree-scan       mode 'argmax': probe with highest power (no learning)
           mode 'dnn_only': DNN estimate is held until the next scan (ablation without P&O)
       HANDOFF: V_est becomes the command and the baseline P&O (same rules, 0.5 V step) starts there.
       SWITCH-BACK to stage 1: P/G < (1 - scan_trigger) x its running maximum since the last scan (>= scan_min_interval cycles after the last scan), or every scan_period cycles."""
    def __init__(self, name, mode, estimator=None, v_start=None):
        super().__init__(name, v_start)
        if mode in ('dnn', 'dnn_only', 'tree') and (estimator is None or not hasattr(estimator, 'vest')):
            raise ValueError('a trained estimator is required')
        self.mode, self.est, self.po = mode, estimator, PerturbAndObserveMPPT(v_start=v_start)
        self.need_scan, self.since_scan, self.ref_pg, self.handoff = True, 10 ** 9, None, False

    def _scan(self, plant, G, T):
        Vp = hd.PROBES
        meas = [plant.calculate_pv_characteristics(G, T, v) for v in Vp]
        Ip, Pp = np.array([m[1] for m in meas]), np.array([m[2] for m in meas])
        t0 = time.perf_counter_ns()
        v_est = Vp[int(np.argmax(Pp))] if self.mode == 'argmax' else self.est.vest(G, T, Ip)
        v_est = float(np.clip(v_est, self.min_voltage, self.max_voltage))
        cost = time.perf_counter_ns() - t0
        self.V_operating = self.po.V_operating = v_est
        self.po.V_previous, self.po.P_previous, self.po.perturbation_direction = v_est, 0.0, 1
        self.handoff, self.need_scan, self.since_scan, self.ref_pg = True, False, 0, None
        return dict(V=v_est, I=float(Ip.mean()), P=float(Pp.mean()), V_cmd=v_est, step=0.0, compute_ns=cost, scan=1)

    def mppt_step(self, plant, G, T, time_step):
        if self.need_scan or self.since_scan >= CFG['scan_period']:
            return self._scan(plant, G, T)
        if self.mode == 'dnn_only':
            V, I, P = plant.calculate_pv_characteristics(G, T, self.V_operating)
            rec = dict(V=V, I=I, P=P, V_cmd=self.V_operating, step=0.0, compute_ns=0)
        else:
            rec = self.po.mppt_step(plant, G, T, 0 if self.handoff else 1)
            self.V_operating = self.po.V_operating
        self.handoff = False
        pg = rec['P'] / max(G, 1.0)
        if self.ref_pg is None:
            self.ref_pg = pg
        else:
            self.ref_pg = max(self.ref_pg, pg)
            if pg < (1.0 - CFG['scan_trigger']) * self.ref_pg and self.since_scan >= CFG['scan_min_interval']:
                self.need_scan = True
        self.since_scan += 1
        rec['scan'] = 0
        return rec


class PSOMPPT(MPPTAlgorithm):
    """Particle-swarm global MPP tracker on the operating voltage (V and I sensors only). One particle is evaluated per control cycle.
       Fixed a priori (not tuned): n = 4 particles initialised uniformly over the converter window, w = 0.4, c1 = c2 = 1.5 (inside the ranges usually
       recommended for PSO), r1, r2 ~ U(0,1) from a seeded generator, velocity clamp 25 % of the window. Convergence: all particles within 0.3 V of the best
       -> hold the best voltage; re-initialise the swarm when |P - P_hold| / P_hold > 10 % (or after 600 cycles)."""
    N, W_IN, C1, C2, CONV_V, RESTART, MAX_HOLD = 4, 0.4, 1.5, 1.5, 0.3, 0.10, 600

    def __init__(self, seed=0, v_start=None, name='PSO'):
        super().__init__(name, v_start)
        self.rng = np.random.RandomState(seed)
        self._init_swarm(first=True)

    def _init_swarm(self, first=False):
        lo, hi = self.min_voltage, self.max_voltage
        self.x = np.linspace(lo, hi, self.N); self.v = np.zeros(self.N)
        if first: self.x[0] = self.V_operating
        self.pbest_x, self.pbest_p = self.x.copy(), np.full(self.N, -np.inf)
        self.gbest_x, self.gbest_p = self.x[0], -np.inf
        self.i, self.hold, self.hold_p, self.hold_cycles = 0, False, None, 0
        self.V_operating = float(self.x[0])

    def mppt_step(self, plant, G, T, time_step):
        Vc = self.V_operating
        V_meas, I_meas, P_meas = plant.calculate_pv_characteristics(G, T, Vc)
        t0 = time.perf_counter_ns()
        if self.hold:
            self.hold_cycles += 1
            if self.hold_p is None:
                self.hold_p = P_meas
            elif abs(P_meas - self.hold_p) / max(self.hold_p, 1e-9) > self.RESTART or self.hold_cycles > self.MAX_HOLD:
                self._init_swarm()
        else:
            k = self.i
            if P_meas > self.pbest_p[k]:
                self.pbest_p[k], self.pbest_x[k] = P_meas, self.x[k]
            if P_meas > self.gbest_p:
                self.gbest_p, self.gbest_x = P_meas, self.x[k]
            self.i += 1
            if self.i == self.N:                                 # all particles evaluated: update velocities and positions
                r1, r2 = self.rng.uniform(size=self.N), self.rng.uniform(size=self.N)
                vmax = 0.25 * (self.max_voltage - self.min_voltage)
                self.v = np.clip(self.W_IN * self.v + self.C1 * r1 * (self.pbest_x - self.x) + self.C2 * r2 * (self.gbest_x - self.x), -vmax, vmax)
                self.x = np.clip(self.x + self.v, self.min_voltage, self.max_voltage); self.i = 0
                self.gbest_p = max(self.pbest_p)                  # re-evaluate the swarm best from the particles' own bests
                self.gbest_x = self.pbest_x[int(np.argmax(self.pbest_p))]
                if np.max(np.abs(self.x - self.gbest_x)) < self.CONV_V:
                    self.hold, self.hold_p, self.hold_cycles = True, None, 0
            self.V_operating = float(self.gbest_x) if self.hold else float(self.x[self.i])
        cost = time.perf_counter_ns() - t0
        return dict(V=V_meas, I=I_meas, P=P_meas, V_cmd=self.V_operating, step=0.0, compute_ns=cost)
