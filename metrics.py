"""Definitions of every figure of merit used in the study (single source; see docs/METRICS.md).

All functions work on plain numpy arrays / the per-run log DataFrame written by mppt.sim.run_algorithm.
Conventions: P is the power actually delivered at the applied operating point, P_mpp the true (global) maximum power of the ACTUAL plant at
the same instant. One control cycle = one sample; scenario sample period dt is taken from the time axis (1 s here).
"""
import numpy as np

# ------------------------------------------------------------------ constants (fixed a priori, never tuned on test data)
CONV_TOL = 0.02          # converged <=> P/P_mpp >= 1 - CONV_TOL ...
CONV_HOLD = 10           # ... for CONV_HOLD consecutive control cycles
OSC_WINDOW = 10          # cycles per oscillation window
OSC_QUASI_STEADY_V = 0.25  # V: a window is quasi-steady if V_mpp moves at most this much inside it
OSC_SKIP_AFTER_EVENT = 50  # cycles ignored after an event (acquisition transient)
GMPP_TOL = 0.02          # GMPP success: |P - P_gmpp| / P_gmpp <= 2 %
GMPP_TAIL = 10           # ... evaluated as the mean over the last 10 non-scan cycles of a pattern
WRONG_PEAK_V = 2.0       # V: a scan estimate is a 'wrong peak' if it is more than this far from the global-MPP voltage


def _dt(time):
    time = np.asarray(time, float)
    d = np.diff(time)
    return np.append(d, d[-1] if len(d) else 1.0)


def tracking_efficiency(P, P_mpp, time=None):
    """eta_track [%] = 100 * sum(P*dt) / sum(P_mpp*dt)  (energy weighted)."""
    P, P_mpp = np.asarray(P, float), np.asarray(P_mpp, float)
    dt = np.ones_like(P) if time is None else _dt(time)
    return float(100.0 * np.sum(P * dt) / np.sum(P_mpp * dt))


def instantaneous_efficiency(P, P_mpp):
    """eta(k) [%] = 100 * P(k) / P_mpp(k)."""
    return 100.0 * np.asarray(P, float) / np.asarray(P_mpp, float)


def convergence_cycles(eta, events, tol=CONV_TOL, hold=CONV_HOLD):
    """Cycles from each event e (run start, irradiance step, new shading pattern) until eta >= 100(1-tol) holds for `hold` consecutive cycles.
    The search window of an event ends at the next event. Returns a list with an integer per event or nan (not converged).
    NB: single-run values are integers; non-integers appear only when averaged over events/runs."""
    eta = np.asarray(eta, float)
    ok_all = eta >= 100.0 * (1.0 - tol)
    out = []
    for j, e in enumerate(events):
        end = events[j + 1] if j + 1 < len(events) else len(eta)
        ok = ok_all[e:end].astype(int)
        if len(ok) < hold:
            out.append(np.nan); continue
        idx = np.where(np.convolve(ok, np.ones(hold, int), 'valid') == hold)[0]
        out.append(float(idx[0]) if len(idx) else np.nan)
    return out


def oscillation(V, eta, V_mpp, scan, events, window=OSC_WINDOW, quasi_v=OSC_QUASI_STEADY_V, skip=OSC_SKIP_AFTER_EVENT):
    """Steady-state oscillation: mean peak-to-peak of the operating voltage [V] and of eta [%] over non-overlapping `window`-cycle windows that
    (i) are quasi-steady (true V_mpp moves <= quasi_v), (ii) contain no scan cycle, (iii) start >= `skip` cycles after the last event.
    Returns (V_pp, eta_pp, n_windows); nan if no window qualifies."""
    V, eta, V_mpp, scan = (np.asarray(x, float) for x in (V, eta, V_mpp, scan))
    ev = np.array(events)
    av, ap = [], []
    for s in range(0, len(V) - window + 1, window):
        if s < (ev[ev <= s].max() if np.any(ev <= s) else 0) + skip:
            continue
        if np.ptp(V_mpp[s:s + window]) <= quasi_v and scan[s:s + window].sum() == 0:
            av.append(np.ptp(V[s:s + window])); ap.append(np.ptp(eta[s:s + window]))
    return (float(np.mean(av)) if av else np.nan, float(np.mean(ap)) if ap else np.nan, len(av))


def gmpp_success(P, P_mpp, scan, events, tol=GMPP_TOL, tail=GMPP_TAIL):
    """GMPP success rate [%] over the shading patterns (= segments between events): a pattern is a success if the mean power over its last `tail`
    NON-SCAN cycles is within `tol` of the mean true global-MPP power over the same cycles. Returns (rate_percent, list_of_bool)."""
    P, P_mpp, scan = np.asarray(P, float), np.asarray(P_mpp, float), np.asarray(scan, int)
    flags = []
    for j, e in enumerate(events):
        end = events[j + 1] if j + 1 < len(events) else len(P)
        idx = np.arange(e, end)
        idx = idx[scan[idx] == 0][-tail:]
        flags.append(bool(len(idx) and abs(P[idx].mean() - P_mpp[idx].mean()) / P_mpp[idx].mean() <= tol))
    return 100.0 * float(np.mean(flags)), flags


def scan_overhead(P, P_mpp, scan):
    """Energy given up during scan cycles as a percentage of the available energy: 100*sum((P_mpp-P)[scan])/sum(P_mpp)."""
    P, P_mpp, scan = np.asarray(P, float), np.asarray(P_mpp, float), np.asarray(scan, int)
    return float(100.0 * np.sum((P_mpp - P)[scan == 1]) / np.sum(P_mpp))


def wrong_peak_rate(v_cmd, v_gmpp, scan, thr=WRONG_PEAK_V):
    """Fraction [%] of SCAN cycles whose commanded estimate is more than `thr` volts away from the true global-MPP voltage; nan if no scan."""
    v_cmd, v_gmpp, scan = np.asarray(v_cmd, float), np.asarray(v_gmpp, float), np.asarray(scan, int)
    m = scan == 1
    return float(100.0 * np.mean(np.abs(v_cmd[m] - v_gmpp[m]) > thr)) if m.any() else float('nan')


def compute_cost(compute_us):
    """Mean wall-clock controller decision time per control cycle [us] (plant model and logging excluded)."""
    return float(np.mean(compute_us))


def summarise_run(df, events):
    """All per-run metrics from a run log (columns: time, V, P, V_cmd, mpp_voltage, mpp_power, efficiency, scan, compute_us)."""
    conv = convergence_cycles(df.efficiency.to_numpy(), events)
    ov, op, nw = oscillation(df.V, df.efficiency, df.mpp_voltage, df.scan, events)
    return {'tracking_efficiency_%': tracking_efficiency(df.P, df.mpp_power, df.time), 'mean_inst_efficiency_%': float(df.efficiency.mean()),
            'convergence_cycles': float(np.nanmean(conv)) if not np.all(np.isnan(conv)) else np.nan, 'n_events': len(conv), 'n_not_converged': int(np.sum(np.isnan(conv))),
            'osc_amp_V': ov, 'osc_amp_P_%': op, 'osc_windows': nw, 'n_scans': int(df.scan.sum()), 'scan_overhead_%': scan_overhead(df.P, df.mpp_power, df.scan),
            'wrong_peak_%': wrong_peak_rate(df.V_cmd, df.mpp_voltage, df.scan), 'compute_us_per_cycle': compute_cost(df.compute_us)}
