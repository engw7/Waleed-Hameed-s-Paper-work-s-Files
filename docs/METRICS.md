# Metric definitions (single source: `metrics.py`)

Every figure of merit in the study is computed by the functions below; constants are fixed a priori and never tuned on the 2019–2020 test years.

| constant | value | meaning |
|---|---|---|
| `CONV_TOL` | 0.02 | converged when P/P_mpp >= 1 - CONV_TOL |
| `CONV_HOLD` | 10 | ... for this many consecutive control cycles |
| `OSC_WINDOW` | 10 | cycles per oscillation window |
| `OSC_QUASI_STEADY_V` | 0.25 | a window is quasi-steady if V_mpp moves at most this much (V) |
| `OSC_SKIP_AFTER_EVENT` | 50 | cycles ignored after an event |
| `GMPP_TOL` | 0.02 | GMPP success: |P - P_gmpp|/P_gmpp <= this |
| `GMPP_TAIL` | 10 | ... mean over the last this-many non-scan cycles of a pattern |
| `WRONG_PEAK_V` | 2.0 | a scan estimate is a wrong peak beyond this distance from V_gmpp (V) |

## Functions

### `tracking_efficiency(P, P_mpp, time=None)`

eta_track [%] = 100 * sum(P*dt) / sum(P_mpp*dt)  (energy weighted).

### `instantaneous_efficiency(P, P_mpp)`

eta(k) [%] = 100 * P(k) / P_mpp(k).

### `convergence_cycles(eta, events, tol=0.02, hold=10)`

Cycles from each event e (run start, irradiance step, new shading pattern) until eta >= 100(1-tol) holds for `hold` consecutive cycles.
The search window of an event ends at the next event. Returns a list with an integer per event or nan (not converged).
NB: single-run values are integers; non-integers appear only when averaged over events/runs.

### `oscillation(V, eta, V_mpp, scan, events, window=10, quasi_v=0.25, skip=50)`

Steady-state oscillation: mean peak-to-peak of the operating voltage [V] and of eta [%] over non-overlapping `window`-cycle windows that
(i) are quasi-steady (true V_mpp moves <= quasi_v), (ii) contain no scan cycle, (iii) start >= `skip` cycles after the last event.
Returns (V_pp, eta_pp, n_windows); nan if no window qualifies.

### `gmpp_success(P, P_mpp, scan, events, tol=0.02, tail=10)`

GMPP success rate [%] over the shading patterns (= segments between events): a pattern is a success if the mean power over its last `tail`
NON-SCAN cycles is within `tol` of the mean true global-MPP power over the same cycles. Returns (rate_percent, list_of_bool).

### `scan_overhead(P, P_mpp, scan)`

Energy given up during scan cycles as a percentage of the available energy: 100*sum((P_mpp-P)[scan])/sum(P_mpp).

### `wrong_peak_rate(v_cmd, v_gmpp, scan, thr=2.0)`

Fraction [%] of SCAN cycles whose commanded estimate is more than `thr` volts away from the true global-MPP voltage; nan if no scan.

### `compute_cost(compute_us)`

Mean wall-clock controller decision time per control cycle [us] (plant model and logging excluded).

## Notes

- *Tracking efficiency* is energy-weighted: eta = 100 sum(P dt) / sum(P_mpp dt), P_mpp is the true maximum of the actual plant (global maximum under shading). The earlier manuscript quantity "tracking accuracy" (99.57 %) is not reproducible from any definition and is dropped.
- *Convergence* is an integer number of control cycles (1 cycle = 1 s) per event, averaged over events that converge before the next event; non-integer values such as 2.8 are means over events/scenarios, and runs that never converge are counted separately (`n_not_converged`).
- *Scan overhead* is the energy given up during scan cycles; it is included in the tracking efficiency of scan-based controllers. The scan is assumed not to be slew-limited (one control cycle); the converter study shows that 14 settling windows fit in 0.84 s.
- DNN offline metrics (RMSE/MAE of V_gmpp, wrong-peak rate) describe the stage-1 estimator only; they are not MPPT performance metrics.
