# Changelog Round 4 (old -> new -> reason)

Generated/maintained during the round; every number that changes relative to `REBUILD_REPORT_v3.md` / `reference/v3` is listed here.

## Step 1 - regression gate (refactored `mppt/` library vs frozen v3 notebook)
The full regression (`tools/regress_v3.py`, retrains all models) was lost twice to container restarts, so the reduced gate of `tools/regress_v3_reduced.py` was used: cached seed-42 models, all seven scenarios, nominal + mismatched plant (`logs/regress_reduced_join.csv`).

| controller | max abs difference in tracking efficiency vs v3 (points) | remark |
|---|---|---|
| P&O, IncCond, Scan+P&O (no ML) | < 1e-5 | non-ML controllers reproduce v3 exactly |
| DNN scan-only, Hybrid (DNN+P&O) | < 1e-5 | cached hybrid network is the one trained in the same pipeline |
| ANN-MPPT (plain G,T,V,I) | 0.006 (mean 0.001) | network re-trained in the new library; the origin of the small difference (re-training vs refactor) was not isolated |

GMPP success on the nine shading patterns is identical to v3 for all controllers except ANN-MPPT on the mismatched plant (22.2 % new vs 33.3 % v3). Cause: the success criterion was changed on purpose (v3: >=95 % over the last 25 cycles; Round 4: >=98 % over the last 10 non-scan cycles, `metrics.py`) and the ANN was re-trained. Poly2(G,T) was not part of the reduced gate. Gate passed: tolerance +-0.05 points on tracking efficiency met everywhere.

<!-- GENERATED: do not edit below this line -->

## Every number that changed against `reference/v3` (old -> new -> reason)

91 numbers have a v3 counterpart; 9 differ by more than 0.1 % relative / 0.005 absolute; the others are unchanged.

| key | old (v3) | new | unit | reason |
|---|---|---|---|---|
| headline.PO.compute_us_per_cycle | 4.677 | 8.247 | us | wall-clock timing, depends on machine and run |
| headline.IncCond.compute_us_per_cycle | 4.344 | 9.082 | us | wall-clock timing, depends on machine and run |
| headline.ANN.gmpp_success_mismatched_% | 33.33 | 22.22 | % | success criterion changed on purpose (>=98 % over the last 10 non-scan cycles, v3: >=95 % over the last 25 % of an episode) |
| headline.ANN.compute_us_per_cycle | 3200 | 55.83 | us | the learned model is now evaluated with a numpy re-implementation of the forward pass (v3 timed the Keras/sklearn call); framework latencies are in Table_HPC |
| headline.Poly2.gmpp_success_mismatched_% | 33.33 | 22.22 | % | success criterion changed on purpose (>=98 % over the last 10 non-scan cycles, v3: >=95 % over the last 25 % of an episode) |
| headline.Poly2.compute_us_per_cycle | 441.9 | 42.24 | us | the learned model is now evaluated with a numpy re-implementation of the forward pass (v3 timed the Keras/sklearn call); framework latencies are in Table_HPC |
| headline.ScanPO.compute_us_per_cycle | 4.357 | 9.706 | us | wall-clock timing, depends on machine and run |
| headline.DNNonly.compute_us_per_cycle | 14.85 | 0.5246 | us | the learned model is now evaluated with a numpy re-implementation of the forward pass (v3 timed the Keras/sklearn call); framework latencies are in Table_HPC |
| headline.Hybrid.compute_us_per_cycle | 17.65 | 8.898 | us | the learned model is now evaluated with a numpy re-implementation of the forward pass (v3 timed the Keras/sklearn call); framework latencies are in Table_HPC |

## Claims of the earlier manuscript that the new evidence does not support

| earlier claim | what the Round-4 data show |
|---|---|
| ANN-MPPT tracking efficiency 98.7 %, P&O 94.15 %, IncCond 95.8 % | standard scenarios, nominal plant: ANN-MPPT 100.00 %, P&O 99.73 %, IncCond 99.77 %; the old figures are not reproducible with any metric definition |
| tracking accuracy 99.57 %, R2 = 0.744, MSE table | no definition reproduces these; the quantities are dropped (docs/METRICS.md) |
| convergence 2.8 steps (ANN) vs 6.2 / 8.5 | event convergence (control cycles): ANN-MPPT 0.88, P&O 1.75, IncCond 2.50, Hybrid 1.75; integer per event, non-converging runs counted separately |
| ANN-MPPT is robust to partial shading | shading efficiency ANN-MPPT 77.99 % (GMPP success 33 %), mismatched plant 76.98 %; it is a single-peak map |
| oscillations are eliminated | steady-state voltage oscillation p-p: P&O 1.16 V, Scan+P&O 1.17 V, Hybrid 1.23 V (the hybrid ends in P&O); only the map-based controllers and the DNN scan-only variant hold a constant voltage |
| a DNN is needed for shading robustness | Scan+P&O without any learning reaches 99.36 % vs Hybrid 99.32 % |
| Introduction: 1.3-2.1x speed-up on four workers | measured on this machine: training-data generation 2.62x on four workers (2 workers: 1.62x; 8 workers on 4 CPUs: 1.90x, oversubscribed) |
| controller compute 17.6 us per cycle (Hybrid, v3) | 8.9 us per cycle with the numpy forward pass used in the closed loop (framework call timings are in Table_HPC) |

## Converter plant versus quasi-static plant: efficiency changes larger than 0.1 points

4 of 25 scenario/controller pairs; the quasi-static plant stays the main result, the converter plant is a secondary study.

| scenario | controller | static (%) | converter (%) | change (points) |
|---|---|---|---|---|
| comprehensive | IncCond | 99.566 | 99.465 | -0.101 |
| partial_shading | P&O | 80.408 | 80.279 | -0.129 |
| partial_shading | IncCond | 80.438 | 80.297 | -0.141 |
| partial_shading | ANN-MPPT | 77.989 | 77.405 | -0.584 |
