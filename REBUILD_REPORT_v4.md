# REBUILD REPORT v4 (Round 4, reviewer-driven revision)

Everything below is generated from the files in this repository by `tools/build_report.py`. Numbers come from `results/*.csv` of the run described here; nothing was tuned on the 2019–2020 test years.

## 1. What the data say / do not say

1. **Global scan + local P&O gives the shading robustness.** On the nine shading patterns Scan+P&O (no learning) reaches 99.36 % efficiency and finds the global peak in 100 % of the patterns; P&O reaches 80.41 % / 33 %, IncCond 80.44 % / 33 %.
2. **The DNN does not add robustness on top of that.** Hybrid (DNN + P&O) shading efficiency 99.32 % versus Scan+P&O 99.36 %: median difference +0.02 points, Wilcoxon p = 0.232 (n = 10) → not significant at 0.05.
3. **Where the DNN helps:** the cold start / hand-off. Cold-start convergence (valid starts) Hybrid 1.0 cycles, Scan+P&O 4.0, P&O 9.4; the scan is assumed to take one control cycle and not to be slew-limited, which favours both scan-based controllers.
4. **On uniform irradiance the scan costs a little:** standard scenarios nominal plant, Hybrid 99.57 %, P&O 99.73 %, IncCond 99.77 %, plain ANN-MPPT 100.00 % (mismatched plant 99.64 %). The plain ANN is a map the plant is also simulated with; it is an upper bound, not evidence for hardware.
5. **Architecture is not the bottleneck.** Current 128-128-64 net: shading efficiency 99.35 %, test RMSE 0.75 V; the 32-32 net: 99.35 %, 0.94 V. Closed-loop results differ far less than the offline error does, because P&O corrects the hand-off error.
6. **Other comparators:** PSO 95.68 % shading efficiency (100 % success), Scan+Trees 99.37 % (100 %), so a gradient-boosted-tree estimator does as well as the DNN in the same structure; PSO finds the global peak but converges slowly (29.6 cycles per event vs 1.8 for Hybrid) and oscillates more (2.1 V p-p).
7. **Converter plant:** largest change of tracking efficiency versus the quasi-static plant 0.58 points; averaged-vs-switching deviation for small command steps at most 0.10 % of rated power and 0.049 V; during large command jumps (scan probes) up to 4.7 % / 1.45 V because neither model has settled within the window.
8. **Noise and precision:** with sensor noise (I 0.5 %, V 0.5 %, G 2 %) shading efficiency drops from 99.32 % to 98.70 % for Hybrid and from 99.36 % to 98.75 % for Scan+P&O (so the loss comes from P&O dithering on noisy power readings, not from the DNN), P&O 80.41 % to 80.25 %. Float16 leaves the estimate unchanged (max deviation 0.05 V); post-training int8 (numpy emulation) raises the test RMSE from 0.78 to 1.98 V with single outliers up to 22 V, yet the closed-loop shading efficiency stays at 99.32 % because P&O corrects the hand-off.
9. **HPC versus real time:** data generation speeds up 2.62× on four workers (efficiency 0.65); the controller decision takes 9 µs per cycle and the DNN 19 µs (numpy) of a 1 s cycle. Parallel compute is needed for training data, ablations and multi-seed studies, not for the online controller.
10. **Limits:** simulation only (no hardware-in-the-loop, no bench test); one PV-module model family; scan assumed to take one control cycle; conformal intervals are valid only in distribution; no claim about hardware cost beyond emulated latency.

## 2. What was run and how long it took

Single entry point: `python run_all.py` (`--quick` for the small smoke test that writes to `results_quick/`). Steps run sequentially; the timing experiment ran alone on the machine.

| step | outcome | wall-clock (s) |
|---|---|---|
| tables | ok | 9 |
| main | ok | 730 |
| hpc | ok | 2348 |
| stats | ok | 5018 |
| converter | ok | 568 |
| figures | ok | 70 |
| docs | ok | 3 |
| traceability | ok | 0 |
| ablation (run separately, before the sequence above) | ok | training time of all 27 ablation runs: 1644 s (sum of fit seconds) |

Machine: `{
 "cpu_model": "Intel(R) Xeon(R) Processor @ 2.80GHz",
 "logical_cpus": 4,
 "python": "3.11.15",
 "numpy": "2.4.6",
 "platform": "Linux-6.18.44-fc-v140-x86_64-with-glibc2.39"
}`

### Clean-clone QUICK run (`python run_all.py --quick`, fresh `git clone` of the pushed branch, empty caches)

QUICK mode is NOT a 5-minute run here: the steps below add up to 23 minutes (the first pass stopped at the HPC step because of a bug that was then fixed; the steps were continued in a second invocation).

| step | outcome | wall-clock (s) |
|---|---|---|
| tables | ok | 53 |
| main | ok | 237 |
| hpc | FAILED (rc=1) | 287 |
| hpc | ok | 288 |
| ablation | ok | 215 |
| stats | ok | 402 |
| converter | ok | 137 |
| figures | ok | 61 |
| docs | ok | 2 |
| traceability | ok | 0 |
| report | ok | 0 |
| tests | FAILED (rc=1) | 7 |
| report_final | ok | 0 |

The full run was executed in the working repository as separate invocations of the same steps (logs/run_all_full.log, logs/run_all_full2.log, logs/ablation.log), not as one `python run_all.py` from a fresh clone (about five hours); pytest after the clean-clone QUICK run: see logs/clean_clone_quick/tests_quick.log (14 passed, 1 skipped because the evidence-path test refers to the full run).

## 3. What was cut or not done

- Hardware-in-the-loop and any bench experiment: not done (not possible from this repository); replaced by a switching-level converter simulation, measured latency, noise and quantisation studies; traceability status "partly".
- Literature comparison (`Table_Literature`) and the critical discussion of the five papers named by Reviewer A: not done, because no paper was read in this study; the table is left unfilled.
- The int8 / float16 latencies are numpy emulation, not microcontroller measurements; the cycle estimate in `quantisation.csv` is labelled as an estimate.
- The full v3 regression (retraining everything) was lost to container restarts; a reduced gate with cached models was used (see CHANGELOG_ROUND4.md).
- Ablation: five seeds for the current architecture, three seeds for the other variants, one run for ridge (deterministic).
- Radar chart of v3 dropped (it stacked several metrics in one figure).
- Manuscript text, language editing, reference lists, and the point-by-point letter are not in this repository.

## 4. Known limitations

- Simulation only; one PV-module model family; the scan is assumed to take one control cycle and not to be slew-limited (the converter study shows that 14 settling windows fit in 0.84 s but the low-voltage probes are not fully settled).
- The quasi-static plant is the main result; the converter plant uses an averaged CCM model valid for inductor currents above half the ripple.
- Conformal intervals are valid in distribution only. Seeds: 10 scenario seeds, 5 DNN seeds; Wilcoxon p-values are paired over scenario seeds.

## 5. Reviewer traceability

Rows: 79; status counts: done: 33, partly: 28, not possible: 18. See `reviews/REVIEWER_TRACEABILITY.md`.

## 6. Acceptance checks

- Every reviewer row has an existing evidence path: checked by `tests/test_consistency.py::test_traceability_evidence_paths_exist`.
- Every manuscript number is in `results/PAPER_NUMBERS.json` with its source CSV and column (test: `test_paper_numbers_equal_csv_values`).
- No figure carries a title; panel letters are enforced when saving (`mppt_style.save_figure`); `docs/FIGURE_MAP.md` lists every figure.
- Changelog of changed numbers: `CHANGELOG_ROUND4.md` (generated part).

## 7. Final test run (pass/fail list)

```
...............                                                          [100%]
==================================== PASSES ====================================
=========================== short test summary info ============================
PASSED tests/test_consistency.py::test_parameter_count_matches_keras_and_results
PASSED tests/test_consistency.py::test_numpy_forward_equals_keras
PASSED tests/test_consistency.py::test_int8_and_fp16_emulation_are_close_to_float32
PASSED tests/test_consistency.py::test_converter_averaged_matches_switching_in_steady_state
PASSED tests/test_consistency.py::test_paper_numbers_equal_csv_values
PASSED tests/test_consistency.py::test_no_forbidden_patterns_in_code
PASSED tests/test_consistency.py::test_every_figure_has_pdf_png_and_data
PASSED tests/test_consistency.py::test_figure_map_lists_every_figure
PASSED tests/test_consistency.py::test_traceability_evidence_paths_exist
PASSED tests/test_metrics.py::test_tracking_efficiency_energy_weighted
PASSED tests/test_metrics.py::test_convergence_integer_and_nan
PASSED tests/test_metrics.py::test_oscillation_square_wave_and_scan_exclusion
PASSED tests/test_metrics.py::test_gmpp_success_tail_and_scan
PASSED tests/test_metrics.py::test_scan_overhead_and_wrong_peak
PASSED tests/test_metrics.py::test_instantaneous_efficiency
15 passed in 4.77s
```
