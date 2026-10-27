# MPPT notebook — v3: hybrid DNN+P&O controller and publication figure suite

Supersedes the controller conclusions of `REBUILD_REPORT_v2.md` (v2's physics, data and baseline rows are retained unchanged). `20236359-code-new.ipynb` is committed **executed**; all numbers below are read from `results/` (CSV/JSON written by the notebook). Executed three times (two in parallel): identical non-timing results every time (fingerprint `8edac44edf34ec96`); all 16 automated acceptance checks pass. Figure captions: `results/figures/FIGURE_CAPTIONS.md`.

## 1. What was built

**Hybrid controller (two stage, `ScanPOMPPT`, Cell 6).**
1. *Stage 1 — global search.* A fast I-V scan at 14 fixed probe voltages (8-35 V) is done within one control cycle; the energy delivered in that cycle is the *mean* power over the probes, so the scan has a real cost (included in every efficiency number; reported as `scan_overhead_%`). The DNN receives (G_in, T, 14 scan currents) and returns the voltage of the **global** maximum.
2. *Handoff.* The estimate becomes the operating voltage and the **same P&O as the baseline** (same rules, 0.5 V step) starts there.
3. *Switch-back.* Re-scan when P/G falls more than 10 % below its running maximum since the last scan (≥ 30 cycles after the previous scan), or every 300 cycles. All constants were fixed a priori. One design change was made *before any full-scale result existed*: the first rule triggered on any ±10 % drift and a debug run with an under-trained DNN showed rescan loops, so it was changed to drop-only with a ratcheting reference (documented in the class docstring).
4. Ablations/baselines isolating each stage: **Scan+P&O (no ML)** (probe with the highest measured power → P&O) and **DNN scan-only** (DNN estimate held until the next scan, no P&O). Plain **ANN-MPPT** and **Poly2(G,T)** from v2 are kept as rows.

**Domain-randomised, shading-aware training data** (`pv_hybrid_data.py`): every one of the 200,000 training samples has its own plant — Isc ×0.90-1.05, Voc ×0.95-1.06, Rs ×0.8-2.4, Rsh ×0.3-1.5, n 1.05-1.40 around the datasheet extraction (Iph, I0 follow in closed form) — a real (G, T2m) hour from 2005-2014, and (60 % of samples) its own shading pattern: three bypass-diode substrings of 18 cells, 0-18 shaded cells per substring, shaded-cell irradiance factor 0.10-0.85. Method: bypass-diode series-string model (Patel & Agarwal, IEEE T-EC 23(1), 2008) with single-diode cells solved by the Lambert-W inverse (Jain & Kapoor 2004; De Soto et al. 2006); validated against the existing shaded-string model (agreement to 3 decimals). Splits are chronological (train ≤ 2014, validate 2015-16, conformal calibration 2017-18, test 2019-20); the nine closed-loop test patterns were fixed a priori and are *not* in the training list (they belong to the same pattern family).

**Compute actually spent (measured):** 200,000 plant × shading samples generated in **42 s wall-clock on 4 worker processes** (≈74 s serial-equivalent; results bit-identical to serial). The measured multiprocessing speedup on a 12,000-sample benchmark was only **1.30×** (process start-up dominates at this size). DNN fit: 48 epochs, 57 s, 27,009 parameters. Total ≈ 99 s — modest, not orders of magnitude above v2's plain ANN (~72 s).

## 2. Results (nothing tuned toward a desired outcome)

| metric | P&O | IncCond | Poly2(G,T) | ANN-MPPT | Scan+P&O (no ML) | DNN scan-only | **Hybrid (DNN+P&O)** |
|---|---|---|---|---|---|---|---|
| Sensors | V,I | V,I | G,T | G,T,V,I | G,T + fast I-V scan | same | same |
| Tracking eff., std4, nominal | 99.73 | 99.77 | 99.81 | **99.996** | 99.58 | 99.58 | 99.57 |
| Tracking eff., 2 real days, nominal | 99.80 | 99.81 | 99.96 | **99.999** | 99.64 | 99.73 | 99.65 |
| Tracking eff., std4, **mismatched plant** | **99.75** | **99.77** | 99.34 | 99.64 | 99.62 | 99.57 | 99.61 |
| Tracking eff., real days, mismatched | **99.82** | **99.82** | 99.25 | 99.43 | 99.68 | 99.71 | 99.68 |
| **Partial shading, nominal: efficiency / patterns solved** | 80.4 / 3 of 9 | 80.4 / 3 | 78.2 / 3 | 78.0 / 3 | 99.36 / **9** | 99.31 / **9** | **99.33 / 9** |
| **Partial shading, mismatched: efficiency** | 81.1 | 81.2 | 77.2 | 77.0 | 99.36 | 99.33 | **99.37** |
| Event convergence (cycles, std4) | 1.75 | 2.5 | 0.88 | 0.88 | 2.9 | 1.75 | 1.75 |
| Cold-start convergence, protocol B (cycles) | 9.4 | 8.6 | 7.8 | 7.8 | 4.0 | **1.0** | **1.0** |
| Steady-state V oscillation p-p (V) | 1.16 | 1.04 | 0.04 | 0.03 | 1.17 | 0.00 | 1.23 |
| Scan overhead (% energy) | – | – | – | – | 0.15 | 0.15 | 0.15 |
| Compute per cycle (µs, this machine) | 4.7 | 4.3 | 442 | 3,200 | 4.4 | 14.8 | **17.6** (scan cycle ≈ 2.8 ms) |

**Plant-mismatch sweep** (efficiency %, severity 0 / 0.5 / 1 / 1.5 / 2; severities 1.5 and 2 lie *outside* the hybrid DNN's randomisation box), "comprehensive" scenario: ANN 99.99 / 99.94 / 99.76 / 99.44 / 99.00; Poly2 99.51 / 99.32 / 98.96 / 98.46 / 97.85; P&O 99.54-99.57 throughout; **Hybrid 99.40 / 99.44 / 99.42 / 99.45 / 99.43 (flat)**. Partial-shading sweep (severity 0 / 1 / 2): Hybrid 99.33 / 99.37 / 99.47, P&O 80.4 / 81.1 / 81.6, ANN 78.0 / 77.0 / 76.0. *(Correction to v2: its severity-3 sweep point had a negative Rsh, 1 − 0.4·3 < 0, and is invalid; it is dropped here.)*

**Hybrid stage-1 DNN, offline (held-out 2019-20):** power ratio P(V_est)/P_gmpp before any P&O refinement — in-distribution mixture 99.22 % (best-probe rule: 99.02 %); nominal uniform 99.97 % (99.50 %); nominal shaded 98.77 % (98.64 %); **out-of-range plants 98.44 % (best-probe rule 99.37 % — the DNN is *worse* than the simple rule outside its training range)**. Conformal intervals (90 % / 95 %): in-distribution coverage 0.899 / 0.949 (half-width 0.41 / 0.73 V); shaded samples 0.83 / 0.92; **out-of-range plants only 0.41 / 0.76**. Plain-ANN conformal results are unchanged from v2.

## 3. Honest assessment

**What is genuinely better.** Under partial shading the hybrid reaches 99.3 % where P&O/IncCond/Poly2/ANN reach 78-81 %, and finds the global peak in 9 of 9 patterns versus 3 of 9 — on both the nominal and the mismatched plant. Its efficiency is also nearly independent of plant mismatch (flat 99.4 %), unlike the plain ANN and Poly2, and it needs ≈ 180× less compute per cycle than the plain ANN (the network runs only at scans).

**What the DNN is *not* responsible for.** The non-ML **Scan+P&O reaches the same shading and mismatch performance** (99.36 % vs 99.33 %; 99.36 % vs 99.37 %). The gain comes from the *global scan + local P&O architecture*, not from the neural network. The DNN's measurable extra value is (i) faster convergence after a scan (1 versus 4 cycles from a cold start, because its estimate is precise rather than on a 2 V probe grid) and (ii) a more accurate stage-1 estimate in-distribution (99.97 % vs 99.50 % on uniform irradiance) — worth little in closed loop because P&O closes the gap anyway. Out of its training range the simple probe rule is the better estimator.

**Gaps that are not closed.**
- On uniform irradiance the hybrid is *below* P&O by ≈ 0.16 percentage points (std4 nominal 99.57 vs 99.73 %; real days 99.65 vs 99.80 %) — almost exactly its scan overhead (0.15 %) — and ≈ 0.35 points below the plain ANN on the nominal plant (99.57 vs 99.996 %).
- On the mismatched plant (severity 1) the sensor-less P&O/IncCond still beat it by ≈ 0.14 points on the standard scenarios (99.75 / 99.77 vs 99.61 %), and the plain ANN is marginally better there too (99.64 %) until severity ≥ 1.5. On the two real days the hybrid (99.68 %) beats ANN and Poly2 but not P&O (99.82 %).
- The DNN scan-only variant matches the hybrid in efficiency with zero ripple: the P&O stage adds almost nothing on this flat P-V peak except adaptation to mismatch (+0.04 pp) and robustness when irradiance drifts between scans (cloud scenarios).
- Conformal uncertainty is valid only in distribution; it fails for out-of-range plants.

**Defensible claim for the manuscript:** a *scan-guided hybrid* (global search + P&O) is the first architecture here that is robust to partial shading and to plant mismatch at once, at ≈ 4× P&O compute and ~0.15 % scan overhead; the DNN improves cold-start/handoff precision but is not required for the robustness. Claims of general ANN superiority are not supported.

## 4. Caveats (read before citing)
- **Cold-start advantage assumes** the converter can perform the I-V sweep and the handoff jump without the 0.5 V/cycle slew limit that P&O/IncCond/Poly2/ANN obey. With that limit applied to the handoff the scan-based controllers would not converge faster; the 1-cycle figure is conditional on that hardware assumption (also stated in the B4_11 and B5_02 captions).
- Scan = one control cycle at the mean probe power; real sweeps take milliseconds to hundreds of ms and need dedicated hardware support.
- Labels, plants and the nominal test plant all derive from the same physics model; the mismatched plant is a declared, arbitrary parameter perturbation (severity 1 lies inside the DR box by construction; severities 1.5-2 are outside). No hardware or sensor noise was simulated.
- The re-scan constants (10 %, 30 cycles, 300 cycles), probe grid (14 points) and DR ranges were fixed a priori and not tuned; different values may shift the 0.15 % overhead trade-off.
- Cold-start protocol B and the severity range were set after v2 (disclosed in v2). Real-day scenarios are time-compressed hourly data.

## 5. Figure suite (31 figures, each as 300-dpi PNG + vector PDF in `results/figures/`; one style module `mppt_style.py`; captions in `FIGURE_CAPTIONS.md`)
**B.1 data:** B1_01_hist_irradiance, B1_02_hist_temperature, B1_03_scatter_G_vs_T_by_month, B1_04_heatmap_hour_month_G, B1_05_representative_days, B1_06_sanity_P_vs_G.
**B.2 physics:** B2_01_PV_curve_families, B2_02_Vmpp_vs_G_and_T, B2_03_STC_validation.
**B.3 DNN:** B3_01_loss_curves, B3_02_predicted_vs_actual, B3_03_residual_histograms, B3_04_conformal_calibration.
**B.4 time domain (P(t), V(t), duty(t); all controllers overlaid; true MPP and 98 % band):** B4_01_timedomain_step_change, B4_02_…cloud_passing, B4_03_…ramp_change, B4_04_…comprehensive, B4_05_…partial_shading, B4_06_…real_clear_day, B4_07_…real_variable_day, B4_08_…comprehensive_MISMATCHED_plant, B4_09_…partial_shading_MISMATCHED_plant, B4_10_…real_clear_day_MISMATCHED_plant, B4_11_coldstart, B4_12_partial_shading_PV_snapshots.
**B.5 summary:** B5_01_tracking_efficiency, B5_02_convergence_time, B5_03_oscillation_amplitude, B5_04_compute_cost (log scale), B5_05_radar_summary, B5_06_mismatch_sweep.
The v2 figure set (01-16) was replaced by this suite; v2 versions remain in git history.

## 6. Files
`20236359-code-new.ipynb` (executed) · `pv_single_diode.py` (single-diode model; faster shaded-string solver, identical to 2.5e-9 A) · `pv_hybrid_data.py` (vectorised + multiprocessing DR/shading generator) · `mppt_style.py` · `make_captions.py` · `results/` (CSV/JSON tables incl. `hybrid_offline_metrics.csv`, `hybrid_conformal_coverage.csv`, `mismatch_sweep.csv`, `ann_advantage.csv`; trained models; figures).
