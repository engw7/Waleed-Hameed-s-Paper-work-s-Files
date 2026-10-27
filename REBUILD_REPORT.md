> **Superseded by `REBUILD_REPORT_v2.md`** (new physics model and real PVGIS data); the v1 numbers below came from the temperature-only PV model and the defective data file.

# MPPT notebook rebuild — results and findings

Rebuilt `20236359-code-new.ipynb` per `MPPT_Notebook_Audit_Report.md`. The notebook is committed **already executed**; every number below is copied from `results/` (produced by the notebook, `results/results.json` + CSVs). Run time is about 13 min on 4 CPU cores.
Reproducibility: executed end-to-end three times (two in parallel, one alone); all non-timing results are identical (fingerprint `a1eb406dec36c2b0`, `results/fingerprint.txt`). Wall-clock timings naturally vary.

## 1. Real results (nothing tuned toward the manuscript)

Four standard scenarios (Cell-2 definitions: comprehensive, step_change, cloud_passing, ramp_change) are averaged; partial shading is reported separately. Same scenarios, same 0.5 V/cycle actuator limit for all three algorithms.

| metric | P&O | IncCond | ANN-MPPT | manuscript claim |
|---|---|---|---|---|
| Tracking efficiency, 4 std scenarios (energy-weighted) | **99.74 %** | **99.76 %** | **99.997 %** | 98.7 % (ANN), 94.15 % (P&O) |
| Convergence after events (control cycles) | 4.0 | 1.6 | 1.6 | 2.8 steps |
| Cold-start convergence, mean over starts that converged (cycles) | 11.25 (1/5 never converged) | 10.25 (1/5 never) | 10.2 (0/5 never) | – |
| Steady-state voltage oscillation, peak-to-peak | 1.18 V | 1.09 V | 0.006 V | – |
| Steady-state power oscillation, peak-to-peak | 0.82 % | 0.74 % | 0.00 % | – |
| Partial-shading tracking efficiency | 78.6 % | 78.6 % | 76.9 % | – |
| Partial-shading success rate (3 of 9 episodes) | 33.3 % | 33.3 % | 33.3 % | – |
| Compute per control cycle | ~7 µs | ~8 µs | ~5 100–5 600 µs (Keras call; `model.predict` ≈ 90 000 µs) | – |

Per-scenario tables, per-episode shading outcomes, cold-start table and an ANN actuator-slew sweep are in the notebook (Cell 8/9) and `results/*.csv`.

Offline DNN (target = V_mpp, held-out *conditions*): R² 0.9979 (complex G,T,V,I) vs 0.9982 (simple G,T); RMSE 0.0175 V vs 0.0162 V. Paired test over 34 held-out conditions: p = 0.20 (Wilcoxon 0.40) — **the extra V,I inputs give no significant gain; the simple model is nominally better.** This replaces the old R² = 0.744 / 99.57 % accuracy figures; neither can be reproduced from this pipeline.
Deep-ensemble spread is *not* a calibrated confidence: ±1.96·σ covers only 17 % of test errors (Spearman σ-vs-|error| = 0.16).

**How to read these numbers honestly**
- ANN-MPPT ≈ 100 % is close to tautological here: the DNN learns the same simulator's V_mpp(T) map that is used to score it, and it is also given a temperature sensor that P&O/IncCond do not have. It is an upper bound for this model, not evidence for hardware.
- In the PV model kept from Cell 4, **V_mpp depends only on temperature, not on irradiance** (`results/figures/01_pv_model.png`). Irradiance steps therefore do not move the MPP voltage, which is why the step-change convergence is ~0–1 cycle for everyone. The model is also an explicit-exponential approximation (no Rs/Rsh use) and its nameplate V_mpp·I_mpp (26.3 V × 11.4 A) exceeds I_sc (9.87 A).
- With a flat P-V peak and 0.5 V steps, P&O/IncCond lose only ~0.25 %; the manuscript's P&O ≈ 94 % is not reproduced.
- Under partial shading all three lock onto the first (≈28 V) peak in 6/9 patterns; the ANN, fed mean irradiance, is slightly worse (76.9 % vs 78.6 %). No algorithm here is a global-MPP tracker.

## 2. Findings beyond the audit (please read)

1. **P&O and IncCond were broken in the original Cell 4** (the "real, working" part). `V_previous` was overwritten with the current operating point right before measurement, so `dV` was always 0 and both algorithms oscillated 26.0↔26.5 V forever, never reaching the 27.8 V MPP (`results/figures/10_legacy_bug.png`). I kept their decision rules and thresholds unchanged and corrected only the state-update order; the old behaviour remains available behind `legacy_bookkeeping=True` solely for that demonstration. This deviates from "don't touch Cell 4" — it was necessary.
2. **`PV_Data.xlsx` is not 166 536 independent observations.** Only 610 distinct rows; 99.7 % of rows are exact repeats of 100 patterns (a random split leaks). Only **223 unique daytime (G,T) conditions** (G 51–794 W/m², T 2.3–19.1 °C). `P` is not row-aligned with `G(i)` (P>0 while G=0 in 18 % of rows; best cross-correlation at a 20-row lag, 0.86 vs 0.21 at lag 0) and agrees with Vm·Im within 15 % on only 5.7 % of daytime rows; `Im`/`Vm` are literally the same closed-form formulas of (G,T) in all 166 536 rows.
   **Decision (audit §2.2 option a):** `Im`, `Vm`, `P` are not used as labels; the real `G`, `Temp` define operating conditions, labels come from the PV model (P = V·I exactly), splits are by condition. If you have the original, correctly aligned PVGIS export, it should replace this file.
3. **More fabrication removed than the audit listed:** Cell 5 multiplied ANN metrics by hand-picked "ANN better" factors (×0.8, 0.7, 0.6, 0.5, energy ×1.02); Cell 3 deliberately handicapped the simple model (fewer epochs, bigger batch, more dropout, "intentionally less optimal"). Both gone; models now share identical settings.
4. **Scenario coverage:** the real-data envelope covers 22 % of the `comprehensive` scenario and 0 % of the others (scenarios run to 35 °C / 1200 W/m²; data ≤ 19 °C / 794 W/m²). Extrapolation turned out benign (V_mpp is smooth in T; RMSE 0.065 V on a synthetic envelope), and a clearly-labelled "wide-envelope ablation" model gives 99.999 % — same conclusion.
5. **Cold start from 35 V** (above V_oc): P&O and IncCond never converge (zero power ⇒ no gradient); the ANN does (12 cycles).

## 3. Acceptance checklist (audit §3.6) — verified by code in the notebook's last cell

| # | item | status |
|---|---|---|
| 1 | `PV_Data.xlsx` loads under its exact name; no synthetic fallback | PASS (loader raises on missing file / bad columns / NaN) |
| 2 | P vs Vm×Im inconsistency resolved and documented | PASS (Cell 2; see §2.2) |
| 3 | `_ensure_complex_superiority`, forced p-value, `performance_targets` physically deleted | PASS (token scan of all code cells: none found) |
| 4 | Exactly one shared state object | PASS (`PipelineState`: 1 instantiation, a 2nd raises, `put` refuses silent overwrite, `require` fails loudly) |
| 5 | `predict_optimal_voltage` path calls the real trained model; no mock path | PASS (30 cycles → 30 Keras forward calls, V_ref equals model output; no `random` in ANN code; the old method is replaced by `ANNMPPT.mppt_step` → `VoltagePredictor.vref`) |
| 6 | `create_demo_data()` / `TRACKING_RESULTS` deleted; tables read from real logs | PASS (no manuscript constants or result literals; all tables from `STATE`) |
| 7 | Same Cell-2 scenarios for all algorithms; time-domain plots from logs | PASS (5 scenarios × identical (G,T) arrays; figures `03`–`08`) |
| 8 | Two end-to-end runs give the same numbers | PASS (identical fingerprint across 3 runs) |
| 9 | Report whatever comes out | Done — §1 |

Metric definitions (precise) are printed in Cell 8 and stored in `results/results.json`. Convergence = first cycle after an event where P/P_mpp ≥ 98 % for 10 consecutive cycles. Because V_mpp ignores irradiance in this model, I added a cold-start protocol (five initial voltages) so convergence is not trivially ~0.

## 4. Files
`20236359-code-new.ipynb` (executed) · `PV_Data.xlsx` (renamed from `PV Data.xlsx`, content unchanged) · `results/` (CSV/JSON tables, 11 figures, trained models) · this report.

## 5. Judgement calls to confirm
- DNN output = V_ref (not power); deployed model = the (G,T,V,I) one, fixed beforehand.
- ANN gets the same 0.5 V/cycle slew as P&O/IncCond (sweep in `results/ann_slew_sensitivity.csv` shows this barely matters here).
- Boost duty cycle (`D = 1 − V/48 V`) is a display-only signal; no converter dynamics are simulated.
- The manuscript's abstract/tables will need to be rewritten to these numbers.
