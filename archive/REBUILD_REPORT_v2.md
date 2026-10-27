> **Controller conclusions superseded by `REBUILD_REPORT_v3.md`** (hybrid DNN+P&O controller, domain-randomised training, full figure suite).

# MPPT notebook — v2: physics fixed, real PVGIS data, fair comparison

Supersedes the numbers in `REBUILD_REPORT.md` (v1). `20236359-code-new.ipynb` is committed **executed**; every number below is copied from `results/` (CSV/JSON written by the notebook). Executed end-to-end twice in parallel: identical non-timing results (fingerprint `c63d14e83d78206a`); all 13 automated acceptance checks pass (last notebook cell).

## 1. What changed, and why it matters

| | v1 (circular) | v2 |
|---|---|---|
| PV model | explicit exponential, Voc fixed, Isc ∝ G ⇒ **V_mpp exactly independent of G** (27.8 V at G = 100/400/1000) — a simplification (no Rs/Rsh use, no log-G Voc), not a coding bug | five-parameter single-diode (Rs, Rsh, Lambert-W, De Soto/Duffie-Beckman equations cited in `pv_single_diode.py`), cell temperature from NOCT. STC reproduced exactly (244.6 W, 32.9 V, 9.87 A); emergent dVoc/dT −0.33 %/K, dPmax/dT −0.41 %/K. At fixed cell temperature V_mpp is −2.1 % at 400, −5.0 % at 200, −8.6 % at 100 W/m² vs STC |
| data | `PV_Data.xlsx`: 610 distinct rows, P misaligned | **PVGIS-SARAH3, Ramadi (33.42 N, 43.30 E), 2005-2020, 140,256 hourly rows** (`PVGIS_Ramadi_2005-2020.csv`, `PV_Data.xlsx` removed) |
| labels / scoring | same T-only relation for training and scoring, random split | chronological split (train ≤2014, calibrate 2017-18, test 2019-20), hot-range split, irradiance-range split, **mismatched plant** the controllers were not trained on, 2 held-out real days |
| sensors | ANN had G,T sensors, P&O/IncCond did not — unstated | stated in the notebook, in `results.json` and here; **Poly2(G,T)-MPPT** (quadratic regression, same sensors, same labels) added as the equal-information baseline |
| confidence | 5-model spread, covered 17 % | split-conformal intervals with verified coverage (§3) — the old "confidence" claim is dropped |

**Data sanity (all checks in Cell 3):** no duplicate/missing timestamps; G = 0 at every night row (G > 0 at night: 0 rows); night 21.5 °C vs day 27.5 °C mean; all 12 months present (≥ 4,212 daytime rows each); annual GHI 1,999 kWh/m²; **61,231 distinct daytime (G,T) pairs of 61,738** (old file: 610 distinct rows in 166,536); P = 0 W whenever G = 0 and P = Vm·Im exactly (computed from the model — no closed-form formulas). V_mpp over the real data spans 20.2–27.6 V. Linear R² of V_mpp: T-only 0.921, G-only 0.289, G+T 0.943: **temperature still dominates; irradiance adds the weaker part.**

## 2. Real final numbers (as computed, nothing tuned)

std4 = comprehensive, step_change, cloud_passing, ramp_change; real days = 2020-06-24 (clear) and 2020-04-06 (variable) from the held-out test years, time-compressed (13/11 hourly samples → 3600 cycles).

| metric | P&O | IncCond | Poly2(G,T) | ANN-MPPT |
|---|---|---|---|---|
| Sensors used | V, I | V, I | **G, T** | **G, T, V, I** |
| Tracking eff., std4, nominal plant | 99.73 % | 99.77 % | 99.81 % | **99.996 %** |
| Tracking eff., 2 real days, nominal | 99.80 % | 99.81 % | 99.96 % | **99.999 %** |
| Tracking eff., std4, **mismatched plant** | **99.75 %** | **99.77 %** | 99.34 % | 99.64 % |
| Tracking eff., real days, **mismatched** | **99.82 %** | **99.82 %** | 99.25 % | 99.43 % |
| Event convergence (cycles, mean) | 1.75 | 2.5 | 0.875 | 0.875 |
| Cold-start convergence, B (cycles) | 9.4 | 8.6 | 7.8 | 7.8 |
| Steady-state V oscillation p-p | 1.16 V | 1.04 V | 0.04 V | 0.03 V |
| Partial-shading efficiency / success rate | **80.4 %** / 33 % | **80.4 %** / 33 % | 78.2 % / 33 % | 78.0 % / 33 % |
| Compute per cycle (this machine) | ~5 µs | ~5 µs | ~400 µs | ~3,000 µs (`model.predict`: ~69,000 µs) |

ANN advantage in efficiency points (mean over scenarios): nominal plant **+0.14 vs Poly2, +0.22 vs best sensor-less**; mismatched plant **+0.26 vs Poly2, −0.22 vs best sensor-less**. Per scenario (nominal, vs Poly2): comprehensive +0.48, cloud_passing +0.26, other four ≤ +0.05.

**Plant-mismatch sweep** ("comprehensive"; controllers trained on the nominal plant; severity scales Iph −8 %, Rs +80 %, Rsh −40 %, n +5 % per unit): ANN 99.99 / 99.94 / 99.76 / 99.00 / 97.34 % at severity 0 / 0.5 / 1 / 2 / 3; P&O stays 99.54–99.61 %, IncCond 99.57–99.63 %, Poly2 99.51 → 95.81 %. The ANN loses to P&O somewhere between severity 1 and 2.

**Offline DNN (target V_mpp, volts):**

| protocol (held-out) | Poly2(G,T) RMSE / R² | ANN(G,T,V,I) RMSE / R² | fixed V RMSE |
|---|---|---|---|
| S1 later years 2019-20 | 0.160 V / 0.988 | **0.011 V / 0.9999** | 1.48 V |
| S2 train T<30 °C → test T≥38 °C | **0.193 V / 0.861** | 0.219 V / 0.820 | 3.01 V |
| S3 train G<600 → test G≥850 | 2.29 V / **−9.37** | **0.175 V / 0.940** | 2.54 V |

In terms of delivered power the S1 gap is tiny (P(V_ref)/P_mpp 99.9998 % ANN vs 99.964 % Poly2) because the P-V peak is flat. The (G,T) ANN is *better* than the (G,T,V,I) one (|err| 0.0028 vs 0.0076 V, paired over 731 test days, p ≈ 0, Wilcoxon 3e-121): feeding V and I adds noise to a label that depends only on (G,T).

## 3. Uncertainty (conformal, replaces the 17 % ensemble "confidence")

Split-conformal absolute-residual intervals on V_mpp, calibrated on 2017-18, tested on 2019-20: nominal 90 % → **empirical 90.0 %** (half-width 0.016 V); nominal 95 % → **94.8 %** (0.021 V). Coverage is marginal only: **74.9 % for low light (G<200)**; and **under range shift it fails completely** (hot-range 5.5 %, high-irradiance 8.0 % at nominal 90 %). So the interval is a valid statement about in-distribution data, not a safety guarantee out of range.

## 4. Honest assessment: is there a genuine ANN advantage?

**Small, conditional, and not where the manuscript claimed it.**
- *Nominal plant:* ANN ≈ 99.996 % vs sensor-less 99.75 % (+0.2 pp), faster (≈1 cycle) and ~30× less voltage ripple. But most of that comes from having G and T sensors: an equal-sensor quadratic regression gets 99.81 %, so the **ML-specific** gain over Poly2 is +0.14 pp mean, concentrated in two scenarios (comprehensive, cloud passing); on real days +0.04 pp.
- *When the plant differs from the training model* (aging/soiling-type mismatch, severity 1) the sensor-less controllers beat the ANN by 0.1–0.4 pp and the gap grows with severity. A model-free tracker cannot be wrong about the plant; a model-based one can.
- *Partial shading:* no advantage — all four lock onto a local peak in 6/9 patterns; the sensor-based ones are 2.4 pp worse.
- *Cost:* ~600× the compute of P&O in this (Python/Keras) implementation, plus irradiance and temperature sensors.
- The v1 claim "P&O ≈ 94 %" is not reproduced; with a correct P&O on this model it is 99.7 %. Manuscript Abstract/Tables must use the numbers above.
- The ANN's main defensible value: fast, ripple-free tracking **when a trustworthy plant model and G/T sensors exist**; it is not better than a polynomial fit of the same data in efficiency by more than a fraction of a percent, and it is worse than a plain P&O when the model is wrong.

## 5. Caveats and judgement calls

- Labels and the nominal plant come from the same physics model (no hardware measurements exist here). The range splits and mismatched plant mitigate, but cannot remove, this; the mismatch perturbation is a declared, arbitrary parameter change.
- Option (b) of the sensor-fairness request was chosen (P&O/IncCond kept sensor-less, as in low-cost hardware) **plus** an equal-sensor baseline; sensor-equipped P&O variants (model-based start-up, step scheduling) were not implemented, and sensor noise/quantisation was not simulated (it would degrade the sensor-based controllers).
- Ambient T2m is the sensor input; cell temperature is internal (NOCT model). The NOCT value, n = 1.2 and Rsh = 400 Ω are assumed typical values, not measured.
- The cold-start protocol A (fixed 15–35 V) turned out degenerate with hot cells (30/35 V are at/above Voc ⇒ P = 0, no gradient ⇒ P&O/IncCond 2/5 never converge). Protocol B (starts at 45–97 % of Voc, all valid) was **added after seeing this**; both are reported, B is the headline.
- Real-day scenarios are hourly data linearly interpolated and time-compressed; irradiance dynamics faster than one hour are absent. Original-bug note (v1): the old P&O/IncCond bookkeeping bug (dV ≡ 0) is still demonstrated in `results/figures/14_legacy_bug.png`.
- Git: PR #1 (v1) was merged; this work sits on the same branch name on top of that history. No new PR was opened.

## 6. Files
`20236359-code-new.ipynb` (executed) · `pv_single_diode.py`, `physics_check.py` · `PVGIS_Ramadi_2005-2020.csv` · `results/` (CSV/JSON tables, 16 figures, trained ANN, `ann_advantage.csv`, `conformal_coverage.csv`, `mismatch_sweep.csv`, …).
