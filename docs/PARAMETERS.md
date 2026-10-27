# Parameters and experimental design

## Data and splits

- Data: PVGIS-SARAH3 hourly series, Ramadi, 2005–2020 (`PVGIS_Ramadi_2005-2020.csv`); daytime threshold G >= 50 W/m².
- Training years <= 2014; early stopping / design decisions 2015–2016; conformal calibration 2017–2018; reporting 2019–2020 only.
- Stage-1 DNN training set: 200,000 domain-randomised plant samples (validation 15,000, calibration 15,000, tests 8,000–16,000).
- Seeds: headline DNN seed 42; DNN seeds [0, 1, 2, 3, 4]; scenario seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; noise seeds 0–4. QUICK mode: False.

## Scenarios (1 control cycle = 1 s)

- comprehensive, step_change (400 → 800 W/m²), cloud_passing, ramp_change: 3600 cycles each (seed = None reproduces the frozen scenarios; seed k varies cloud-noise stream, step instant, cloud phase).
- partial_shading: nine patterns of three bypass-diode substrings, 400 s each: [[800, 800, 800], [800, 800, 400], [800, 800, 200], [800, 500, 500], [800, 600, 300], [1000, 1000, 300], [900, 700, 300], [700, 700, 700], [1000, 600, 200]]; the controller sees the mean substring irradiance.
- real_clear_day / real_variable_day: held-out PVGIS days (ranking rules fixed a priori), hourly data time-compressed.
- Mismatched plant: parameters perturbed by severity s (I_ph x(1-0.08 s), R_s x(1+0.8 s), R_sh x(1-0.4 s), n x(1+0.05 s)); headline severity 1.
- Sensor noise (noise experiment only): sigma_I = 0.5%, sigma_V = 0.5%, sigma_G = 2%.

## Tables of parameters

See `results/tables/Table_PV_Datasheet.md`, `Table_Algorithm_Parameters.md`, `Table_Converter_Parameters.md`.

**Table_PV_Datasheet.** PV module data and extracted single-diode parameters.

| Quantity | Value | Origin |
|---|---|---|
| Module | Generic 60-cell-class module, 54 series cells (single-diode model; not a specific commercial product) | model setting |
| Cells in series N_s | 54 | datasheet-type input |
| Short-circuit current I_sc,ref (A) | 9.87 | datasheet-type input (STC) |
| Open-circuit voltage V_oc,ref (V) | 32.9 | datasheet-type input (STC) |
| Maximum power P_max,ref (W) | 244.6 | datasheet-type input (STC) |
| I_sc temperature coefficient (A/K) | 0.00592 | datasheet-type input |
| NOCT (°C) | 45 | datasheet-type input |
| Ideality factor n | 1.2 | fixed |
| Shunt resistance R_sh,ref (ohm) | 400 | fixed |
| Photo-current I_ph,ref (A) | 9.87449 | extracted |
| Saturation current I_0,ref (A) | 2.5627e-08 | extracted |
| Series resistance R_s (ohm) | 0.18203 | extracted: solved so that the model P_max equals the datasheet P_max |
| Model V_mpp at STC (V) | 26.60436 | model output |
| Model I_mpp at STC (A) | 9.19398 | model output |
| Model P_max at STC (W) | 244.6 | model output |


**Table_Algorithm_Parameters.** Algorithm parameters (all fixed a priori; none tuned on 2019–2020).

| Controller | Parameter | Value |
|---|---|---|
| All | control cycle | 1 s |
| All | converter voltage window | [8, 35] V |
| All | start voltage | 26 V |
| P&O / IncCond / scan hand-off | perturbation step | 0.5 V per cycle |
| P&O | minimum power change | 0.01 W |
| IncCond | voltage tolerance / conductance tolerance | 0.1 V / 0.001 |
| Poly2(G,T) | model | quadratic polynomial of (G,T) with ridge 1e-6, 6 coefficients |
| ANN-MPPT | network / inputs / output | (64, 64, 32) hidden units, inputs (G,T,V,I), output V_ref; slew 0.5 V per cycle |
| Scan-based | probe voltages | 8.00, 10.08, 12.15, 14.23, 16.31, 18.38, 20.46, 22.54, 24.62, 26.69, 28.77, 30.85, 32.92, 35.00 V (14 points) |
| Scan-based | re-scan trigger | P/G below 90% of the running maximum since the last scan (at least 30 cycles after it) or every 300 cycles |
| Hybrid / DNN scan-only | stage-1 DNN | inputs (G_in, T, I_1..I_14), hidden (128, 128, 64), ReLU, output V_gmpp; Adam 1e-3, batch 1024, up to 60 epochs, early stopping patience 8 on 2015-16 |
| PSO | swarm | 4 particles, w = 0.4, c1 = c2 = 1.5; converged when all particles within 0.3 V; restart when \|dP\|/P > 10% or after 600 cycles (fixed a priori) |
| Tree scan+P&O | estimator | histogram gradient-boosted trees, 31 leaves, learning rate 0.1, iteration count chosen on 2015-16 |
| Metrics | convergence / GMPP success / wrong peak | 2% for 10 cycles / 2% over the last 10 non-scan cycles / > 2 V |


**Table_Converter_Parameters.** Boost-converter parameters (averaged and switching-level models).

| parameter | value | note |
|---|---|---|
| L | 0.001 | chosen so that the ripple at the nominal operating point (V_pv = 26 V, d = 0.46, f_sw = 20 kHz) is dI = V_pv d / (f_sw L) = 0.6 A (about 7 % of 9 A); CCM holds above 0.3 A |
| R_L | 0.02 | inductor series resistance |
| C_in | 0.0005 | input ripple dV = dI / (8 f_sw C_in) = 0.08 V; also fixes the input-capacitor time constant of the voltage loop |
| f_sw | 20000 | typical for a 250 W module-level converter |
| V_bus | 48 | constant 48 V bus (battery or regulated DC link), the same value used for the display-only duty of the quasi-static results |
| V_f | 0.5 | diode forward drop |
| R_on | 0.015 | switch on-resistance |
| Kp | 0.05 | PI proportional gain [1/V], fixed once on the step response at G = 800 W/m2, T = 25 C (tools in this module), before any scenario was simulated |
| Ki | 10 | PI integral gain [1/(V s)] |
| D_MIN | 0.05 |  |
| D_MAX | 0.95 |  |
| W_MS | 20 | settling window per measurement [ms] (3x for command jumps larger than 2 V); one MPPT control cycle (1 s) holds at least one window; a 14-point scan needs at most 14 x 60 ms = 0.84 s |
| DT_AVG | 2e-05 | integration step of the averaged model [s] |
| STEPS_PER_PERIOD | 100 | switching-level steps per PWM period |

