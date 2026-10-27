# Figure captions (generated from the result files)

**Fig. 01.** System overview: PV module or shaded string, input capacitor, boost converter to a constant bus, voltage/current sensing, MPPT controller (command V_cmd) and inner voltage loop (duty D).

**Fig. 02.** Bypass-diode shaded-string model: three substrings of 18 cells with bypass diodes; shaded cells receive a fraction of the irradiance.

**Fig. 03.** Block diagram of the Hybrid controller: I-V scan, DNN estimate of V_gmpp, P&O refinement and re-scan logic.

**Fig. 04.** Stage-1 DNN data flow: input vector (G_in, T, I_1..I_14), standardisation, hidden layers 128-128-64, output V_est; parameters 27,009.

**Fig. 05.** Flowchart of the P&O algorithm exactly as implemented (threshold 0.01 W, step 0.5 V).

**Fig. 06.** Flowchart of the incremental-conductance algorithm with explicit handling of dV = 0 before dI/dV is evaluated.

**Fig. 07.** State diagram of the Hybrid controller: SCAN and TRACK states, hand-off and re-scan triggers.

**Fig. 08.** Flowchart of the PSO global-MPP tracker used as comparator.

**Fig. 10.** Histograms of hourly irradiance (a: all hours, log scale; b: daytime hours) and ambient temperature (c), Ramadi 2005–2020.

**Fig. 11.** Daytime (G, T) operating conditions coloured by month (every fifth hour plotted).

**Fig. 12.** Mean irradiance by month and hour of day.

**Fig. 13.** Representative winter (a) and summer (b) days of the test years: irradiance and ambient temperature.

**Fig. 14.** P-V curves of the single-diode model at cell temperatures 25, 45 and 65 °C (a–c) for irradiances 200–1000 W/m²; dots mark the maxima.

**Fig. 15.** Maximum-power voltage versus irradiance at fixed cell temperature (a) and versus cell temperature at fixed irradiance (b).

**Fig. 16.** Training and validation loss of the stage-1 DNN (headline seed).

**Fig. 17.** Stage-1 DNN on the in-distribution test set: estimated vs true V_gmpp (a) and error histogram (b); RMSE 0.78 V.

**Fig. 18.** Empirical coverage of split-conformal intervals versus nominal confidence for four test sets; intervals are valid only in distribution (T1) and under-cover for out-of-range plants (T4).

**Fig. 19.** Irradiance step 400 → 800 W/m² (nominal plant): power with true MPP (dashed) and 98–100 % band, PV voltage and duty cycle; left full run, right zoom. Duty = 1 - V_cmd/V_bus (display).

**Fig. 20.** Cloud-passing scenario (nominal plant); layout as in Fig. 19.

**Fig. 21.** Ramp scenario (nominal plant); layout as in Fig. 19.

**Fig. 22.** Nine partial-shading patterns (nominal plant); triangles mark scan cycles of the scan-based controllers in the zoom.

**Fig. 23.** Cold start from the lowest (left) and highest (right) valid initial voltage at G = 800 W/m², T = 25 °C; scan cycles show the mean scan power.

**Fig. 24.** P-V curves of the nine shading patterns with the operating point of every controller at the end of its episode; crosses mark the global MPP.

**Fig. 25.** Tracking efficiency per scenario, mean ± std over seeds, nominal (a) and mismatched (b) plant; the y-axis starts at 70 % (axis-break marks).

**Fig. 26.** Event convergence time (a) and steady-state voltage oscillation (b), mean ± std over seeds, nominal plant.

**Fig. 27.** Stage-1 design ablation: test RMSE of V_gmpp (a) and closed-loop shading efficiency (b), mean ± std over seeds.

**Fig. 28.** Parallel speed-up of training-data generation and of the scenario-seed sweep (a) and DNN inference latency per precision/runtime against the 1 s control cycle (b).

**Fig. 30.** Irradiance step on the averaged boost-converter plant: power, PV voltage, duty cycle and inductor current; left full run, right zoom.

**Fig. 31.** Averaged versus switching-level model after the irradiance step: PV voltage (top) and inductor current (bottom), full window (left) and last millisecond (right).

**Fig. 32.** Tracking efficiency mean ± std over scenario seeds for standard scenarios (a), real days (b) and partial shading (c), nominal and mismatched plant.

**Fig. 33.** Shading efficiency with and without sensor noise (I 0.5 %, V 0.5 %, G 2 %).

**Fig. 34.** Model size (a) and largest deviation of the estimated voltage from float32 (b) for float32, float16 and int8 (numpy emulation).

**Fig. 35.** Mean controller decision time per cycle (logarithmic axis); plant model excluded.

**Fig. 36.** Tracking efficiency versus plant-mismatch severity for the comprehensive (a) and partial-shading (b) scenarios.

**Fig. 37.** Comprehensive scenario (nominal plant); layout as in Fig. 19.

**Fig. 38.** Real clear held-out day (nominal plant); layout as in Fig. 19.

**Fig. 39.** Real variable held-out day (nominal plant); layout as in Fig. 19.

**Fig. 40.** Comprehensive scenario, mismatched plant (severity 1); layout as in Fig. 19.

**Fig. 41.** Partial shading, mismatched plant (severity 1); layout as in Fig. 19.

**Fig. 42.** Real clear day, mismatched plant (severity 1); layout as in Fig. 19.

**Fig. 43.** Sanity check of the module power computed from PVGIS irradiance: power versus irradiance (a) and a three-day series (b).

**Fig. 44.** Single-diode model at STC against the datasheet-type inputs: I-V (a) and P-V (b).
