# Supplementary material

This document accompanies the manuscript “Isolating the Contribution of Deep Neural Networks in Hybrid Scan-Based MPPT Controllers Under Partial Shading”. It contains the statistical-robustness, design-ablation, boost-converter and sensor-noise/quantization studies summarized in the section “Robustness studies” of the main text. Table and figure numbers without the prefix S refer to the main text.

### Statistical robustness

To separate real differences from seed noise, the scenario suites were repeated with 10 scenario seeds (which vary the cloud-noise stream, the step instant and the cloud phase), and the stage-1 DNN was retrained with five seeds (paired with the scenario seed modulo 5). Table S1 gives the mean ± standard deviation and Table S2 the paired Wilcoxon signed-rank tests; with only ten pairs, a difference can reach p < 0.05 only if its sign is consistent across the seeds, so small but consistent differences are flagged even when they are practically irrelevant. The seed means differ slightly from the single deterministic runs of Table 11 (for example 99.36% against 99.32% for the hybrid on shading).

The tests agree with the headline results. The hybrid beats P&O under partial shading by 18.9 points (p = 0.002, ten of ten seeds) and loses to it on the standard scenarios and real days by 0.14–0.15 points (p = 0.002). The hybrid and Scan+P&O do not differ significantly under partial shading on the nominal plant (median difference +0.016 points, p = 0.23, **Table S2**); on the real days the hybrid is ahead in all ten seeds (p = 0.002), but by only 0.005 points, and on the mismatched plant its shading advantage is 0.055 points (p = 0.002). These differences are consistent but far too small to matter in practice. The DNN scan-only variant is less reliable than the hybrid under shading (98.32 ± 1.00% against 99.36 ± 0.05%; p = 0.027) because an occasional wrong-peak estimate is not corrected without the P&O stage. PSO has a much larger seed-to-seed spread than all other controllers (96.09 ± 2.37% under shading and a success rate of 97.8 ± 7.0%).

**Table S1** Mean ± standard deviation of the tracking efficiency (%) over 10 scenario seeds on the nominal plant, with the stage-1 DNN retrained for five seeds. Standard scenarios, real days and shading as in Table 11; GMPP success is the percentage of shading patterns solved; event convergence in control cycles.

| Controller | Standard scenarios | Real days | Shading | Shading, mismatched plant | GMPP success (%) | Event convergence (cycles) |
| --- | --- | --- | --- | --- | --- | --- |
| P&O | 99.73 ± 0.01 | 99.81 ± 0.01 | 80.43 ± 0.02 | 81.11 ± 0.02 | 33.3 ± 0.0 | 2.27 ± 0.43 |
| IncCond | 99.76 ± 0.01 | 99.83 ± 0.01 | 80.46 ± 0.02 | 81.13 ± 0.02 | 33.3 ± 0.0 | 2.58 ± 0.43 |
| Poly2(G,T)-MPPT | 99.81 ± 0.00 | 99.98 ± 0.01 | 78.17 ± 0.00 | 77.21 ± 0.00 | 33.3 ± 0.0 | 0.95 ± 0.43 |
| ANN-MPPT | 100.00 ± 0.00 | 100.00 ± 0.00 | 78.00 ± 0.00 | 76.99 ± 0.00 | 33.3 ± 0.0 | 0.95 ± 0.43 |
| Scan+P&O (no ML) | 99.59 ± 0.01 | 99.66 ± 0.01 | 99.34 ± 0.04 | 99.35 ± 0.05 | 100.0 ± 0.0 | 2.00 ± 0.50 |
| DNN scan-only (no P&O) | 99.67 ± 0.03 | 99.77 ± 0.02 | 98.32 ± 1.00 | 98.31 ± 0.95 | 100.0 ± 0.0 | 0.99 ± 0.36 |
| Hybrid (DNN+P&O) | 99.59 ± 0.01 | 99.66 ± 0.01 | 99.36 ± 0.05 | 99.40 ± 0.05 | 100.0 ± 0.0 | 1.14 ± 0.34 |
| Tree scan+P&O | 99.59 ± 0.00 | 99.66 ± 0.01 | 99.33 ± 0.06 | 99.36 ± 0.05 | 100.0 ± 0.0 | 1.14 ± 0.34 |
| PSO | 98.04 ± 0.09 | 97.65 ± 0.26 | 96.09 ± 2.37 | 96.30 ± 2.14 | 97.8 ± 7.0 | 34.25 ± 16.00 |

**Table S2** Paired two-sided Wilcoxon signed-rank tests over 10 scenario seeds (a versus b). A positive median difference means that a is more efficient than b; the last column gives the number of seeds in which a is better.

| Plant | Comparison (a vs b) | Metric | Median difference (points) | p | a better in |
| --- | --- | --- | --- | --- | --- |
| Nominal | Hybrid vs Scan+P&O | Standard scenarios | +0.004 | 0.131 | 6/10 |
| Nominal | Hybrid vs Scan+P&O | Real days | +0.005 | 0.002 | 10/10 |
| Nominal | Hybrid vs Scan+P&O | Shading | +0.016 | 0.232 | 6/10 |
| Nominal | Hybrid vs P&O | Standard scenarios | −0.141 | 0.002 | 0/10 |
| Nominal | Hybrid vs P&O | Real days | −0.152 | 0.002 | 0/10 |
| Nominal | Hybrid vs P&O | Shading | +18.907 | 0.002 | 10/10 |
| Nominal | Scan+P&O vs P&O | Shading | +18.905 | 0.002 | 10/10 |
| Nominal | Hybrid vs DNN scan-only | Standard scenarios | −0.081 | 0.004 | 1/10 |
| Nominal | Hybrid vs DNN scan-only | Shading | +1.183 | 0.027 | 7/10 |
| Mismatched | Hybrid vs Scan+P&O | Shading | +0.055 | 0.002 | 10/10 |
| Mismatched | Hybrid vs P&O | Shading | +18.291 | 0.002 | 10/10 |

### Design ablation of the stage-1 estimator

To test whether the network design matters, the stage-1 estimator was replaced by smaller and larger networks, by regularized variants, by a linear regressor on the same 16 inputs, and by a network that sees only G and T (Table S3). The architecture used in the other experiments was chosen a priori and was not tuned on the test years (there was no hyper-parameter search); the training years, the early-stopping years (2015–2016) and the test years (2019–2020) are disjoint. All neural variants with the 16 inputs end up within 0.02 points of each other in closed-loop shading efficiency (99.33–99.35%), although their offline RMSE ranges from 0.75 to 0.94 V and their size from 1,633 to 103,169 parameters; the differences are of the order of the seed-to-seed spread. A network with only 1,633 parameters and 1,568 multiply–accumulate operations therefore performs as well in the closed loop as the 27,009-parameter network. The information content of the scan, not the network capacity, is what matters. A linear regressor on the scan currents already reaches 97.79% (success 88.9%), and a network that sees only G and T, without the scan currents, falls to 80.18% with the success rate of P&O (33.3%). This result shows that the benefit comes from the scan and that the DNN is one of several adequate estimators; a smaller network or a tree-based estimator would do equally well in this setting.

**Table S3** Design ablation of the stage-1 estimator: offline error on the 2019–2020 in-distribution test set and closed-loop results on the nine shading patterns, mean ± standard deviation over the stated number of seeds.

| Variant | Seeds | Parameters | MACs | RMSE in distribution (V) | Wrong peak (%) | Shading eff., nominal (%) | Shading eff., mismatched (%) | GMPP success, nominal (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DNN (32,32) | 3 | 1,633 | 1,568 | 0.939 ± 0.011 | 3.41 ± 0.09 | 99.35 ± 0.02 | 99.39 ± 0.02 | 100.0 |
| DNN (64,64) | 3 | 5,313 | 5,184 | 0.879 ± 0.037 | 2.91 ± 0.18 | 99.35 ± 0.02 | 99.38 ± 0.01 | 100.0 |
| DNN (128,128,64), used in this study | 5 | 27,009 | 26,688 | 0.748 ± 0.026 | 1.90 ± 0.18 | 99.35 ± 0.03 | 99.41 ± 0.03 | 100.0 |
| DNN (256,256,128) | 3 | 103,169 | 102,528 | 0.750 ± 0.075 | 1.84 ± 0.34 | 99.33 ± 0.02 | 99.41 ± 0.01 | 100.0 |
| DNN (128,128,128,64) | 3 | 43,521 | 43,072 | 0.790 ± 0.059 | 1.78 ± 0.35 | 99.33 ± 0.03 | 99.41 ± 0.02 | 100.0 |
| Used network + dropout 0.1 | 3 | 27,009 | 26,688 | 0.746 ± 0.044 | 1.90 ± 0.15 | 99.33 ± 0.04 | 99.41 ± 0.01 | 100.0 |
| Used network + L2 1e-4 | 3 | 27,009 | 26,688 | 0.775 ± 0.021 | 2.11 ± 0.09 | 99.33 ± 0.01 | 99.40 ± 0.01 | 100.0 |
| Ridge regression (16 inputs) | 1 | 17 | 16 | 3.197 | 51.78 | 97.79 | 96.94 | 88.9 |
| Used network, G and T only (no scan currents) | 3 | 25,217 | 24,896 | 4.431 | 77.49 ± 0.09 | 80.18 ± 0.01 | 80.79 ± 0.01 | 33.3 |

### Boost-converter plant

To check that the conclusions do not depend on the quasi-static plant, the controllers were also run on a plant that includes the converter (Fig. 9, Table 7): the PV source feeds an input capacitor and a boost converter to a constant 48 V bus, an inner PI loop regulates the PV voltage to the command of the MPPT algorithm, and the measurement is the mean over the last quarter of a 20 ms settling window (60 ms for command jumps larger than 2 V). **Fig. S1** shows the irradiance step on this plant, with power, PV voltage, duty cycle and inductor current. The averaged continuous-conduction model was compared with a switching-level model (PWM with 100 steps per period, diode with discontinuous-conduction clamp) driven by the same command sequence (Table S4). For the irradiance step and for the shading pattern outside the large scan jumps, the largest deviation is 0.049 V and 0.10% of the rated power. During the 14 large jumps of a scan (probe voltages up to 18 V away from the present voltage), the deviation reaches 1.45 V and 4.7% of the rated power because neither model has settled within the window, so the averaged model should not be trusted for the detailed shape of the scan transients. Both models were implemented in this study and were not compared with a hardware converter.

**Table S4** Averaged versus switching-level boost-converter model driven by the same command sequence: largest deviation of the window-averaged PV voltage and power (percent of the rated power). Large jump: the command differs by more than 2 V from the PV voltage at the start of the window.

| Case | Large jump | Calls | Max. deviation of V (V) | Max. deviation of P (W) | Max. deviation of P (% of rated) | Mean deviation of P (% of rated) |
| --- | --- | --- | --- | --- | --- | --- |
| Irradiance step (P&O) | no | 30 | 0.049 | 0.25 | 0.102 | 0.015 |
| Shading pattern 1 including scan (Hybrid) | no | 29 | 0.024 | 0.06 | 0.025 | 0.007 |
| Shading pattern 1 including scan (Hybrid) | yes | 14 | 1.447 | 11.54 | 4.718 | 1.179 |

The converter dynamics change the efficiency of the controllers by little (Table S5). The largest change is −0.58 points, for the plain ANN under partial shading; P&O and IncCond change by −0.13 and −0.14 points under shading and by −0.09 and −0.10 points on the comprehensive scenario, and the hybrid by −0.10 points on the comprehensive scenario and −0.004 points under shading. On the step, cloud and ramp scenarios no controller changes by more than 0.06 points. The ranking of the controllers is unchanged.

**Table S5** Tracking efficiency (%) on the quasi-static plant and on the averaged boost-converter plant (nominal PV model) for the comprehensive scenario and for partial shading.

| Scenario | Controller | Quasi-static | With converter | Change (points) |
| --- | --- | --- | --- | --- |
| Comprehensive | P&O | 99.54 | 99.45 | −0.09 |
| Comprehensive | IncCond | 99.57 | 99.46 | −0.10 |
| Comprehensive | Scan+P&O (no ML) | 99.35 | 99.32 | −0.03 |
| Comprehensive | ANN-MPPT | 99.99 | 99.99 | −0.005 |
| Comprehensive | Hybrid (DNN+P&O) | 99.40 | 99.31 | −0.10 |
| Partial shading | P&O | 80.41 | 80.28 | −0.13 |
| Partial shading | IncCond | 80.44 | 80.30 | −0.14 |
| Partial shading | Scan+P&O (no ML) | 99.36 | 99.36 | −0.007 |
| Partial shading | ANN-MPPT | 77.99 | 77.40 | −0.58 |
| Partial shading | Hybrid (DNN+P&O) | 99.32 | 99.32 | −0.004 |

&#91;image: Fig. 33\]

**Fig. S1** Irradiance step on the averaged boost-converter plant: power, PV voltage, duty cycle and inductor current; left, full run; right, zoom around the step.



### Sensor noise and numerical precision

To test sensitivity to measurement errors, Gaussian noise was added to the measured currents (0.5%), voltages (0.5%) and irradiance input (2%) on the nine shading patterns, and the efficiency was scored on the true values (Table S6). The scan-based controllers still solve every pattern, but their efficiency drops by about 0.6 points (hybrid 99.32% to 98.70%, Scan+P&O 99.36% to 98.75%, tree-based variant 99.37% to 98.67%). The DNN scan-only variant, which has no P&O stage, is unaffected (99.30% to 99.31%), and the sensor-less trackers change little (P&O 80.41% to 80.25%), so the loss comes from P&O dithering on noisy power readings and not from the network. PSO loses its perfect success rate (97.8%).

**Table S6** Nine shading patterns, nominal plant, with sensor noise σ\_I = 0.5%, σ\_V = 0.5% and σ\_G = 2%; noisy values are mean ± standard deviation over noise seeds.

| Controller | Efficiency, clean (%) | Efficiency, noisy (%) | GMPP success, clean (%) | GMPP success, noisy (%) |
| --- | --- | --- | --- | --- |
| P&O | 80.41 | 80.25 ± 0.01 | 33.3 | 33.3 |
| IncCond | 80.44 | 80.25 ± 0.01 | 33.3 | 33.3 |
| Poly2(G,T)-MPPT | 78.17 | 78.16 ± 0.00 | 33.3 | 33.3 |
| ANN-MPPT | 77.99 | 77.99 ± 0.00 | 33.3 | 33.3 |
| Scan+P&O (no ML) | 99.36 | 98.75 ± 0.12 | 100 | 100 |
| DNN scan-only (no P&O) | 99.31 | 99.31 ± 0.18 | 100 | 100 |
| Hybrid (DNN+P&O) | 99.33 | 98.70 ± 0.19 | 100 | 100 |
| Tree scan+P&O | 99.37 | 98.67 ± 0.13 | 100 | 100 |
| PSO | 95.68 | 95.79 ± 0.16 | 100 | 97.8 |

Storing the weights in half precision halves the model size (108,036 to 54,660 bytes) and leaves the estimate essentially unchanged (largest deviation from float32 0.049 V, no change in closed-loop efficiency; Table S7). Post-training int8 quantization (emulated in numpy, not run on a microcontroller) gives 27,972 bytes but raises the RMSE from 0.78 to 1.98 V and the wrong-peak rate from 2.1% to 7.7%, with single outliers of up to 22 V; the closed-loop shading efficiency nevertheless stays at 99.32%, because the P&O stage corrects the hand-off. A naive int8 conversion is therefore not recommended without calibration or quantization-aware training, which was not studied here.

**Table S7** Post-training quantization of the stage-1 DNN (emulated in numpy). The latency is that of the emulation on the benchmark machine and is not a microcontroller measurement.

| Precision | Size (bytes) | RMSE (V) | Wrong peak (%) | Max. deviation from float32 (V) | Shading eff., nominal (%) | Shading eff., mismatched (%) | GMPP success (%) | Latency, emulation (µs) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| float32 | 108,036 | 0.784 | 2.11 | 0 | 99.33 | 99.37 | 100 | 19.3 |
| float16 | 54,660 | 0.784 | 2.12 | 0.049 | 99.33 | 99.37 | 100 | 18.6 |
| int8 | 27,972 | 1.976 | 7.68 | 22.08 | 99.32 | 99.38 | 100 | 68.9 |

