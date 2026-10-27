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
