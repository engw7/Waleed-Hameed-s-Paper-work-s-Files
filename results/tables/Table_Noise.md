**Table_Noise.** Nine shading patterns, nominal plant, sensor noise sigma_I = 0.5%, sigma_V = 0.5%, sigma_G = 2% (noisy values are mean ± std over noise seeds; scoring uses true values).

| Controller | Eff. clean (%) | Eff. noisy (%) | GMPP success clean (%) | GMPP success noisy (%) |
|---|---|---|---|---|
| ANN-MPPT | 77.989 | 77.99 ± 0.00 | 33.333 | 33.333 |
| DNN scan-only (no P&O) | 99.305 | 99.31 ± 0.18 | 100 | 100 |
| Hybrid (DNN+P&O) | 99.325 | 98.70 ± 0.19 | 100 | 100 |
| IncCond | 80.438 | 80.25 ± 0.01 | 33.333 | 33.333 |
| P&O | 80.408 | 80.25 ± 0.01 | 33.333 | 33.333 |
| PSO | 95.678 | 95.79 ± 0.16 | 100 | 97.778 |
| Poly2(G,T)-MPPT | 78.165 | 78.16 ± 0.00 | 33.333 | 33.333 |
| Scan+P&O (no ML) | 99.363 | 98.75 ± 0.12 | 100 | 100 |
| Tree scan+P&O | 99.366 | 98.67 ± 0.13 | 100 | 100 |
