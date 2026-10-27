**Table_Converter_vs_Static.** Tracking efficiency on the quasi-static plant versus the averaged boost-converter plant (nominal PV model).

| scenario | controller | Eff. static (%) | Eff. converter (%) | Delta (points) |
|---|---|---|---|---|
| comprehensive | P&O | 99.54 | 99.45 | -0.09 |
| comprehensive | IncCond | 99.57 | 99.46 | -0.1 |
| comprehensive | Scan+P&O (no ML) | 99.35 | 99.32 | -0.03 |
| comprehensive | ANN-MPPT | 99.99 | 99.99 | -0.0045 |
| comprehensive | Hybrid (DNN+P&O) | 99.4 | 99.31 | -0.1 |
| step_change | P&O | 99.81 | 99.8 | -0.0065 |
| step_change | IncCond | 99.89 | 99.88 | -0.0064 |
| step_change | Scan+P&O (no ML) | 99.67 | 99.67 | -0.0076 |
| step_change | ANN-MPPT | 100 | 100 | -0.00024 |
| step_change | Hybrid (DNN+P&O) | 99.61 | 99.67 | 0.06 |
| cloud_passing | P&O | 99.77 | 99.77 | -0.003 |
| cloud_passing | IncCond | 99.78 | 99.78 | -0.0015 |
| cloud_passing | Scan+P&O (no ML) | 99.62 | 99.62 | -0.0026 |
| cloud_passing | ANN-MPPT | 99.99 | 99.99 | -0.00024 |
| cloud_passing | Hybrid (DNN+P&O) | 99.63 | 99.63 | -0.001 |
| ramp_change | P&O | 99.81 | 99.8 | -0.0095 |
| ramp_change | IncCond | 99.83 | 99.82 | -0.0089 |
| ramp_change | Scan+P&O (no ML) | 99.66 | 99.65 | -0.01 |
| ramp_change | ANN-MPPT | 100 | 100 | -0.00026 |
| ramp_change | Hybrid (DNN+P&O) | 99.65 | 99.66 | 0.0043 |
| partial_shading | P&O | 80.41 | 80.28 | -0.13 |
| partial_shading | IncCond | 80.44 | 80.3 | -0.14 |
| partial_shading | Scan+P&O (no ML) | 99.36 | 99.36 | -0.0071 |
| partial_shading | ANN-MPPT | 77.99 | 77.4 | -0.58 |
| partial_shading | Hybrid (DNN+P&O) | 99.32 | 99.32 | -0.0041 |
