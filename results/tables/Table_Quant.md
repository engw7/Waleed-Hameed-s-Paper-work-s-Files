**Table_Quant.** Post-training quantisation of the stage-1 DNN (emulated in numpy). Latencies are those of the emulation on the benchmark machine, not MCU measurements.

| precision | Size (bytes) | RMSE V_gmpp (V) | Wrong-peak (%) | Max dev. vs float32 (V) | Shading eff. nominal (%) | Shading eff. mismatched (%) | GMPP success (%) | Latency, numpy emulation (us) |
|---|---|---|---|---|---|---|---|---|
| float32 | 108036 | 0.784 | 2.113 | 0 | 99.325 | 99.372 | 100 | 19.312 |
| float16 | 54660 | 0.784 | 2.119 | 0.049 | 99.325 | 99.371 | 100 | 18.587 |
| int8 | 27972 | 1.976 | 7.681 | 22.083 | 99.323 | 99.379 | 100 | 68.892 |
