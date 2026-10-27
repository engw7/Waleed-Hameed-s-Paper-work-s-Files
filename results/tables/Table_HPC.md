**Table_HPC.** Where parallel compute and where real-time inference are needed (measured on the machine recorded in results/hpc/hardware.json).

| Part | Need | Measure | Result | Bit-identical |
|---|---|---|---|---|
| Training-data generation (hybrid_n_train = 200,000 plants in the full run) | parallel compute | speed-up / efficiency at 2, 4, 8 workers | 2: 1.62x / 0.81; 4: 2.62x / 0.65; 8: 1.90x / 0.24 | yes |
| Design-space sweeps, multi-seed statistics | parallel compute | speed-up / efficiency at 2, 4, 8 workers (shading suites) | 2: 1.94x / 0.97; 4: 3.82x / 0.95; 8: 3.76x / 0.47 | - |
| Simulation throughput (Hybrid, serial) | - | controller cycles per second incl. plant model | 377 cycles/s | - |
| Online DNN inference (float32, numpy) | real-time, not HPC | median / p99 latency vs 1 s cycle | 19 / 87 us = 1.9e-05 of the cycle | - |
| Online DNN inference (int8, numpy emulation) | real-time, not HPC | median / p99 latency | 69 / 191 us | - |
| Online DNN inference (Keras predict) | real-time, not HPC | median / p99 latency | 64796 / 101017 us = 6.5e-02 of the cycle | - |
| Scan overhead (Hybrid, shading suite, nominal plant) | real-time | share of cycles / share of available energy | 0.42 % / 0.19 % | - |
