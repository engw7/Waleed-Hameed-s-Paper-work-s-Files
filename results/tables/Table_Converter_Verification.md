**Table_Converter_Verification.** Averaged versus switching-level model driven by the same command sequence: largest deviation of the window-averaged PV voltage and power (percent of the rated power). large_jump = command differs by more than 2 V from the PV voltage at the start of the window (scan probes, start-up): the averaged model and the switching model are both still settling there.

| case | large_jump | calls | max_dV_V | max_dP_W | max_dP_rel_pct | mean_dP_rel_pct | max_jump_V | max_ripple_i_A | max_ripple_v_V |
|---|---|---|---|---|---|---|---|---|---|
| irradiance step (P&O) | no | 30 | 0.049 | 0.25 | 0.102 | 0.015 | 0.596 | 2.997 | 1.822 |
| shading pattern 1 incl. scan (Hybrid) | no | 29 | 0.024 | 0.061 | 0.025 | 0.007 | 0.633 | 2.837 | 6.969 |
| shading pattern 1 incl. scan (Hybrid) | yes | 14 | 1.447 | 11.541 | 4.718 | 1.179 | 18 | 17.539 | 26 |
