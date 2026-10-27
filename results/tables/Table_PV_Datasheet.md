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
