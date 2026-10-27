"""Static/descriptive tables: PV module datasheet + extracted model parameters, algorithm parameters, metric definitions, dataset excerpt (12 PVGIS rows + the 14 scan currents of a
shaded sample), and the literature comparison skeleton (cells stay 'not reported' unless the paper was actually read - see results/tables/Table_Literature)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, pandas as pd


def run():
    from mppt.config import CFG, RESULTS
    from mppt.data import get_pv_series
    from mppt import models as MD
    from mppt.tables import save_table
    from mppt.controllers import PSOMPPT, PerturbAndObserveMPPT, IncrementalConductanceMPPT
    import metrics as M, pv_hybrid_data as hd
    pv, pvm, meta = get_pv_series(); ds = pvm.ds; p = pvm.p
    v, pm, im = pvm.mpp(1000.0, 25.0 - (ds['NOCT'] - 20) / 800 * 1000.0)
    rows = [('Module', 'Generic 60-cell-class module, 54 series cells (single-diode model; not a specific commercial product)', 'model setting'),
            ('Cells in series N_s', ds['Ns'], 'datasheet-type input'), ('Short-circuit current I_sc,ref (A)', ds['Isc'], 'datasheet-type input (STC)'), ('Open-circuit voltage V_oc,ref (V)', ds['Voc'], 'datasheet-type input (STC)'),
            ('Maximum power P_max,ref (W)', ds['Pmax'], 'datasheet-type input (STC)'), ('I_sc temperature coefficient (A/K)', ds['mu_isc'], 'datasheet-type input'), ('NOCT (°C)', ds['NOCT'], 'datasheet-type input'),
            ('Ideality factor n', p['n'], 'fixed'), ('Shunt resistance R_sh,ref (ohm)', p['Rsh_ref'], 'fixed'), ('Photo-current I_ph,ref (A)', p['Iph_ref'], 'extracted'), ('Saturation current I_0,ref (A)', p['I0_ref'], 'extracted'),
            ('Series resistance R_s (ohm)', p['Rs'], 'extracted: solved so that the model P_max equals the datasheet P_max'), ('Model V_mpp at STC (V)', float(v), 'model output'), ('Model I_mpp at STC (A)', float(im), 'model output'), ('Model P_max at STC (W)', float(pm), 'model output')]
    save_table('Table_PV_Datasheet', pd.DataFrame(rows, columns=['Quantity', 'Value', 'Origin']), 'PV module data and extracted single-diode parameters.', digits=5)
    po, ic = PerturbAndObserveMPPT(), IncrementalConductanceMPPT()
    alg = [('All', 'control cycle', '1 s'), ('All', 'converter voltage window', f"[{CFG['v_min_op']:g}, {CFG['v_max_op']:g}] V"), ('All', 'start voltage', f"{CFG['v_start']:g} V"), ('P&O / IncCond / scan hand-off', 'perturbation step', f"{CFG['step_size']:g} V per cycle"),
           ('P&O', 'minimum power change', '0.01 W'), ('IncCond', 'voltage tolerance / conductance tolerance', f'{ic.voltage_tolerance:g} V / {ic.conductance_tolerance:g}'),
           ('Poly2(G,T)', 'model', 'quadratic polynomial of (G,T) with ridge 1e-6, 6 coefficients'), ('ANN-MPPT', 'network / inputs / output', f"{CFG['nn_hidden']} hidden units, inputs (G,T,V,I), output V_ref; slew {CFG['ref_slew']:g} V per cycle"),
           ('Scan-based', 'probe voltages', ', '.join(f'{x:.2f}' for x in hd.PROBES) + ' V (14 points)'), ('Scan-based', 're-scan trigger', f"P/G below {1 - CFG['scan_trigger']:.0%} of the running maximum since the last scan (at least {CFG['scan_min_interval']} cycles after it) or every {CFG['scan_period']} cycles"),
           ('Hybrid / DNN scan-only', 'stage-1 DNN', f"inputs (G_in, T, I_1..I_14), hidden {CFG['hybrid_hidden']}, ReLU, output V_gmpp; Adam 1e-3, batch {CFG['hybrid_batch']}, up to {CFG['hybrid_epochs']} epochs, early stopping patience {CFG['hybrid_patience']} on 2015-16"),
           ('PSO', 'swarm', f'{PSOMPPT.N} particles, w = {PSOMPPT.W_IN}, c1 = c2 = {PSOMPPT.C1}; converged when all particles within {PSOMPPT.CONV_V} V; restart when |dP|/P > {PSOMPPT.RESTART:.0%} or after {PSOMPPT.MAX_HOLD} cycles (fixed a priori)'),
           ('Tree scan+P&O', 'estimator', 'histogram gradient-boosted trees, 31 leaves, learning rate 0.1, iteration count chosen on 2015-16'), ('Metrics', 'convergence / GMPP success / wrong peak', f'{M.CONV_TOL:.0%} for {M.CONV_HOLD} cycles / {M.GMPP_TOL:.0%} over the last {M.GMPP_TAIL} non-scan cycles / > {M.WRONG_PEAK_V:g} V')]
    save_table('Table_Algorithm_Parameters', pd.DataFrame(alg, columns=['Controller', 'Parameter', 'Value']), 'Algorithm parameters (all fixed a priori; none tuned on 2019–2020).')
    day = pv[(pv.year == 2019) & (pv.month == 6) & (pv.day == 24) & (pv['G(i)'] > 0)] if 'day' in pv else pv[(pv.year == 2019) & (pv.month == 6) & (pv.t.dt.day == 24) & (pv['G(i)'] > 0)]
    ex = day.iloc[:12][['t', 'G(i)', 'T2m']].copy(); ex['t'] = ex.t.dt.strftime('%Y-%m-%d %H:%M'); Tc = pvm.cell_temperature(ex['G(i)'].to_numpy(), ex['T2m'].to_numpy()); vm = [pvm.mpp(g, t)[0] for g, t in zip(ex['G(i)'], ex['T2m'])]; pmx = [pvm.mpp(g, t)[1] for g, t in zip(ex['G(i)'], ex['T2m'])]
    ex['T_cell'] = Tc; ex['V_mpp'] = vm; ex['P_mpp'] = pmx; ex.columns = ['time (UTC)', 'G (W/m2)', 'T_2m (degC)', 'T_cell (degC)', 'V_mpp (V)', 'P_mpp (W)']
    save_table('Table_Dataset_Excerpt_PVGIS', ex, 'Excerpt of the PVGIS-SARAH3 hourly series (Ramadi, 24 June 2019, first 12 non-zero hours) and the model outputs computed from it.', digits=2)
    data, _ = MD.hybrid_data(pv); d = data['T3']; k = int(np.argmax(d['shaded'])) if 'shaded' in d else 0
    sc = pd.DataFrame({'probe V_k (V)': hd.PROBES, 'I_k (A)': d['I_probe'][k]}); sc.loc[len(sc)] = [np.nan, np.nan]
    sc = sc.iloc[:-1]; sc['sample'] = f"G_in = {d['G_in'][k]:.0f} W/m2, T = {d['T'][k]:.1f} degC, V_gmpp = {d['V_gmpp'][k]:.2f} V" ; sc.loc[1:, 'sample'] = ''
    save_table('Table_Dataset_Excerpt_Scan', sc, 'One partially shaded training-type sample (nominal plant, test-year conditions): the 14 scan currents that form the DNN input together with G_in and T.', digits=3)
    lit = pd.DataFrame([{'Study': s, 'Method': 'not reported', 'Shading handled': 'not reported', 'Compared with': 'not reported', 'Tracking efficiency': 'not reported', 'Convergence': 'not reported', 'DOI': 'not read in this study'} for s in (
        'Bouchetob and Nadji (2022)', 'Abouzeid et al. (2024)', 'Gugulothu and Naik (2023)', 'Shyam and Ghosh (2023)', 'Sorensen and Mihet-Popa (2024)')])
    save_table('Table_Literature', lit, 'Comparison with existing studies: left unfilled on purpose - cells are filled only from papers that were actually read (none were read for this table).')
    print('static tables written')


if __name__ == '__main__':
    run()
