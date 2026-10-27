"""Numerical check of the corrected PV model (step 1): V_mpp must depend on irradiance AND temperature."""
import numpy as np
from pv_single_diode import SingleDiodePV

m = SingleDiodePV()
amb_for_cell = lambda G, Tc: Tc - (m.ds['NOCT'] - 20) / 800 * G       # ambient temp that gives a chosen cell temp
print('extracted parameters:', {k: float(v) for k, v in m.p.items()})
v, p, i = m.mpp(1000, amb_for_cell(1000, 25))
print(f'STC: Voc={float(m.voc(1000, 25.0)):.3f} V  Isc={float(m.current(1000, 25.0, 0.0)):.3f} A  Vmpp={v:.3f} V  Pmax={p:.2f} W (datasheet 244.6 W)')
print(f'emergent dVoc/dT = {float((m.voc(1000, 26.0) - m.voc(1000, 25.0)) / m.voc(1000, 25.0) * 100):.3f} %/K   '
      f'dPmax/dT = {(m.mpp(1000, amb_for_cell(1000, 45))[1] / p - 1) / 20 * 100:.3f} %/K   (typical Si: -0.30..-0.35 / -0.35..-0.45)')
print('\nVmpp vs irradiance at FIXED CELL temperature 25 C (intrinsic, logarithmic dependence):')
base = m.mpp(1000, amb_for_cell(1000, 25))[0]
for G in (100, 200, 400, 600, 800, 1000, 1200):
    vv = m.mpp(G, amb_for_cell(G, 25))[0]
    print(f'  G={G:5d} W/m2  Vmpp={vv:6.3f} V  ({100 * (vv / base - 1):+5.1f} % vs STC)')
print('\nVmpp vs irradiance at FIXED AMBIENT temperature 25 C (cell heating via NOCT adds a larger, opposite-sign effect):')
for G in (100, 200, 400, 600, 800, 1000, 1200):
    print(f'  G={G:5d} W/m2  Vmpp={m.mpp(G, 25.0)[0]:6.3f} V')
print('\nVmpp vs cell temperature at G=1000:')
for Tc in (0, 15, 25, 40, 60):
    print(f'  Tc={Tc:3d} C  Vmpp={m.mpp(1000, amb_for_cell(1000, Tc))[0]:6.3f} V')
# the old Cell-4 model, for contrast: V_mpp is exactly independent of G
import importlib, types
old_V = lambda G, T: None
