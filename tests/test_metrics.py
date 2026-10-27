import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import metrics as M


def test_tracking_efficiency_energy_weighted():
    P_mpp = np.array([100.0, 200.0, 100.0, 200.0]); P = np.array([100.0, 100.0, 100.0, 200.0])
    assert abs(M.tracking_efficiency(P, P_mpp) - 100 * 500 / 600) < 1e-12
    assert M.tracking_efficiency(P_mpp, P_mpp) == 100.0
    t = np.array([0.0, 1.0, 3.0, 4.0])                      # non-uniform time axis weights samples by dt (last dt repeats the previous one)
    assert abs(M.tracking_efficiency(P, P_mpp, t) - 100 * (100 * 1 + 100 * 2 + 100 * 1 + 200 * 1) / (100 * 1 + 200 * 2 + 100 * 1 + 200 * 1)) < 1e-9


def test_convergence_integer_and_nan():
    eta = np.array([50, 60, 97, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 50, 50.0])
    assert M.convergence_cycles(eta, [0], hold=10) == [3.0]         # first cycle that starts a 10-cycle run >= 98 %
    assert np.isnan(M.convergence_cycles(np.full(30, 90.0), [0])[0])
    assert np.isnan(M.convergence_cycles(np.full(5, 100.0), [0])[0])  # shorter than the hold window
    r = M.convergence_cycles(np.r_[np.full(10, 100.0), np.full(10, 50.0), np.full(10, 100.0)], [0, 10, 20])   # events delimit windows
    assert r[0] == 0.0 and np.isnan(r[1]) and r[2] == 0.0


def test_oscillation_square_wave_and_scan_exclusion():
    n = 200; V = np.where(np.arange(n) % 2 == 0, 24.0, 24.5); eta = np.where(np.arange(n) % 2 == 0, 99.0, 100.0)
    ov, op, nw = M.oscillation(V, eta, np.full(n, 24.2), np.zeros(n), [0])
    assert abs(ov - 0.5) < 1e-12 and abs(op - 1.0) < 1e-12 and nw == (n - 50) // 10
    scan = np.zeros(n); scan[100] = 1                           # a scan cycle disqualifies exactly its window
    assert M.oscillation(V, eta, np.full(n, 24.2), scan, [0])[2] == nw - 1
    assert np.isnan(M.oscillation(V, eta, np.linspace(20, 30, n), np.zeros(n), [0])[0])   # V_mpp moving too fast: no quasi-steady window


def test_gmpp_success_tail_and_scan():
    P_mpp = np.full(40, 100.0); P = np.r_[np.full(20, 80.0), np.full(20, 99.0)]
    rate, flags = M.gmpp_success(P, P_mpp, np.zeros(40), [0, 20])
    assert flags == [False, True] and rate == 50.0
    scan = np.zeros(40, int); scan[19] = 1; P2 = P.copy(); P2[19] = 0.0                 # the scan cycle at the end of pattern 1 is ignored
    assert M.gmpp_success(P2, P_mpp, scan, [0, 20])[1] == [False, True]
    edge = np.r_[np.full(10, 98.0)]                              # exactly 2 % below passes, 2.5 % fails
    assert M.gmpp_success(edge, np.full(10, 100.0), np.zeros(10), [0])[1] == [True]
    assert M.gmpp_success(np.full(10, 97.5), np.full(10, 100.0), np.zeros(10), [0])[1] == [False]


def test_scan_overhead_and_wrong_peak():
    P_mpp = np.full(100, 100.0); P = P_mpp.copy(); scan = np.zeros(100, int); scan[[0, 50]] = 1; P[[0, 50]] = 40.0
    assert abs(M.scan_overhead(P, P_mpp, scan) - 100 * 120 / 10000) < 1e-12
    v_cmd = np.array([15.0, 15.5, 24.0]); v_g = np.full(3, 15.2); assert abs(M.wrong_peak_rate(v_cmd, v_g, np.array([1, 1, 1])) - 100 / 3) < 1e-9
    assert np.isnan(M.wrong_peak_rate(v_cmd, v_g, np.zeros(3)))


def test_instantaneous_efficiency():
    assert np.allclose(M.instantaneous_efficiency([50, 99], [100, 100]), [50, 99])
