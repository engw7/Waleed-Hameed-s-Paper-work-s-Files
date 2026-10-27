"""Block diagrams and flowcharts (Fig01-Fig08). Every decision box mirrors the implementation in mppt/controllers.py (thresholds are read from the code constants)."""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle, Circle
import mppt_style as sty
from .config import CFG
import pv_hybrid_data as hd


def _wrap(s, n):
    res = []
    for seg in s.split('\n'):
        out, line = [], ''
        for w in seg.split(' '):
            if len(line) + len(w) + 1 > n and line: out.append(line); line = w
            else: line = (line + ' ' + w).strip()
        res += out + [line]
    return '\n'.join(res)


class Flow:
    """Tiny flow-chart engine: nodes on a grid, rectangular/diamond/rounded shapes, orthogonal arrows with labels."""
    def __init__(self, w, h, figsize):
        self.fig, self.ax = plt.subplots(figsize=figsize); self.ax.set_xlim(0, w); self.ax.set_ylim(0, h); self.ax.axis('off'); self.ax.grid(False)
        self.nodes, self.edges = {}, []

    def node(self, key, x, y, text, shape='box', w=2.6, h=0.8, fs=7.2, wrap=26, fc='white'):
        t = _wrap(text, wrap if shape != 'diamond' else wrap - 6)
        if shape == 'box': self.ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc=fc, ec='black', lw=1.0, zorder=2))
        elif shape == 'round': self.ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.02,rounding_size=0.25', fc='0.92', ec='black', lw=1.0, zorder=2))
        elif shape == 'diamond': self.ax.add_patch(Polygon([(x - w / 2, y), (x, y + h / 2), (x + w / 2, y), (x, y - h / 2)], closed=True, fc='#FFF4D6', ec='black', lw=1.0, zorder=2))
        self.ax.text(x, y, t, ha='center', va='center', fontsize=fs, zorder=3)
        self.nodes[key] = dict(x=x, y=y, w=w, h=h, shape=shape, text=text)

    def _anchor(self, k, side):
        n = self.nodes[k]; return {'l': (n['x'] - n['w'] / 2, n['y']), 'r': (n['x'] + n['w'] / 2, n['y']), 't': (n['x'], n['y'] + n['h'] / 2), 'b': (n['x'], n['y'] - n['h'] / 2)}[side]

    def edge(self, a, sa, b, sb, label='', via=(), lab_pos=0.5, off=(0.0, 0.12), ha='center'):
        p = [self._anchor(a, sa)] + list(via) + [self._anchor(b, sb)]
        for i in range(len(p) - 2):
            self.ax.plot([p[i][0], p[i + 1][0]], [p[i][1], p[i + 1][1]], 'k-', lw=0.9, zorder=1)
        self.ax.annotate('', xy=p[-1], xytext=p[-2], arrowprops=dict(arrowstyle='-|>', color='k', lw=0.9, shrinkA=0, shrinkB=0), zorder=1)
        if label:
            i = min(int(lab_pos * (len(p) - 1)), len(p) - 2); mx, my = (p[i][0] + p[i + 1][0]) / 2, (p[i][1] + p[i + 1][1]) / 2
            self.ax.text(mx + off[0], my + off[1], label, fontsize=6.8, ha=ha, va='bottom', zorder=4, bbox=dict(fc='white', ec='none', pad=0.4))
        self.edges.append(dict(src=a, dst=b, label=label))

    def data(self):
        return pd.DataFrame([{'kind': 'node', 'id': k, 'text': v['text'], 'x': v['x'], 'y': v['y'], 'shape': v['shape']} for k, v in self.nodes.items()] +
                            [{'kind': 'edge', 'id': f"{e['src']}->{e['dst']}", 'text': e['label'], 'x': np.nan, 'y': np.nan, 'shape': ''} for e in self.edges])


def fig_po():
    from .controllers import PerturbAndObserveMPPT
    p = PerturbAndObserveMPPT(); dP_th, step = 0.01, CFG['step_size']                    # thresholds quoted from controllers.py (|dP| > 0.01 W)
    f = Flow(10.8, 14.2, (sty.W2, 8.6))
    f.node('s', 5, 13.7, 'Start of control cycle k', 'round', w=3.4, h=0.6)
    f.node('m', 5, 12.6, 'Measure V(k), I(k) at the applied command;\nP(k) = V(k)·I(k)', w=5.4, h=0.9, wrap=40)
    f.node('k0', 5, 11.2, 'k = 0 ?', 'diamond', w=2.0, h=1.0)
    f.node('dp', 5, 9.6, f'ΔP = P(k) − P(k−1)\nΔV = V(k) − V(k−1)', w=3.8, h=0.9, wrap=40)
    f.node('th', 5, 8.1, f'|ΔP| > {dP_th} W ?', 'diamond', w=2.8, h=1.1)
    f.node('s1', 2.2, 6.7, 'ΔP > 0 ?', 'diamond', w=1.9, h=0.9)
    f.node('a1', 0.9, 5.2, 'ΔV > 0 ?', 'diamond', w=1.7, h=0.8, fs=6.8)
    f.node('a2', 3.6, 5.2, 'ΔV > 0 ?', 'diamond', w=1.7, h=0.8, fs=6.8)
    f.node('up', 0.8, 3.8, 'd = +1', 'box', w=1.3, h=0.55, fs=6.8); f.node('dn', 2.2, 3.8, 'd = −1', 'box', w=1.3, h=0.55, fs=6.8)
    f.node('dn2', 3.6, 3.8, 'd = −1', 'box', w=1.3, h=0.55, fs=6.8); f.node('up2', 4.9, 3.8, 'd = +1', 'box', w=1.3, h=0.55, fs=6.8)
    f.node('keep', 8.6, 5.2, 'd unchanged\n(previous direction)', 'box', w=2.4, h=0.8, fs=6.8)
    f.node('upd', 5, 2.4, f'V_cmd(k+1) = clip( V_cmd(k) + d·{step} V , {CFG["v_min_op"]:.0f} V , {CFG["v_max_op"]:.0f} V )', 'box', w=7.0, h=0.8, wrap=60)
    f.node('st', 5, 1.2, 'Store V(k), P(k) as V(k−1), P(k−1);  apply V_cmd(k+1)', 'box', w=7.0, h=0.7, wrap=60)
    f.node('e', 5, 0.35, 'End of cycle', 'round', w=2.6, h=0.5)
    f.node('k0b', 9.3, 11.2, 'No perturbation\n(command unchanged)', 'box', w=2.1, h=0.8, fs=6.8)
    f.edge('s', 'b', 'm', 't'); f.edge('m', 'b', 'k0', 't'); f.edge('k0', 'b', 'dp', 't', 'No')
    f.edge('k0', 'r', 'k0b', 'l', 'Yes'); f.edge('k0b', 'b', 'st', 'r', via=[(9.3, 1.2)])
    f.edge('dp', 'b', 'th', 't'); f.edge('th', 'l', 's1', 't', 'Yes', via=[(2.2, 8.1)]); f.edge('th', 'r', 'keep', 't', 'No', via=[(8.6, 8.1)])
    f.edge('s1', 'l', 'a1', 't', 'Yes', via=[(0.9, 6.7)]); f.edge('s1', 'r', 'a2', 't', 'No', via=[(3.6, 6.7)])
    f.edge('a1', 'b', 'up', 't', 'Yes'); f.edge('a1', 'r', 'dn', 't', 'No', via=[(1.55, 5.2), (1.55, 4.4), (2.2, 4.4)])
    f.edge('a2', 'l', 'dn2', 't', '', via=[(2.75, 5.2)]) if False else None
    f.edge('a2', 'b', 'dn2', 't', 'Yes'); f.edge('a2', 'r', 'up2', 't', 'No', via=[(4.45, 5.2), (4.45, 4.4), (4.9, 4.4)])
    for k in ('up', 'dn', 'dn2', 'up2'): f.edge(k, 'b', 'upd', 't', via=[(f.nodes[k]['x'], 3.1), (5, 3.1)]) if k in ('up', 'dn') else None
    f.edge('dn2', 'b', 'upd', 't', via=[(3.6, 3.1), (5, 3.1)]); f.edge('up2', 'b', 'upd', 't', via=[(4.9, 3.1), (5, 3.1)])
    f.edge('keep', 'b', 'upd', 'r', via=[(8.6, 2.4)]); f.edge('upd', 'b', 'st', 't'); f.edge('st', 'b', 'e', 't')
    sty.panel(f.ax, 'a', 'tl') if False else None
    return f


def fig_inc():
    f = Flow(11.2, 14.2, (sty.W2, 8.6)); step = CFG['step_size']
    f.node('s', 5, 13.7, 'Start of control cycle k', 'round', w=3.4, h=0.6)
    f.node('m', 5, 12.6, 'Measure V(k), I(k) at the applied command', w=4.6, h=0.8)
    f.node('k0', 5, 11.3, 'k = 0 ?', 'diamond', w=2.0, h=1.0)
    f.node('init', 9.9, 11.3, f'ΔV_cmd = +{step} V\n(first perturbation)', 'box', w=2.5, h=0.8, fs=6.8)
    f.node('d', 5, 9.9, 'ΔV = V(k) − V(k−1)\nΔI = I(k) − I(k−1)', w=3.6, h=0.9)
    f.node('v0', 5, 8.4, '|ΔV| > 0.1 V ?', 'diamond', w=2.8, h=1.1)
    f.node('inc', 2.0, 6.8, 'G = ΔI/ΔV + I(k)/V(k)\n(second term = 0 if V ≤ 0.1 V)', w=3.6, h=0.9, fs=6.6, wrap=30)
    f.node('c1', 2.0, 5.3, '|G| ≤ 0.001 ?', 'diamond', w=2.2, h=0.9, fs=6.8)
    f.node('c2', 2.0, 3.9, 'G > 0.001 ?', 'diamond', w=2.2, h=0.9, fs=6.8)
    f.node('z', 0.7, 2.5, 'ΔV_cmd = 0\n(at MPP)', 'box', w=1.4, h=0.7, fs=6.3); f.node('p', 2.0, 2.5, f'ΔV_cmd =\n+{step} V', 'box', w=1.3, h=0.7, fs=6.3); f.node('n', 3.3, 2.5, f'ΔV_cmd =\n−{step} V', 'box', w=1.3, h=0.7, fs=6.3)
    f.node('dv0', 7.6, 6.8, '|ΔI| > 0.001 A ?\n(ΔV ≈ 0 branch)', 'diamond', w=3.0, h=1.1, fs=6.8)
    f.node('i1', 6.7, 5.0, 'ΔI > 0 ?', 'diamond', w=1.7, h=0.8, fs=6.8)
    f.node('ip', 5.9, 3.7, f'ΔV_cmd = +{step} V', 'box', w=1.6, h=0.6, fs=6.3); f.node('in', 7.6, 3.7, f'ΔV_cmd = −{step} V', 'box', w=1.6, h=0.6, fs=6.3)
    f.node('iz', 9.0, 5.0, 'ΔV_cmd = 0', 'box', w=1.5, h=0.6, fs=6.3)
    f.node('upd', 5, 1.5, f'V_cmd(k+1) = clip( V_cmd(k) + ΔV_cmd , {CFG["v_min_op"]:.0f} V , {CFG["v_max_op"]:.0f} V );  store V(k), I(k)', 'box', w=8.0, h=0.8, wrap=60)
    f.node('e', 5, 0.4, 'End of cycle', 'round', w=2.6, h=0.5)
    f.edge('s', 'b', 'm', 't'); f.edge('m', 'b', 'k0', 't'); f.edge('k0', 'r', 'init', 'l', 'Yes'); f.edge('k0', 'b', 'd', 't', 'No'); f.edge('init', 'b', 'upd', 'r', via=[(9.9, 1.5)])
    f.edge('d', 'b', 'v0', 't'); f.edge('v0', 'l', 'inc', 't', 'Yes', via=[(2.0, 8.4)]); f.edge('v0', 'r', 'dv0', 't', 'No', via=[(7.6, 8.4)])
    f.edge('inc', 'b', 'c1', 't'); f.edge('c1', 'l', 'z', 't', 'Yes', via=[(0.7, 5.3)]); f.edge('c1', 'b', 'c2', 't', 'No'); f.edge('c2', 'b', 'p', 't', 'Yes'); f.edge('c2', 'r', 'n', 't', 'No', via=[(3.3, 3.9)])
    f.edge('dv0', 'b', 'i1', 't', 'Yes'); f.edge('dv0', 'r', 'iz', 't', 'No', via=[(9.0, 6.8)]); f.edge('i1', 'b', 'ip', 't', 'Yes'); f.edge('i1', 'r', 'in', 't', 'No', via=[(7.6, 5.0)])
    for k in ('z', 'p', 'n'): f.edge(k, 'b', 'upd', 't', via=[(f.nodes[k]['x'], 1.95), (f.nodes[k]['x'], 1.95)])
    for k in ('ip', 'in'): f.edge(k, 'b', 'upd', 't', via=[(f.nodes[k]['x'], 1.95)])
    f.edge('iz', 'b', 'upd', 't', via=[(9.0, 1.95)]); f.edge('upd', 'b', 'e', 't')
    return f


def fig_hybrid_blocks():
    f = Flow(15, 7.0, (sty.W2, 3.5))
    f.node('pv', 1.4, 5.6, 'PV module\n(3 substrings,\nbypass diodes)', w=2.3, h=1.2, fs=7); f.node('cv', 6.0, 5.6, 'Boost converter\n+ voltage loop', w=2.8, h=1.1, fs=7)
    f.node('vc', 13.6, 5.6, 'V_cmd', w=1.6, h=0.7)
    f.node('sen', 1.4, 3.3, 'Sensors:\nV, I, G, T', w=2.3, h=1.0, fs=7)
    f.node('scan', 5.2, 3.3, 'Stage 1: fast I-V scan\nI at 14 probe voltages\n8 ... 35 V', w=3.4, h=1.3, fs=6.8)
    f.node('dnn', 9.2, 3.3, 'DNN: G_in, T, I1..I14\n-> V_est (global MPP)', w=3.4, h=1.3, fs=6.8)
    f.node('po', 13.0, 3.3, 'Stage 2: P&O\n(0.5 V steps)\nstarts at V_est', w=2.8, h=1.3, fs=6.8)
    f.node('sw', 9.2, 1.1, 'Switch-back to stage 1: P/G < 0.9 x max(P/G) since the last scan (>= 30 cycles), or every 300 cycles', w=9.0, h=0.8, fs=6.8, wrap=80)
    f.edge('pv', 'r', 'cv', 'l', 'V_pv, I_pv'); f.edge('pv', 'b', 'sen', 't'); f.edge('sen', 'r', 'scan', 'l'); f.edge('scan', 'r', 'dnn', 'l', 'I1..I14'); f.edge('dnn', 'r', 'po', 'l', 'V_est')
    f.edge('po', 't', 'vc', 'b'); f.edge('vc', 'l', 'cv', 'r')
    f.edge('po', 'b', 'sw', 'r', via=[(13.0, 1.1)]); f.edge('sw', 'l', 'scan', 'b', via=[(5.2, 1.1)])
    return f


def fig_dnn_dataflow():
    f = Flow(14.2, 6.4, (sty.W2, 3.5)); n_in = 2 + len(hd.PROBES)
    f.node('in', 1.9, 3.2, f'Input vector x ({n_in} values)\n[G_in, T, I1 ... I14]\nG_in: mean irradiance (W/m2)\nT: ambient temperature (degC)\nI_k: current at probe V_k (A)', w=3.3, h=2.0, fs=5.6, wrap=30)
    f.node('sc', 4.6, 3.2, 'Standardise\n(training-set\nmean/std)', w=1.5, h=1.0, fs=6.8)
    f.node('h1', 6.1, 3.2, 'Dense 128\nReLU', w=1.3, h=1.0, fs=6.8); f.node('h2', 7.6, 3.2, 'Dense 128\nReLU', w=1.3, h=1.0, fs=6.8); f.node('h3', 9.1, 3.2, 'Dense 64\nReLU', w=1.3, h=1.0, fs=6.8)
    f.node('o', 10.9, 3.2, 'Dense 1\nlinear', w=1.3, h=1.0, fs=6.8); f.node('y', 12.4, 3.2, 'V_est\n(V)', w=1.1, h=0.9, fs=6.8)
    f.node('loss', 10.9, 1.0, 'Training: MSE between standardised V_est and V_gmpp (global-MPP voltage of the simulated string); Adam 1e-3; early stopping on 2015-16', w=5.0, h=1.5, fs=6.0, wrap=40)
    f.node('use', 11.2, 5.4, 'Use: V_est = initial command of P&O after a scan; the converter voltage loop applies V_cmd (D = PI(V_cmd - V_pv) + 1 - V_cmd/V_bus)', w=5.0, h=1.5, fs=6.0, wrap=40)
    f.node('cnt', 3.6, 5.4, 'Hidden layers: 3 (128-128-64); parameters: 27,009', 'box', w=4.6, h=0.7, fs=6.6, wrap=44)
    for a, b in (('in', 'sc'), ('sc', 'h1'), ('h1', 'h2'), ('h2', 'h3'), ('h3', 'o'), ('o', 'y')): f.edge(a, 'r', b, 'l')
    f.edge('loss', 't', 'o', 'b'); f.edge('y', 't', 'use', 'b')
    return f


def fig_hybrid_states():
    f = Flow(12, 5.6, (sty.W2, 3.5))
    f.node('A', 2.2, 3.0, 'SCAN\n(one control cycle,\nenergy = mean probe power)', w=3.2, h=1.5, fs=7)
    f.node('B', 9.4, 3.0, 'TRACK\nP&O from V_est,\nstep 0.5 V', w=3.2, h=1.5, fs=7)
    f.node('i', 0.5, 4.9, 'start', 'round', w=0.9, h=0.5, fs=6.5)
    f.edge('i', 'b', 'A', 't') if False else None
    f.edge('A', 'r', 'B', 'l', 'V_est = DNN(G, T, I₁…I₁₄)\nhandoff: P&O restarts at V_est', off=(-1.9, 0.25))
    f.edge('B', 'b', 'A', 'b', 'trigger: P/G < 0.9·max(P/G) since scan (≥ 30 cycles)\n or 300 cycles since the last scan', via=[(9.4, 1.1), (2.2, 1.1)], off=(-5.6, -0.65))
    return f


def fig_pso():
    f = Flow(12, 11.5, (sty.W2, 6.6)); from .controllers import PSOMPPT as P
    f.node('s', 2.8, 11.0, 'Start of control cycle', 'round', w=3.2, h=0.55)
    f.node('h', 2.8, 9.8, 'Holding best voltage ?', 'diamond', w=3.8, h=1.1, fs=6.4, wrap=30)
    f.node('chk', 7.2, 9.8, f'|P - P_hold| / P_hold > {P.RESTART:.0%}\nor hold > {P.MAX_HOLD} cycles ?', 'diamond', w=4.4, h=1.5, fs=6.2, wrap=34)
    f.node('re', 7.2, 8.2, 'Re-initialise swarm\n(particles spread over the window)', w=3.6, h=0.8, fs=6.6)
    f.node('ev', 2.8, 8.2, 'Apply particle i; measure P;\nupdate pbest_i and gbest', w=3.8, h=0.9, fs=6.6)
    f.node('all', 2.8, 7.0, f'All {P.N} particles evaluated ?', 'diamond', w=3.4, h=1.0, fs=6.6)
    f.node('up', 2.8, 5.4, f'v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)\nx = clip(x + v);  w={P.W_IN}, c1={P.C1}, c2={P.C2}', w=4.4, h=1.0, fs=5.8, wrap=60)
    f.node('cv', 2.8, 3.8, f'all particles within {P.CONV_V} V of gbest ?', 'diamond', w=3.6, h=1.0, fs=6.4)
    f.node('hd', 7.2, 3.8, 'Hold gbest voltage', w=2.4, h=0.7, fs=6.6); f.node('nx', 2.8, 2.2, 'Next particle: command x_i', w=3.0, h=0.7, fs=6.6)
    f.node('e', 5, 0.8, 'End of cycle', 'round', w=2.6, h=0.5)
    f.edge('s', 'b', 'h', 't'); f.edge('h', 'r', 'chk', 'l', 'Yes'); f.edge('h', 'b', 'ev', 't', 'No')
    f.edge('chk', 'b', 're', 't', 'Yes'); f.edge('re', 'l', 'ev', 'r')
    f.edge('chk', 'r', 'e', 'r', 'No: keep holding', via=[(10.6, 9.8), (10.6, 0.8)], off=(0.0, 0.12))
    f.edge('ev', 'b', 'all', 't'); f.edge('all', 'b', 'up', 't', 'Yes'); f.edge('all', 'l', 'nx', 'l', 'No', via=[(0.3, 7.0), (0.3, 2.2)], lab_pos=0.0, off=(0.25, 0.12))
    f.edge('up', 'b', 'cv', 't'); f.edge('cv', 'r', 'hd', 'l', 'Yes'); f.edge('cv', 'b', 'nx', 't', 'No')
    f.edge('hd', 'b', 'e', 't', via=[(7.2, 1.7), (5, 1.7)]); f.edge('nx', 'b', 'e', 'l', via=[(2.8, 0.8)])
    return f


def fig_system():
    fig, ax = plt.subplots(figsize=(sty.W2, 3.3)); ax.set_xlim(0, 12.4); ax.set_ylim(0, 6.6); ax.axis('off'); ax.grid(False)
    def box(x, y, w, h, t, fs=7): ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc='white', ec='k', lw=1.0, zorder=2)); ax.text(x, y, t, ha='center', va='center', fontsize=fs, zorder=3)
    def arr(a, b, t=''): ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='-|>', color='k', lw=0.9)); ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.15, t, fontsize=6.5, ha='center')
    ax.add_patch(Rectangle((0.3, 3.0), 4.0, 3.0, fc='none', ec='k', lw=1.0, ls='--'))
    for i in range(3):
        x0 = 0.6 + i * 1.15; ax.add_patch(Rectangle((x0, 4.1), 0.9, 1.5, fc='#DCE9F5', ec='k', lw=1.0)); ax.text(x0 + 0.45, 4.85, '18\ncells', ha='center', va='center', fontsize=6.5)
        ax.text(x0 + 0.45, 3.5, 'bypass\ndiode', ha='center', va='center', fontsize=6)
    ax.text(2.3, 6.25, 'PV module: 3 substrings', ha='center', fontsize=7)
    box(5.8, 4.55, 1.4, 1.0, 'C$_{in}$\n470 uF'); box(7.6, 4.55, 1.3, 1.0, 'L\n0.5 mH'); box(9.4, 4.55, 1.5, 1.0, 'Q + D\nboost stage'); box(11.4, 4.55, 1.3, 1.0, 'DC bus\n48 V')
    arr((4.3, 4.55), (5.1, 4.55), 'V$_{pv}$, I$_{pv}$'); arr((6.5, 4.55), (6.95, 4.55)); arr((8.25, 4.55), (8.65, 4.55)); arr((10.15, 4.55), (10.75, 4.55))
    box(2.3, 1.5, 2.4, 1.0, 'Sensors\nV$_{pv}$, I$_{pv}$, G, T'); box(5.6, 1.5, 2.4, 1.0, 'MPPT controller\n-> V$_{cmd}$'); box(8.4, 1.5, 1.9, 1.0, 'Voltage loop\n(PI) -> D'); box(10.9, 1.5, 1.2, 1.0, 'PWM')
    arr((3.5, 1.5), (4.4, 1.5)); arr((6.8, 1.5), (7.45, 1.5), 'V$_{cmd}$'); arr((9.35, 1.5), (10.3, 1.5), 'D'); arr((10.9, 2.0), (9.6, 4.05), 'gate')
    arr((2.3, 3.0), (2.3, 2.0)); ax.text(6.2, 0.45, 'Controller variants: P&O, IncCond, PSO, Poly2, ANN, Scan+P&O, Tree-scan, DNN scan-only, Hybrid (DNN+P&O)', ha='center', fontsize=7)
    return fig, pd.DataFrame({'block': ['PV module', 'C_in', 'L', 'Q+D', 'DC bus', 'Sensors', 'MPPT controller', 'Voltage loop', 'PWM'], 'note': ['3 substrings x 18 cells', '470 uF', '0.5 mH', '50 kHz', '48 V', 'V,I,G,T', 'V_cmd', 'PI -> D', 'switching']})


def fig_string():
    fig, ax = plt.subplots(figsize=(sty.W2, 2.6)); ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis('off'); ax.grid(False)
    for i, (lab, g) in enumerate((('substring 1\n(18 cells, G)', 'G'), ('substring 2\n(m cells at f·G,\n18−m at G)', 'f·G'), ('substring 3\n(18 cells, G)', 'G'))):
        x0 = 0.8 + i * 3.6; ax.add_patch(Rectangle((x0, 1.6), 2.6, 1.6, fc='#DCE9F5', ec='k')); ax.text(x0 + 1.3, 2.4, lab, ha='center', va='center', fontsize=7)
        ax.annotate('', xy=(x0 + 1.3, 3.9), xytext=(x0 + 1.3, 3.2), arrowprops=dict(arrowstyle='-', lw=0.8)); ax.text(x0 + 1.3, 4.1, 'bypass diode (V ≥ −0.7 V)', ha='center', fontsize=6.5)
    ax.text(6, 0.6, 'V_string(I) = Σ_k max( −0.7 V , Σ_groups V_group(I) ),   V_group(I) = R_sh(I_ph + I_0) − I(R_s + R_sh) − a·W( I_0 R_sh / a · exp(R_sh(I_ph + I_0 − I)/a) )', ha='center', fontsize=7)
    return fig, pd.DataFrame({'item': ['substrings', 'cells per substring', 'bypass threshold V', 'group voltage'], 'value': [3, 18, -0.7, 'Lambert-W inverse single-diode']})


def build_all():
    """Draws Fig01-Fig08 and exports them (data = node/edge lists)."""
    out = []
    fig, d = fig_system(); sty.save_figure(fig, 'Fig01_system_overview', d); out.append('Fig01_system_overview')
    fig, d = fig_string(); sty.save_figure(fig, 'Fig02_shaded_string_model', d); out.append('Fig02_shaded_string_model')
    for fn, name in ((fig_hybrid_blocks, 'Fig03_hybrid_block_diagram'), (fig_dnn_dataflow, 'Fig04_dnn_dataflow'), (fig_po, 'Fig05_flowchart_PO'), (fig_inc, 'Fig06_flowchart_IncCond'),
                     (fig_hybrid_states, 'Fig07_hybrid_handoff_states'), (fig_pso, 'Fig08_flowchart_PSO')):
        f = fn(); sty.save_figure(f.fig, name, f.data()); out.append(name)
    return out
