"""Single shared plotting style (MATLAB-like) and figure export for every figure of the study.

Rules enforced by save_figure(): no title inside the image (neither axes titles nor suptitle), panel letters '(a)', '(b)', ... drawn inside each axes
at the top-left on multi-axes figures, PDF (vector) + PNG at 600 dpi, and the plotted data written to results/figures/data/<name>.csv.
Widths: 3.5 in (single column) or 7.2 in (double column). Fonts: Times New Roman if installed, else the metric-compatible Liberation Serif."""
import os
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from mppt.config import RESULTS

W1, W2 = 3.5, 7.2                      # inches
# MATLAB default colour order; one fixed colour/marker per controller in every figure (PSO and tree-scan are not in MATLAB's seven)
MATLAB = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']
COLORS = {'P&O': '#D95319', 'IncCond': '#7E2F8E', 'Poly2(G,T)-MPPT': '#EDB120', 'ANN-MPPT': '#0072BD', 'Scan+P&O (no ML)': '#4DBEEE', 'DNN scan-only (no P&O)': '#A2142F',
          'Hybrid (DNN+P&O)': '#77AC30', 'PSO': '#4D4D4D', 'Tree scan+P&O': '#8C564B', 'True MPP': '#000000'}
MARKERS = {'P&O': 'o', 'IncCond': 's', 'Poly2(G,T)-MPPT': '^', 'ANN-MPPT': 'D', 'Scan+P&O (no ML)': 'v', 'DNN scan-only (no P&O)': 'P', 'Hybrid (DNN+P&O)': '*', 'PSO': 'X', 'Tree scan+P&O': 'h'}
SHORT = {'P&O': 'P&O', 'IncCond': 'IncCond', 'Poly2(G,T)-MPPT': 'Poly2', 'ANN-MPPT': 'ANN', 'Scan+P&O (no ML)': 'Scan+P&O', 'DNN scan-only (no P&O)': 'DNN-only', 'Hybrid (DNN+P&O)': 'Hybrid', 'PSO': 'PSO', 'Tree scan+P&O': 'Tree-scan'}
CMAP = 'viridis'                       # perceptually uniform stand-in for MATLAB's parula

_FONTS = {f.name for f in fm.fontManager.ttflist}
FONT = next((f for f in ('Times New Roman', 'Liberation Serif', 'DejaVu Serif') if f in _FONTS), 'DejaVu Serif')
REGISTRY = []                          # (name, caption-key) in save order


def apply_style():
    mpl.rcParams.update({
        'font.family': 'serif', 'font.serif': [FONT], 'mathtext.fontset': 'stix', 'font.size': 9, 'axes.labelsize': 9.5, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
        'legend.fontsize': 8, 'axes.linewidth': 0.9, 'axes.edgecolor': 'black', 'axes.facecolor': 'white', 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
        'axes.grid': True, 'grid.color': '0.78', 'grid.linestyle': '--', 'grid.linewidth': 0.5, 'axes.axisbelow': True, 'axes.spines.top': True, 'axes.spines.right': True,
        'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True, 'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
        'legend.frameon': True, 'legend.edgecolor': 'black', 'legend.fancybox': False, 'legend.framealpha': 1.0, 'legend.facecolor': 'white', 'legend.borderpad': 0.4,
        'lines.linewidth': 1.6, 'lines.markersize': 4.5, 'axes.prop_cycle': mpl.cycler(color=MATLAB), 'figure.dpi': 110, 'savefig.dpi': 600, 'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.03, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'image.cmap': CMAP})


def fig_dir():
    d = os.path.join(RESULTS, 'figures'); os.makedirs(os.path.join(d, 'data'), exist_ok=True); return d


def panel(ax, letter, loc='tl', dx=0.02, dy=0.03, **kw):
    """Draw '(a)' inside the axes (top-left by default)."""
    pos = {'tl': (dx, 1 - dy, 'left', 'top'), 'tr': (1 - dx, 1 - dy, 'right', 'top'), 'bl': (dx, dy, 'left', 'bottom'), 'br': (1 - dx, dy, 'right', 'bottom')}[loc]
    t = ax.text(pos[0], pos[1], f'({letter})', transform=ax.transAxes, ha=pos[2], va=pos[3], fontsize=10, fontweight='bold', zorder=20,
                bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none', alpha=0.85), **kw)
    t._is_panel_letter = True
    return t


def _check(fig):
    assert fig._suptitle is None, 'figures must not carry a title inside the image'
    axes = [a for a in fig.axes if a.get_visible() and a.get_label() != '<colorbar>' and not (getattr(a, '_colorbar', None))]
    axes = [a for a in axes if a.get_xlabel() or a.get_ylabel() or a.lines or a.patches or a.collections or a.images]
    for a in axes:
        assert not a.get_title(), 'figures must not carry axes titles; use the caption'
    real = [a for a in axes if a.get_position().width > 0.05 and not getattr(a, '_twinned', False)]
    groups = {}
    for a in real: groups.setdefault(a.get_subplotspec(), []).append(a)          # twin axes share the subplot spec: one panel letter per group
    if len(groups) > 1:
        for spec, ax_list in groups.items():
            assert any(getattr(t, '_is_panel_letter', False) for a in ax_list for t in a.texts), f'missing panel letter on axes {ax_list[0].get_ylabel() or ax_list[0].get_xlabel()!r}'


def save_figure(fig, name, data):
    """Export fig as <name>.pdf and <name>.png (600 dpi) and the plotted `data` (DataFrame, or dict panel->DataFrame) as data/<name>.csv."""
    _check(fig)
    d = fig_dir()
    if isinstance(data, dict):
        data = pd.concat([v.assign(panel=k) for k, v in data.items()], ignore_index=True)
    assert isinstance(data, pd.DataFrame) and len(data), 'every figure must export the plotted data'
    data.to_csv(os.path.join(d, 'data', name + '.csv'), index=False)
    fig.savefig(os.path.join(d, name + '.png')); fig.savefig(os.path.join(d, name + '.pdf'))
    REGISTRY.append(name); plt.close(fig)


def axis_break(ax, y_bottom_label=None, size=0.015):
    """Draw '//' break marks on the left axis (zoomed y-axis does not start at zero) and optionally write the axis start."""
    kw = dict(transform=ax.transAxes, color='k', clip_on=False, lw=1.0)
    for x in (0, 1):
        ax.plot((x - size, x + size), (0.05 - size, 0.05 + size), **kw); ax.plot((x - size, x + size), (0.075 - size, 0.075 + size), **kw)


def grouped_bars(ax, groups, series, colors, ylim=None, label_values=True, fmt='{:.2f}', break_axis=False, hatch_nan='//', width=0.8):
    """series: dict name -> (mean array, std array or None) aligned with `groups`. NaN means = 'not converged / not applicable' (hatched stub + 'n/c')."""
    n = len(series); w = width / n; x = np.arange(len(groups)); rows = []
    base = ylim[0] if ylim else 0.0
    for k, (nm, (mu, sd)) in enumerate(series.items()):
        mu = np.asarray(mu, float); sd = None if sd is None else np.asarray(sd, float)
        xs = x + (k - (n - 1) / 2) * w
        ok = ~np.isnan(mu)
        ax.bar(xs[ok], mu[ok] - base, w, bottom=base, color=colors[nm], edgecolor='black', linewidth=0.5, label=nm, zorder=3)
        if sd is not None:
            ax.errorbar(xs[ok], mu[ok], yerr=np.where(np.isnan(sd[ok]), 0, sd[ok]), fmt='none', ecolor='k', elinewidth=0.8, capsize=1.8, zorder=4)
        for xi, m_ in zip(xs[~ok], mu[~ok]):
            ax.text(xi, base, 'n/c', rotation=90, ha='center', va='bottom', fontsize=6.5, zorder=5)
        if label_values and n * len(groups) <= 8:
            for xi, m_ in zip(xs[ok], mu[ok]): ax.text(xi, m_, fmt.format(m_), ha='center', va='bottom', fontsize=6.5, zorder=5)
        rows += [{'group': g, 'series': nm, 'mean': m_, 'std': (np.nan if sd is None else s_)} for g, m_, s_ in zip(groups, mu, (sd if sd is not None else [np.nan] * len(groups)))]
    ax.set_xticks(x); ax.set_xticklabels(groups)
    if ylim: ax.set_ylim(*ylim)
    if break_axis and ylim and ylim[0] > 0: axis_break(ax)
    ax.grid(axis='x', visible=False)
    return pd.DataFrame(rows)


def legend_fig(fig, handles=None, labels=None, ncol=4, y=-0.02):
    kw = dict(loc='lower center', ncol=ncol, frameon=True, edgecolor='black', fancybox=False, bbox_to_anchor=(0.5, y))
    return fig.legend(handles=handles, labels=labels, **kw) if handles is not None else fig.legend(**kw)
