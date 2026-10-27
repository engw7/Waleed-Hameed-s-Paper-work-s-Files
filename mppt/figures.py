"""Figure builders shared by the experiments: time-domain layout (rows = power / voltage / duty [/ inductor current], left = full run, right = zoom, scan cycles marked by triangles)."""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mppt_style as sty

ROWKEY = {'P': ('Power (W)', 'P', 'mpp_power'), 'V': ('PV voltage (V)', 'V', 'mpp_voltage'), 'D': ('Duty cycle (-)', None, None), 'IL': ('Inductor current (A)', 'I_L', None)}


def _duty(df): return df['D_conv'] if 'D_conv' in df else df['duty']


def time_domain(logs, name, rows=('P', 'V', 'D'), zoom=(0, 100), controllers=None, band=True, markevery=400, height_per_row=1.75):
    """logs: dict controller -> run DataFrame (columns time, P, V, mpp_power, mpp_voltage, duty / D_conv, I_L, scan). Returns the plotted data and saves the figure."""
    sty.apply_style(); ctr = controllers or list(logs); nr = len(rows)
    fig, axes = plt.subplots(nr, 2, figsize=(sty.W2, height_per_row * nr + 0.5), squeeze=False, gridspec_kw=dict(width_ratios=[1.35, 1]))
    data = []; letters = iter('abcdefghijkl')
    ref = logs[ctr[0]]; t = ref.time.to_numpy()
    for i, row in enumerate(rows):
        ylabel, col, truth = ROWKEY[row]
        for j, ax in enumerate(axes[i]):
            lo, hi = (t[0], t[-1]) if j == 0 else zoom
            if truth is not None:
                ax.plot(t, ref[truth], 'k--', lw=1.0, zorder=5, label='True MPP')
                if row == 'P' and band: ax.fill_between(t, 0.98 * ref[truth], ref[truth], color='0.85', zorder=0, lw=0)
            ymins, ymaxs = [], []
            for c in ctr:
                df = logs[c]; y = (_duty(df) if row == 'D' else df[col]).to_numpy()
                ax.plot(df.time, y, color=sty.COLORS[c], marker=sty.MARKERS[c], markevery=(markevery if j == 0 else max(1, markevery // 8)), ms=3.2, lw=1.1, zorder=4, label=sty.SHORT[c])
                m = (df.time >= lo) & (df.time <= hi); ymins.append(y[m].min()); ymaxs.append(y[m].max())
                if j == 1 and row == 'P' and 'scan' in df and df.scan.sum():
                    sm = (df.scan.to_numpy() == 1) & m.to_numpy()
                    if sm.any(): ax.plot(df.time[sm], np.full(sm.sum(), 0.035), marker='v', ls='none', color=sty.COLORS[c], ms=3.5, transform=ax.get_xaxis_transform(), zorder=6)
                if i == 0 and j == 0:
                    data.append(df[['time']].assign(controller=c, P=df.P, V=df.V, D=_duty(df), mpp_power=df.mpp_power, mpp_voltage=df.mpp_voltage, scan=df.scan if 'scan' in df else 0))
            if truth is not None:
                m = (t >= lo) & (t <= hi); ymins.append(ref[truth].to_numpy()[m].min()); ymaxs.append(ref[truth].to_numpy()[m].max())
            pad = 0.06 * (max(ymaxs) - min(ymins) + 1e-9); ax.set_xlim(lo, hi); ax.set_ylim(min(ymins) - pad , max(ymaxs) + pad)
            if j == 0: ax.set_ylabel(ylabel)
            if i == nr - 1: ax.set_xlabel('Time (s)')
            sty.panel(ax, next(letters), loc='br' if row == 'P' else 'bl')
    handles = [Line2D([], [], color=sty.COLORS[c], marker=sty.MARKERS[c], ms=4, lw=1.2, label=sty.SHORT[c]) for c in ctr] + [Line2D([], [], color='k', ls='--', lw=1.0, label='True MPP')]
    if any(('scan' in logs[c] and logs[c].scan.sum()) for c in ctr): handles.append(Line2D([], [], color='k', marker='v', ls='none', ms=4, label='Scan cycle'))
    fig.tight_layout(rect=(0, 0.07, 1, 1)); sty.legend_fig(fig, handles=handles, ncol=min(len(handles), 5), y=-0.005)
    d = pd.concat(data, ignore_index=True); d.attrs['zoom'] = zoom
    sty.save_figure(fig, name, d); return d
