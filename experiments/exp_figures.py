"""Step 7 - all data/result figures in the common MATLAB-style (mppt_style.py), regenerated from the CSV/pickles written by the experiments:
Fig10-Fig18 data, PV model, DNN; Fig19-Fig24 time-domain, cold start, shading snapshots; Fig25-Fig26 headline bars; Fig35-Fig42 extras. Plotted data -> results/figures/data/<name>.csv."""
import os, sys, pickle, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mppt_style as sty
from mppt.config import CFG, RESULTS, CACHE
from mppt.figures import time_domain

LBL = {'G': 'Irradiance (W/m$^2$)', 'T': 'Ambient temperature (°C)', 'V': 'Voltage (V)', 'P': 'Power (W)'}
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def _fig(nrows=1, ncols=1, h=2.8, w=None, **kw):
    sty.apply_style(); return plt.subplots(nrows, ncols, figsize=(w or sty.W2, h), **kw)


def data_figures(pv, pvm):
    day = pv['G(i)'] >= CFG['g_min']; d_ = pv[day]
    fig, ax = _fig(1, 3, 2.7); bins = np.arange(0, 1125, 25)
    h1, _ = np.histogram(pv['G(i)'], bins); ax[0].bar(bins[:-1] + 12.5, h1, 25, color=sty.MATLAB[0], edgecolor='k', lw=0.3); ax[0].set_yscale('log'); ax[0].set(xlabel=LBL['G'], ylabel='Hours (log scale)')
    h2, _ = np.histogram(pv.loc[day, 'G(i)'], bins); ax[1].bar(bins[:-1] + 12.5, h2, 25, color=sty.MATLAB[4], edgecolor='k', lw=0.3); ax[1].set(xlabel=LBL['G'], ylabel='Hours')
    tb = np.arange(-4, 52, 1.0); h3, _ = np.histogram(pv['T2m'], tb); ax[2].bar(tb[:-1] + 0.5, h3, 1.0, color=sty.MATLAB[1], edgecolor='k', lw=0.3); ax[2].set(xlabel=LBL['T'], ylabel='Hours')
    for a, l in zip(ax, 'abc'): sty.panel(a, l, loc='tr')
    fig.tight_layout(); sty.save_figure(fig, 'Fig10_irradiance_temperature_histograms', pd.concat([pd.DataFrame({'panel': 'a', 'bin_left': bins[:-1], 'hours': h1}), pd.DataFrame({'panel': 'b', 'bin_left': bins[:-1], 'hours': h2}), pd.DataFrame({'panel': 'c', 'bin_left': tb[:-1], 'hours': h3})]))
    fig, ax = _fig(1, 1, 3.4, sty.W1 + 1.0); cm = plt.get_cmap('twilight_shifted', 12)
    s = ax.scatter(d_['G(i)'], d_['T2m'], c=d_['month'], cmap=cm, s=2, alpha=0.5, vmin=0.5, vmax=12.5, rasterized=True); cb = fig.colorbar(s, ax=ax, ticks=range(1, 13)); cb.set_label('Month'); cb.ax.set_yticklabels(MONTHS, fontsize=7)
    ax.set(xlabel=LBL['G'], ylabel=LBL['T']); fig.tight_layout(); sty.save_figure(fig, 'Fig11_daytime_conditions_by_month', d_[['G(i)', 'T2m', 'month']].rename(columns={'G(i)': 'G', 'T2m': 'T'}).iloc[::5])
    hm = pv.pivot_table(index='month', columns='hour', values='G(i)', aggfunc='mean').reindex(columns=range(24))
    fig, ax = _fig(1, 1, 3.2, sty.W1 + 2.0); im = ax.pcolormesh(np.arange(25) - 0.5, np.arange(13) + 0.5, hm.values, cmap=sty.CMAP, shading='flat'); cb = fig.colorbar(im, ax=ax); cb.set_label('Mean irradiance (W/m$^2$)')
    ax.set(xlabel='Hour of day (UTC)', ylabel='Month', yticks=range(1, 13), xticks=range(0, 24, 3)); ax.grid(False); ax.invert_yaxis(); fig.tight_layout()
    sty.save_figure(fig, 'Fig12_hour_month_irradiance_map', hm.reset_index().melt(id_vars='month', var_name='hour', value_name='mean_G'))
    te = pv[pv.year > CFG['chrono']['calib_end']].copy(); te['date'] = te.t.dt.date; tot = te.groupby('date')['G(i)'].sum(); mon = pd.Series([d.month for d in tot.index], index=tot.index)
    days = {'Winter': tot[mon.isin([12, 1, 2])].idxmax(), 'Summer': tot[mon.isin([6, 7, 8])].idxmax()}; fig, ax = _fig(1, 2, 2.7); fd = []
    for a, (nm, d), l in zip(ax, days.items(), 'ab'):
        x = te[te.date == d]; a.plot(x.hour, x['G(i)'], '-o', color=sty.MATLAB[0], ms=3, label='$G$'); a.set(xlabel='Hour (UTC)', ylabel=LBL['G'], xlim=(0, 23), ylim=(0, 1200)); b = a.twinx(); b.plot(x.hour, x['T2m'], '-s', color=sty.MATLAB[1], ms=3, label='$T$'); b.set_ylabel(LBL['T']); b.grid(False)
        h1_, l1_ = a.get_legend_handles_labels(); h2_, l2_ = b.get_legend_handles_labels(); a.legend(h1_ + h2_, l1_ + l2_, loc='upper left'); sty.panel(a, l, loc='tr'); fd.append(x[['hour', 'G(i)', 'T2m']].assign(day=str(d), season=nm))
    fig.tight_layout(); sty.save_figure(fig, 'Fig13_representative_days', pd.concat(fd))
    fig, ax = _fig(1, 3, 3.0, sharey=True); fd = []; amb = lambda G_, Tc_: Tc_ - (pvm.ds['NOCT'] - 20) / 800 * G_
    for a, Tc, l in zip(ax, (25.0, 45.0, 65.0), 'abc'):
        for k, g in enumerate((200, 400, 600, 800, 1000)):
            voc = float(pvm.voc(g, Tc)); V = np.linspace(0, voc, 300); P_ = V * pvm.current(g, Tc, V); a.plot(V, P_, color=sty.MATLAB[k], label=f'{g}'); i = P_.argmax(); a.plot(V[i], P_[i], 'k.', ms=6); fd.append(pd.DataFrame({'panel': l, 'Tcell': Tc, 'G': g, 'V': V, 'P': P_}))
        a.set(xlabel=LBL['V'], xlim=(0, 36), ylim=(0, 270)); sty.panel(a, l, loc='tr')
    ax[0].set_ylabel(LBL['P']); ax[0].legend(title='$G$ (W/m$^2$)', loc='upper left', fontsize=7, title_fontsize=7); fig.tight_layout(); sty.save_figure(fig, 'Fig14_PV_curve_families', pd.concat(fd))
    Gg = np.linspace(50, 1200, 40); Tg = np.linspace(-5, 75, 40); fig, ax = _fig(1, 2, 2.8); fd = []
    for k, Tc in enumerate((25.0, 45.0, 65.0)):
        v = [pvm.mpp(g, amb(g, Tc))[0] for g in Gg]; ax[0].plot(Gg, v, color=sty.MATLAB[k], label=f'{Tc:.0f} °C'); fd.append(pd.DataFrame({'panel': 'a', 'series': f'Tcell={Tc:.0f}', 'x': Gg, 'Vmpp': v}))
    ax[0].set(xlabel=LBL['G'], ylabel='$V_{mpp}$ (V)'); ax[0].legend(title='Cell temp.', loc='lower right', fontsize=7, title_fontsize=7)
    for k, g in enumerate((200, 600, 1000)):
        v = [pvm.mpp(g, amb(g, t))[0] for t in Tg]; ax[1].plot(Tg, v, color=sty.MATLAB[k], label=f'{g}'); fd.append(pd.DataFrame({'panel': 'b', 'series': f'G={g}', 'x': Tg, 'Vmpp': v}))
    ax[1].set(xlabel='Cell temperature (°C)', ylabel='$V_{mpp}$ (V)'); ax[1].legend(title='$G$ (W/m$^2$)', loc='upper right', fontsize=7, title_fontsize=7)
    sty.panel(ax[0], 'a', loc='tr'); sty.panel(ax[1], 'b', loc='bl'); fig.tight_layout(); sty.save_figure(fig, 'Fig15_Vmpp_vs_G_and_T', pd.concat(fd))
    # sanity: P vs G and three days; STC validation
    fig, ax = _fig(1, 2, 2.8); ax[0].scatter(pv['G(i)'], pv.P, s=1.5, alpha=0.2, color=sty.MATLAB[0], rasterized=True); ax[0].set(xlabel=LBL['G'], ylabel='Module MPP power (W)')
    d0 = te[te.date == days['Summer']].t.iloc[0]; w = pv[(pv.t >= d0 - pd.Timedelta(hours=12)) & (pv.t < d0 + pd.Timedelta(hours=60))]; th = (w.t - w.t.iloc[0]).dt.total_seconds() / 3600
    ax[1].plot(th, w['G(i)'], color=sty.MATLAB[0]); ax[1].set(xlabel='Time (h)', ylabel=LBL['G']); b = ax[1].twinx(); b.plot(th, w.P, color=sty.MATLAB[1]); b.set_ylabel(LBL['P']); b.grid(False)
    sty.panel(ax[0], 'a', loc='tl'); sty.panel(ax[1], 'b', loc='tr'); fig.tight_layout(); sty.save_figure(fig, 'Fig43_sanity_P_vs_G', pd.concat([pv[['G(i)', 'P']].iloc[::10].assign(panel='a'), pd.DataFrame({'t_h': th, 'G': w['G(i)'].to_numpy(), 'P': w.P.to_numpy(), 'panel': 'b'})]))
    ds = pvm.ds; v_stc, p_stc, i_stc = pvm.mpp(1000.0, amb(1000.0, 25.0)); voc_m = float(pvm.voc(1000.0, 25.0)); V = np.linspace(0, voc_m, 400); I_ = pvm.current(1000.0, 25.0, V); fig, ax = _fig(1, 2, 2.8)
    ax[0].plot(V, I_, color=sty.MATLAB[0], label='Model'); ax[0].plot([0, ds['Voc']], [ds['Isc'], 0], 's', mfc='none', mec=sty.MATLAB[6], ms=8, mew=1.5, label='Datasheet $I_{sc}$, $V_{oc}$'); ax[0].plot(v_stc, i_stc, 'k.', ms=7, label='Model MPP'); ax[0].set(xlabel=LBL['V'], ylabel='Current (A)'); ax[0].legend(loc='lower left', fontsize=7)
    ax[1].plot(V, V * I_, color=sty.MATLAB[0], label='Model'); ax[1].plot(v_stc, ds['Pmax'], 's', mfc='none', mec=sty.MATLAB[6], ms=8, mew=1.5, label='Datasheet $P_{max}$'); ax[1].plot(v_stc, p_stc, 'k.', ms=7, label='Model $P_{max}$'); ax[1].set(xlabel=LBL['V'], ylabel=LBL['P']); ax[1].legend(loc='lower center', fontsize=7)
    sty.panel(ax[0], 'a', loc='tr'); sty.panel(ax[1], 'b', loc='tl'); fig.tight_layout(); sty.save_figure(fig, 'Fig44_STC_validation', pd.concat([pd.DataFrame({'panel': 'a', 'V': V, 'I': I_}), pd.DataFrame({'panel': 'b', 'V': V, 'P': V * I_})]))


def dnn_figures(data, hyb):
    from mppt import models as MD
    h = hyb.history; fig, ax = _fig(1, 1, 2.7, sty.W1 + 0.8)
    ax.semilogy(np.arange(1, len(h['loss']) + 1), h['loss'], color=sty.MATLAB[0], label='Training'); ax.semilogy(np.arange(1, len(h['val_loss']) + 1), h['val_loss'], color=sty.MATLAB[1], ls='--', label='Validation (2015–16)')
    ax.set(xlabel='Epoch', ylabel='MSE (standardised target)'); ax.legend(); fig.tight_layout(); sty.save_figure(fig, 'Fig16_dnn_loss_curves', pd.DataFrame({'epoch': np.arange(1, len(h['loss']) + 1), 'train_loss': h['loss'], 'val_loss': h['val_loss']}))
    d = data['T1']; y = d['V_gmpp']; yh = np.clip(hyb.mlp.predict(MD.hybrid_features(d)), CFG['v_min_op'], CFG['v_max_op']); shd = d['shaded'] if 'shaded' in d else np.zeros(len(y), bool)
    fig, ax = _fig(1, 2, 2.9); ax[0].scatter(y[~shd], yh[~shd], s=1.5, alpha=0.25, color=sty.MATLAB[0], rasterized=True, label='Unshaded'); ax[0].scatter(y[shd], yh[shd], s=1.5, alpha=0.25, color=sty.MATLAB[1], rasterized=True, label='Partially shaded')
    lim = [min(y.min(), yh.min()), max(y.max(), yh.max())]; ax[0].plot(lim, lim, 'k--', lw=1); ax[0].set(xlabel='True $V_{gmpp}$ (V)', ylabel='Estimated $V_{gmpp}$ (V)'); ax[0].legend(markerscale=4, loc='lower right', fontsize=7)
    r = yh - y; ax[1].hist(r, bins=80, color=sty.MATLAB[0], edgecolor='k', lw=0.3); ax[1].axvline(0, color='k', lw=0.8); ax[1].set(xlabel='Error (estimated − true) (V)', ylabel='Samples')
    sty.panel(ax[0], 'a', loc='tl'); sty.panel(ax[1], 'b', loc='tr'); fig.tight_layout(); sty.save_figure(fig, 'Fig17_dnn_test_accuracy', pd.concat([pd.DataFrame({'panel': 'a', 'V_true': y, 'V_est': yh, 'shaded': shd}).iloc[::4], pd.DataFrame({'panel': 'b', 'error_V': r}).iloc[::4]]))
    lv = np.linspace(0.5, 0.99, 50); pred = lambda dd: np.clip(hyb.mlp.predict(MD.hybrid_features(dd)), CFG['v_min_op'], CFG['v_max_op']); rc = np.abs(data['calib']['V_gmpp'] - pred(data['calib'])); n = len(rc)
    fig, ax = _fig(1, 1, 3.0, sty.W1 + 1.1); fd = []
    for k, (key, lab) in enumerate((('T1', 'T1 in-distribution'), ('T2', 'T2 nominal, unshaded'), ('T3', 'T3 nominal, shaded'), ('T4', 'T4 out-of-range'))):
        rt = np.abs(data[key]['V_gmpp'] - pred(data[key])); cov = np.array([np.mean(rt <= np.quantile(rc, min(1.0, np.ceil((n + 1) * l) / n), method='higher')) for l in lv]); ax.plot(lv, cov, color=sty.MATLAB[k], label=lab); fd.append(pd.DataFrame({'set': lab, 'nominal': lv, 'coverage': cov}))
    ax.plot([0.5, 1], [0.5, 1], 'k--', lw=1, label='Ideal'); ax.set(xlabel='Nominal confidence $1-\\alpha$', ylabel='Empirical coverage', xlim=(0.5, 1), ylim=(0, 1.02)); ax.legend(loc='lower right', fontsize=7); fig.tight_layout(); sty.save_figure(fig, 'Fig18_conformal_calibration', pd.concat(fd))


ZOOM = {'step_change': (1780, 1850), 'cloud_passing': (900, 960), 'ramp_change': (1800, 1860), 'comprehensive': (1500, 1560), 'partial_shading': (390, 450), 'real_clear_day': (1700, 1760), 'real_variable_day': (1700, 1760)}
TD = ['P&O', 'IncCond', 'Poly2(G,T)-MPPT', 'ANN-MPPT', 'Scan+P&O (no ML)', 'Hybrid (DNN+P&O)']
NAMES = {'step_change': 'Fig19_timedomain_step_change', 'cloud_passing': 'Fig20_timedomain_cloud_passing', 'ramp_change': 'Fig21_timedomain_ramp_change', 'partial_shading': 'Fig22_timedomain_partial_shading',
         'comprehensive': 'Fig37_timedomain_comprehensive', 'real_clear_day': 'Fig38_timedomain_real_clear_day', 'real_variable_day': 'Fig39_timedomain_real_variable_day'}


def time_figures(L, pvm):
    for s, name in NAMES.items():
        zl = ZOOM[s]; time_domain({c: L['logs']['nominal'][s][c] for c in TD}, name, zoom=(zl[0], zl[1]), controllers=TD)
    for s, name in (('comprehensive', 'Fig40_timedomain_comprehensive_mismatched'), ('partial_shading', 'Fig41_timedomain_partial_shading_mismatched'), ('real_clear_day', 'Fig42_timedomain_real_clear_day_mismatched')):
        zl = ZOOM[s]; time_domain({c: L['logs']['mismatched'][s][c] for c in TD}, name, zoom=(zl[0], zl[1]), controllers=TD)
    cs = L['cold']; Bs = sorted(v for (p, v, c) in cs if p == 'B'); picks = [Bs[0], Bs[-1]]; sty.apply_style(); fig, ax = plt.subplots(3, 2, figsize=(sty.W2, 5.6)); fd = []; letters = iter('abcdef')
    for j, v0 in enumerate(picks):
        ref = cs[('B', v0, TD[0])]; x = np.arange(60)
        for r, (key, truth, yl) in enumerate((('P', 'mpp_power', 'Power (W)'), ('V', 'mpp_voltage', 'PV voltage (V)'), ('duty', None, 'Duty cycle (-)'))):
            a = ax[r, j]; a.plot(x, (ref[truth].to_numpy() if truth else 1 - ref.mpp_voltage.to_numpy() / CFG['v_bus'])[:60], 'k--', lw=1.0)
            for c in TD:
                df = cs[('B', v0, c)]; a.plot(x, df[key].to_numpy()[:60], color=sty.COLORS[c], marker=sty.MARKERS[c], ms=3, markevery=4, lw=1.1); fd.append(pd.DataFrame({'V0': v0, 'cycle': x, 'controller': c, 'row': key, 'value': df[key].to_numpy()[:60]}))
            a.set_ylabel(yl if j == 0 else ''); a.set_xlabel('Control cycle' if r == 2 else ''); sty.panel(a, next(letters), loc='br' if r == 0 else 'tr')
    hs = [Line2D([], [], color=sty.COLORS[c], marker=sty.MARKERS[c], ms=4, label=sty.SHORT[c]) for c in TD] + [Line2D([], [], color='k', ls='--', label='True MPP')]
    fig.tight_layout(rect=(0, 0.06, 1, 1)); sty.legend_fig(fig, hs, None, ncol=4, y=-0.005); sty.save_figure(fig, 'Fig23_cold_start', pd.concat(fd))
    sc = L['scenarios']['partial_shading']; from pv_single_diode import ShadedString
    shm = ShadedString(pvm, (CFG['v_min_op'], CFG['v_max_op'])); Ls = L['logs']['nominal']['partial_shading']; nep = len(sc['events']); nr = -(-nep // 3); fig, axes = plt.subplots(nr, 3, figsize=(sty.W2, 2.07 * nr), squeeze=False); fd = []
    for k in range(nr * 3, nr * 3): pass
    for k in range(nep, nr * 3): axes[k // 3, k % 3].axis('off')
    for k in range(nep):
        a = axes[k // 3, k % 3]; e1 = sc['events'][k + 1] if k + 1 < nep else len(sc['time']); pat = sc['patterns'][k]; I = np.linspace(0, 8.4, 3000); V = shm.string_voltage(I, pat, 25.0); m = V > 0
        a.plot(V[m], (V * I)[m], color='k', lw=1.2); vg, pg, _ = shm.mpp(pat, 25.0); a.plot(vg, pg, 'kx', ms=9, mew=1.8); a.axvspan(CFG['v_min_op'], CFG['v_max_op'], color='0.93', lw=0, zorder=0); fd.append(pd.DataFrame({'episode': k + 1, 'V': V[m], 'P': (V * I)[m]}))
        for c in TD: a.plot(Ls[c].V.to_numpy()[e1 - 1], Ls[c].P.to_numpy()[e1 - 1], marker=sty.MARKERS[c], color=sty.COLORS[c], mec='k', mew=0.5, ms=6, ls='none')
        a.set_xlim(0, 36); a.set_xlabel(LBL['V'] if k // 3 == nr - 1 else ''); a.set_ylabel(LBL['P'] if k % 3 == 0 else ''); sty.panel(a, 'abcdefghi'[k], loc='tl')
    hs = [Line2D([], [], color='k', label='P-V curve'), Line2D([], [], marker='x', color='k', ls='none', label='Global MPP')] + [Line2D([], [], marker=sty.MARKERS[c], color=sty.COLORS[c], mec='k', ls='none', ms=5, label=sty.SHORT[c]) for c in TD]
    fig.tight_layout(rect=(0, 0.06, 1, 1)); sty.legend_fig(fig, hs, None, ncol=4, y=-0.005); sty.save_figure(fig, 'Fig24_shading_pv_snapshots', pd.concat(fd))


def bar_figures():
    R = RESULTS; ps = pd.read_csv(os.path.join(R, 'stats', 'per_seed_scenario_metrics.csv')); order = ['comprehensive', 'step_change', 'cloud_passing', 'ramp_change', 'real_clear_day', 'real_variable_day', 'partial_shading']
    lab = ['Compreh.', 'Step', 'Cloud', 'Ramp', 'Clear day', 'Var. day', 'Shading']; ctr = [c for c in sty.COLORS if c in set(ps.controller)]
    def bars(metric, plants, name, ylabel, ylim=None, logy=False):
        fig, axes = plt.subplots(len(plants), 1, figsize=(sty.W2, 2.5 * len(plants)), squeeze=False); fd = []
        for ax, plant, l in zip(axes[:, 0], plants, 'ab'):
            g = ps[ps.plant == plant].groupby(['scenario', 'controller'])[metric].agg(['mean', 'std']).reset_index()
            series = {sty.SHORT[c]: ([g[(g.scenario == s) & (g.controller == c)]['mean'].iloc[0] for s in order], [g[(g.scenario == s) & (g.controller == c)]['std'].iloc[0] for s in order]) for c in ctr}
            fd.append(sty.grouped_bars(ax, lab, series, {sty.SHORT[c]: sty.COLORS[c] for c in ctr}, ylim=ylim, label_values=False, break_axis=bool(ylim and ylim[0] > 0)).assign(plant=plant)); ax.set_ylabel(ylabel)
            if logy: ax.set_yscale('log')
            sty.panel(ax, l, loc='bl', dy=0.16)
        h, l_ = axes[0, 0].get_legend_handles_labels(); fig.tight_layout(rect=(0, 0.06, 1, 1)); sty.legend_fig(fig, h, l_, ncol=5, y=-0.005); sty.save_figure(fig, name, pd.concat(fd))
    bars('tracking_efficiency_%', ['nominal', 'mismatched'], 'Fig25_tracking_efficiency', 'Tracking efficiency (%)', ylim=(70, 100.5))
    fig, axes = plt.subplots(2, 1, figsize=(sty.W2, 5.0)); fd = []
    for ax, (metric, yl, sc_list, lb) in zip(axes, (('convergence_cycles', 'Convergence (control cycles)', ['step_change', 'comprehensive', 'cloud_passing', 'ramp_change'], ['Step', 'Compreh.', 'Cloud', 'Ramp']), ('osc_amp_V', 'Steady-state oscillation p-p (V)', ['comprehensive', 'step_change', 'cloud_passing', 'ramp_change'], ['Compreh.', 'Step', 'Cloud', 'Ramp']))):
        g = ps[ps.plant == 'nominal'].groupby(['scenario', 'controller'])[metric].agg(['mean', 'std']).reset_index()
        series = {sty.SHORT[c]: ([g[(g.scenario == s) & (g.controller == c)]['mean'].iloc[0] for s in sc_list], [g[(g.scenario == s) & (g.controller == c)]['std'].iloc[0] for s in sc_list]) for c in ctr}
        fd.append(sty.grouped_bars(ax, lb, series, {sty.SHORT[c]: sty.COLORS[c] for c in ctr}, label_values=False).assign(metric=metric)); ax.set_ylabel(yl)
    sty.panel(axes[0], 'a', loc='tr'); sty.panel(axes[1], 'b', loc='tr'); h, l_ = axes[0].get_legend_handles_labels(); fig.tight_layout(rect=(0, 0.06, 1, 1)); sty.legend_fig(fig, h, l_, ncol=5, y=-0.005); sty.save_figure(fig, 'Fig26_convergence_oscillation', pd.concat(fd))
    hl = pd.read_csv(os.path.join(R, 'metrics_headline.csv')); c2 = [c for c in sty.COLORS if c in set(hl.controller)]; fig, ax = plt.subplots(figsize=(sty.W2, 2.7)); x = np.arange(len(c2))
    ax.bar(x, [hl[hl.controller == c].compute_us_per_cycle.iloc[0] for c in c2], 0.65, color=[sty.COLORS[c] for c in c2], edgecolor='k', lw=0.6); ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels([sty.SHORT[c] for c in c2], fontsize=8)
    ax.set_ylabel('Controller time per cycle (µs)'); fig.tight_layout(); sty.save_figure(fig, 'Fig35_compute_cost', hl[['controller', 'compute_us_per_cycle']])
    mm = pd.read_csv(os.path.join(R, 'mismatch_sweep.csv')); fig, axes = plt.subplots(1, 2, figsize=(sty.W2, 2.9)); fd = []
    for ax, (sname, l, lo) in zip(axes, (('comprehensive', 'a', 98.0), ('partial_shading', 'b', 70.0))):
        for c in c2:
            q = mm[(mm.scenario == sname) & (mm.controller == c)].sort_values('severity'); ax.plot(q.severity, q['tracking_efficiency_%'], color=sty.COLORS[c], marker=sty.MARKERS[c], ms=4, label=sty.SHORT[c]); fd.append(q[['scenario', 'controller', 'severity', 'tracking_efficiency_%']])
        ax.set(xlabel='Plant mismatch severity (-)', ylabel='Tracking efficiency (%)'); sty.panel(ax, l, loc='bl')
    h, l_ = axes[0].get_legend_handles_labels(); fig.tight_layout(rect=(0, 0.1, 1, 1)); sty.legend_fig(fig, h, l_, ncol=5, y=-0.005); sty.save_figure(fig, 'Fig36_mismatch_sweep', pd.concat(fd))


def run():
    from mppt.data import get_pv_series
    from mppt import models as MD
    from experiments.exp_main import get_models
    from mppt import schematics
    pv, pvm, _ = get_pv_series(); data, _c = MD.hybrid_data(pv); ann, poly, hyb = get_models(pv, pvm, data)
    schematics.build_all(); data_figures(pv, pvm); dnn_figures(data, hyb)
    L = pickle.load(open(os.path.join(CACHE, 'main_logs.pkl'), 'rb')); time_figures(L, pvm); bar_figures()


if __name__ == '__main__':
    run()
