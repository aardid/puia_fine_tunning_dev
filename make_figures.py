"""
Figure pack for the fine-tuning study. Reads the phase result artifacts and
renders five publication figures into figures/ (PNG + PDF).

  F1  Source-pool landscape per target (15 subsets), WIZ-containing vs WIZ-free
  F2  External validation: curated {K,O,S} vs all-5 model on 8 unseen volcanoes
  F3  Mechanism: feature composition with/without WIZ (streams + families)
  F4  Phase G pseudo-prospective scoreboard
  F5  Adaptation gain vs number of target eruptions (the decision ladder)

Colors: validated reference categorical palette, fixed slot order
(blue #2a78d6, orange #eb6834, aqua #1baf7a, yellow #eda100); chart chrome per
the same reference (ink #0b0b0b / #52514e, grid #e1e0d9, baseline #c3c2b7,
surface #fcfcfb). Sub-3:1 slots (aqua, yellow) always carry direct labels.
"""

import os
import re
from glob import glob
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(REPO, 'figures')
os.makedirs(FIG, exist_ok=True)

C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'   # fixed slots
INK, INK2, MUT = '#0b0b0b', '#52514e', '#898781'
GRID, BASE, SURF = '#e1e0d9', '#c3c2b7', '#fcfcfb'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'Arial', 'DejaVu Sans'],
    'figure.facecolor': SURF, 'axes.facecolor': SURF,
    'savefig.facecolor': SURF,
    'text.color': INK, 'axes.labelcolor': INK2,
    'xtick.color': MUT, 'ytick.color': MUT,
    'axes.edgecolor': BASE, 'axes.linewidth': 0.8,
    'axes.grid': False, 'font.size': 9.5,
})


def style_ax(ax, xgrid=False, ygrid=False):
    ax.spines[['top', 'right']].set_visible(False)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, lw=0.7)
        ax.set_axisbelow(True)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, lw=0.7)
        ax.set_axisbelow(True)


def save(fig, name):
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(FIG, f'{name}.{ext}'), dpi=200,
                    bbox_inches='tight')
    plt.close(fig)
    print(f'saved {name}')


STA_INITIAL = {'WIZ': 'W', 'FWVZ': 'F', 'KRVZ': 'K', 'ONTA': 'O', 'SHW': 'S'}


def pool_of_variant(target, variant):
    sources = [s for s in ['WIZ', 'FWVZ', 'KRVZ', 'ONTA', 'SHW'] if s != target]
    if variant == 'full':
        return sources
    if variant.startswith('no_'):
        return [s for s in sources if s != variant[3:]]
    return variant.replace('cur_', '').split('_')


def pool_label(pool):
    order = ['WIZ', 'FWVZ', 'KRVZ', 'ONTA', 'SHW']
    return '+'.join(STA_INITIAL[s] for s in order if s in pool)


# ============================================================
# F1 — pool landscape
# ============================================================
def fig1():
    panels = []
    for target, f in [('FWVZ', r'forecasts\phase_b\ablation_FWVZ.csv'),
                      ('KRVZ', r'forecasts\phase_b\ablation_KRVZ.csv')]:
        df = pd.read_csv(os.path.join(REPO, f))
        df['pool'] = df.variant.map(lambda v: pool_of_variant(target, v))
        panels.append((target, df))
    dfw = pd.read_csv(os.path.join(REPO, r'forecasts\phase_b_wiz\ablation_WIZ.csv'))
    dfw['pool'] = dfw.sources.map(lambda s: s.split('+'))
    panels.append(('WIZ', dfw))

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6))
    names = {'FWVZ': 'Ruapehu (FWVZ)', 'KRVZ': 'Tongariro (KRVZ)',
             'WIZ': 'Whakaari (WIZ)'}
    for ax, (target, df) in zip(axes, panels):
        df = df.copy()
        df['label'] = df.pool.map(pool_label)
        df['wiz'] = df.pool.map(lambda p: 'WIZ' in p)
        df = df.sort_values('auc').reset_index(drop=True)
        colors = [C2 if w else C1 for w in df.wiz]
        ax.hlines(range(len(df)), 0.4, df.auc, color=GRID, lw=0.7, zorder=1)
        ax.scatter(df.auc, range(len(df)), c=colors, s=42, zorder=3)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df.label, fontsize=8)
        # direct-label the best pool
        best = df.iloc[-1]
        ax.annotate(f'{best.auc:.3f}', (best.auc, len(df) - 1),
                    xytext=(4, 0), textcoords='offset points',
                    va='center', fontsize=8.5, color=INK)
        ax.set_xlim(0.4, 1.02)
        ax.set_title(f'target: {names[target]}', loc='left', fontsize=10,
                     color=INK)
        style_ax(ax, xgrid=True)
        ax.set_xlabel('eruption AUC (target fully out-of-sample)')
    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([], [], marker='o', ls='', color=C1, label='pool without Whakaari'),
        Line2D([], [], marker='o', ls='', color=C2, label='pool contains Whakaari')],
        loc='upper right', bbox_to_anchor=(0.995, 1.10), frameon=False,
        fontsize=9, ncol=2)
    fig.suptitle('Every Whakaari-free pool beats every Whakaari-containing pool',
                 x=0.005, ha='left', fontsize=12.5, fontweight='bold', y=1.10)
    fig.text(0.005, 1.03,
             'AUC of all 15 source-pool subsets, leave-target-volcano-out '
             '(W=Whakaari, F=Ruapehu, K=Tongariro, O=Ontake, S=St Helens)',
             fontsize=9, color=INK2)
    fig.tight_layout()
    save(fig, 'F1_pool_landscape')


# ============================================================
# F2 — external validation
# ============================================================
def fig2():
    df = pd.read_csv(os.path.join(
        REPO, r'forecasts\phase_b_external\external_validation.csv'))
    piv = df.pivot(index='target', columns='ensemble', values='auc')
    nev = df.groupby('target').n_events.first()
    piv = piv.sort_values('KOS')
    names = {'PN7A': 'Pavlof (PN7A)', 'PVV': 'Pavlof (PVV)',
             'VNSS': 'Veniaminof', 'BELO': 'Bezymianny', 'COP': 'Copahue',
             'MBGH': 'Montserrat', 'VRLE': 'VRLE', 'VTUN': 'VTUN'}

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    y = np.arange(len(piv))
    for i, (t, r) in enumerate(piv.iterrows()):
        ax.plot([r['all5'], r['KOS']], [i, i], color=GRID, lw=2, zorder=1)
    ax.scatter(piv['all5'], y, color=C2, s=52, zorder=3,
               label='all 5 sources (generalized model)')
    ax.scatter(piv['KOS'], y, color=C1, s=52, zorder=3,
               label='curated pool K+O+S (no Whakaari)')
    ax.set_yticks(y)
    ax.set_yticklabels([f'{names.get(t, t)}  (n={nev[t]})' for t in piv.index],
                       fontsize=9)
    style_ax(ax, xgrid=True)
    ax.set_xlabel('eruption AUC — 31 eruptions never seen by any model')
    ax.set_xlim(0.1, 1.0)
    ax.legend(loc='upper left', frameon=False, fontsize=8.5)
    ax.set_title('The curated pool wins on 8 of 8 out-of-pool volcanoes',
                 loc='left', fontsize=12.5, fontweight='bold', pad=14)
    ax.text(0, 1.015, 'External validation: models applied unchanged to '
            'volcanoes outside the training pool', transform=ax.transAxes,
            fontsize=9, color=INK2)
    fig.tight_layout()
    save(fig, 'F2_external_validation')


# ============================================================
# F3 — feature mechanism
# ============================================================
AMP = {'fft_coefficient', 'change_quantiles', 'quantile', 'abs_energy',
       'standard_deviation', 'variance', 'root_mean_square', 'maximum',
       'minimum', 'mean_abs_change', 'absolute_sum_of_changes',
       'sum_values', 'mean', 'median'}
SHAPE = {'autocorrelation', 'agg_autocorrelation', 'partial_autocorrelation',
         'cwt_coefficients', 'ar_coefficient', 'fft_aggregated',
         'number_peaks', 'number_cwt_peaks', 'symmetry_looking'}


def tally(ens_dir):
    streams, fams = Counter(), Counter()
    n = 0
    for f in glob(os.path.join(ens_dir, '*.fts')):
        for ln in open(f):
            ft = ' '.join(ln.rstrip().split()[1:])
            m = re.match(r'zsc2_(\w+?)F?__(\w+?)(__|$)', ft)
            if not m:
                continue
            n += 1
            streams[m.group(1)] += 1
            calc = m.group(2)
            fams['amplitude / energy' if calc in AMP else
                 'temporal structure' if calc in SHAPE else 'other'] += 1
    return streams, fams, n


def fig3():
    cache = os.path.join(FIG, 'feature_tallies.csv')
    ens = {
        'W+K+O+S\n(with Whakaari)': r'models\phase_b\FWVZ__full',
        'all 5 sources': r'models\cve_WIZ_FWVZ_KRVZ_ONTA_SHW\00',
        'K+O+S\n(no Whakaari)': r'models\phase_b\FWVZ__no_WIZ',
        'O+S\n(star pool)': r'models\phase_b\FWVZ__cur_ONTA_SHW',
    }
    if os.path.isfile(cache):
        df = pd.read_csv(cache, index_col=0)
    else:
        rows = {}
        for name, d in ens.items():
            streams, fams, n = tally(os.path.join(REPO, d))
            rows[name] = {**{f's_{k}': v / n for k, v in streams.items()},
                          **{f'f_{k}': v / n for k, v in fams.items()}}
            print(f'  tallied {name.splitlines()[0]} ({n} features)')
        df = pd.DataFrame(rows).T
        df.to_csv(cache)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    order = list(ens.keys())

    # (a) stream composition, stacked
    ax = axes[0]
    streams = ['rsam', 'mf', 'hf', 'dsar']
    scolors = [C1, C2, C3, C4]
    left = np.zeros(len(order))
    y = np.arange(len(order))[::-1]
    for s, c in zip(streams, scolors):
        vals = np.array([df.loc[k].get(f's_{s}', 0) for k in order])
        ax.barh(y, vals, left=left, color=c, height=0.62,
                edgecolor=SURF, linewidth=1.5, label=s.upper())
        for yi, (l, v) in enumerate(zip(left, vals)):
            if v > 0.07:
                ax.text(l + v / 2, y[yi], f'{v:.0%}', ha='center',
                        va='center', fontsize=8, color=SURF
                        if c in (C1, C2) else INK)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    style_ax(ax)
    ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.12),
              frameon=False, fontsize=8.5)
    ax.set_title('(a) seismic stream of selected features', loc='left',
                 fontsize=10, color=INK)

    # (b) feature families, grouped
    ax = axes[1]
    fams = ['amplitude / energy', 'temporal structure']
    fcolors = [C2, C1]
    w = 0.36
    x = np.arange(len(order))
    for j, (fam, c) in enumerate(zip(fams, fcolors)):
        vals = [df.loc[k].get(f'f_{fam}', 0) for k in order]
        b = ax.bar(x + (j - 0.5) * w, vals, width=w - 0.04, color=c,
                   edgecolor=SURF, linewidth=1.5, label=fam)
        for xi, v in zip(x + (j - 0.5) * w, vals):
            ax.text(xi, v + 0.015, f'{v:.0%}', ha='center', fontsize=8,
                    color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace('\n', ' ') for k in order], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    style_ax(ax, ygrid=True)
    ax.legend(loc='upper right', frameon=False, fontsize=8.5)
    ax.set_title('(b) feature family (share of 6000 selected features)',
                 loc='left', fontsize=10, color=INK)

    fig.suptitle('Why Whakaari poisons pools: loudness does not transfer, '
                 'waveform shape does', x=0.005, ha='left', fontsize=12.5,
                 fontweight='bold', y=1.04)
    fig.text(0.005, 0.965, 'With Whakaari in the pool, feature selection locks '
             'onto raw-amplitude statistics; without it, selection shifts to '
             'temporal-structure features that transfer between volcanoes',
             fontsize=9, color=INK2)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, 'F3_feature_mechanism')


# ============================================================
# F4 — Phase G prospective scoreboard
# ============================================================
def fig4():
    df = pd.read_csv(os.path.join(REPO, r'forecasts\phase_g\phase_g_summary.csv'))
    methods = ['full', 'curated', 'ser_us', 'tboost']
    mlabels = ['full pool', 'curated pool', 'refined trees (ser_us)',
               'target-boosted (tboost)']
    mcolors = [C1, C2, C3, C4]
    targets = ['FWVZ', 'KRVZ', 'WIZ']
    tlabels = ['Ruapehu\n3 eruptions', 'Tongariro\n2 eruptions',
               'Whakaari\n5 eruptions']

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(len(targets))
    w = 0.19
    for j, (m, ml, c) in enumerate(zip(methods, mlabels, mcolors)):
        vals = [df[(df.target == t) & (df.method == m)].auc.iloc[0]
                for t in targets]
        pos = x + (j - 1.5) * w
        ax.bar(pos, vals, width=w - 0.025, color=c, edgecolor=SURF,
               linewidth=1.5, label=ml)
        for xi, v in zip(pos, vals):
            ax.text(xi, v + 0.012, f'{v:.2f}', ha='center', fontsize=8,
                    color=INK2)
    # annotate tboost fallback-driven scores
    for t, xi in [('FWVZ', 0), ('KRVZ', 1)]:
        v = df[(df.target == t) & (df.method == 'tboost')].auc.iloc[0]
        ax.text(xi + 1.5 * w, v - 0.045, '*', ha='center', fontsize=13,
                color=SURF, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tlabels, fontsize=9.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('pseudo-prospective eruption AUC')
    style_ax(ax, ygrid=True)
    ax.legend(ncol=2, loc='upper left', frameon=False, fontsize=8.5)
    fig.suptitle('Prospective test: curation always helps; local boosting '
                 'needs eruption history', x=0.005, ha='left', fontsize=12.5,
                 fontweight='bold', y=1.06)
    fig.text(0.005, 0.985,
             'All target information restricted to before each eruption; '
             'scored on unrest-month windows.\n* score carried by the '
             'no-prior-eruption fallback — tboost itself detected nothing '
             'prospectively at these targets', fontsize=8.4, color=INK2,
             va='top')
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, 'F4_prospective_scoreboard')


# ============================================================
# F5 — adaptation gain vs eruption count
# ============================================================
def fig5():
    r1 = pd.read_csv(os.path.join(REPO, r'forecasts\phase_c\phase_c_results.csv'))
    r2 = pd.read_csv(os.path.join(
        REPO, r'forecasts\phase_c\phase_c_results_variants.csv'))
    rw = pd.read_csv(os.path.join(
        REPO, r'forecasts\phase_c\phase_c_results_WIZ.csv'))
    allr = pd.concat([r1, r2, rw])
    allr = allr[allr.base == 'full']
    base = {t: allr[(allr.target == t) &
                    (allr.method == 'base (no refinement)')].auc.iloc[0]
            for t in ['FWVZ', 'KRVZ', 'WIZ']}
    nerup = {'KRVZ': 2, 'FWVZ': 3, 'WIZ': 5}

    methods = ['strut', 'ser', 'strut_us', 'ser_us']
    mlabels = ['STRUT', 'SER', 'STRUT (per-tree undersample)',
               'SER (per-tree undersample)']
    mcolors = [C1, C2, C3, C4]

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.axhline(0, color=BASE, lw=1)
    ax.text(5.55, 0.006, 'helps', fontsize=8, color=INK2, va='bottom')
    ax.text(5.55, -0.012, 'harms', fontsize=8, color=INK2, va='top')
    for m, ml, c in zip(methods, mlabels, mcolors):
        xs, ys = [], []
        for t in ['KRVZ', 'FWVZ', 'WIZ']:
            row = allr[(allr.target == t) & (allr.method == m)]
            if len(row):
                xs.append(nerup[t])
                ys.append(row.auc.iloc[0] - base[t])
        ax.plot(xs, ys, color=c, lw=2, marker='o', ms=7, label=ml,
                markeredgecolor=SURF, markeredgewidth=1.2)
    ax.set_xticks([2, 3, 5])
    ax.set_xticklabels(['2\n(Tongariro)', '3\n(Ruapehu)', '5\n(Whakaari)'],
                       fontsize=9)
    ax.set_xlim(1.6, 5.95)
    ax.set_xlabel('recorded eruptions at the target volcano')
    ax.set_ylabel('AUC change from refining on target data')
    style_ax(ax, ygrid=True)
    ax.legend(loc='lower right', bbox_to_anchor=(0.99, 0.30), frameon=False,
              fontsize=8.5)
    ax.set_title('Fine-tuning is unreliable below five eruptions — and '
                 'consistently helps at five', loc='left', fontsize=12.5,
                 fontweight='bold', pad=14)
    ax.text(0, 1.015, 'Change in eruption AUC when the foreign-trained '
            'ensemble is refined on the target volcano’s own data '
            '(leave-one-eruption-out)', transform=ax.transAxes, fontsize=9,
            color=INK2)
    fig.tight_layout()
    save(fig, 'F5_decision_ladder')


# ============================================================
# F6 — prospective replay, eruption by eruption
# ============================================================
POOL_NICE = {'full': 'fallback', 'cur_WIZ_KRVZ': 'W+K', 'no_WIZ': 'K+O+S',
             'cur_ONTA_SHW': 'O+S', 'FK': 'F+K', 'F': 'F', 'OS': 'O+S'}


def fig6():
    ev = pd.read_csv(os.path.join(REPO, r'forecasts\phase_g\phase_g_events.csv'))
    methods = ['full', 'curated', 'ser_us', 'tboost']
    mlabels = ['full pool', 'curated pool', 'refined trees (ser_us)',
               'target-boosted (tboost)']
    mcolors = [C1, C2, C3, C4]
    targets = ['FWVZ', 'KRVZ', 'WIZ']
    tnames = {'FWVZ': 'Ruapehu', 'KRVZ': 'Tongariro', 'WIZ': 'Whakaari'}
    counts = [ev[ev.target == t].eruption.nunique() for t in targets]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.8),
                             gridspec_kw={'width_ratios': counts}, sharey=True)
    off = np.linspace(-0.27, 0.27, 4)
    for ax, target in zip(axes, targets):
        sub = ev[ev.target == target]
        erups = sorted(sub.eruption.unique())
        for e in erups:
            rows = sub[sub.eruption == e].set_index('method')
            fallback = rows.pre_q95.nunique() == 1 and e == 0
            if fallback:
                r = rows.loc['full']
                ax.plot([e, e], [r.bg_p95, r.pre_q95], color=MUT, lw=1.6,
                        zorder=2)
                ax.plot(e, r.bg_p95, marker='_', ms=11, color=INK2, zorder=3)
                ax.scatter(e, r.pre_q95, color=MUT, s=48, zorder=4)
                ax.annotate('fallback\n(no prior\neruptions)', (e, 0.03),
                            ha='center', fontsize=7, color=MUT)
                continue
            for m, c, o in zip(methods, mcolors, off):
                r = rows.loc[m]
                x = e + o
                ax.plot([x, x], [r.bg_p95, r.pre_q95], color=GRID, lw=1.6,
                        zorder=2)
                ax.plot(x, r.bg_p95, marker='_', ms=9, color=INK2, zorder=3)
                ax.scatter(x, r.pre_q95, color=c, s=44, zorder=4,
                           edgecolor=SURF, linewidth=0.8)
                if m == 'tboost' and r.pre_q95 == 0:
                    ax.annotate('silent', (x, 0.015), ha='center', fontsize=6.5,
                                color=INK2, rotation=90, va='bottom')
        # x labels: date + curated pick
        labels = []
        for e in erups:
            rows = sub[sub.eruption == e].set_index('method')
            te = rows.loc['full'].te
            pick = rows.loc['curated'].selected_pool
            pick = POOL_NICE.get(pick, str(pick).replace('cur_', ''))
            lab = te[:7]
            if not (e == 0):
                lab += f'\npick: {pick}'
            labels.append(lab)
        ax.set_xticks(erups)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_xlim(-0.6, max(erups) + 0.6)
        ax.set_ylim(0, 1.05)
        ax.set_title(tnames[target], loc='left', fontsize=10, color=INK)
        style_ax(ax, ygrid=True)
    axes[0].set_ylabel('consensus (q95)')

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker='o', ls='', color=c, label=l)
               for c, l in zip(mcolors, mlabels)]
    handles += [Line2D([], [], marker='o', ls='', color=MUT, label='fallback fold'),
                Line2D([], [], marker='_', ls='', color=INK2, ms=10,
                       label='window background q95')]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.995, 1.13),
               frameon=False, fontsize=8, ncol=3)
    fig.suptitle('Replaying history: every eruption forecast with only '
                 'prior information', x=0.005, ha='left', fontsize=12.5,
                 fontweight='bold', y=1.13)
    fig.text(0.005, 1.045,
             'Dot = pre-eruption consensus (q95, 2-day window); tick = that '
             'unrest month’s background q95. A dot above its tick means the '
             'precursor stood out\nfrom its own crisis month. "pick" = the pool '
             'the prospective selection chose using only earlier eruptions.',
             fontsize=8.6, color=INK2, va='top')
    fig.tight_layout()
    save(fig, 'F6_prospective_replay')


# ============================================================
# F7 — operational time series (prospective, "what the screen showed")
# ============================================================
def _master_window(path, t0, t1, col='consensus'):
    con = pd.read_pickle(path)[col]
    return con[(con.index >= t0) & (con.index <= t1)]


def fig7():
    import matplotlib.dates as mdates
    from datetime import datetime, timedelta

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.4))

    # --- (a) Whakaari, December 2019 ---
    ax = axes[0]
    te = datetime(2019, 12, 9, 1, 11)
    res = pd.read_pickle(os.path.join(REPO, r'forecasts\phase_g\WIZ__e4.pkl'))
    tb = res['tboost']
    t0, t1 = tb.index[0], tb.index[-1]
    wizyears = pd.concat([pd.read_pickle(os.path.join(
        REPO, rf'forecasts\phase_b_wiz\WIZ_{y}.pkl')) for y in (2019,)])
    full = wizyears['full'][(wizyears.index >= t0) & (wizyears.index <= t1)]
    cur = wizyears['OS'][(wizyears.index >= t0) & (wizyears.index <= t1)]
    for s, c, lb in [(full, C1, 'full pool'), (cur, C2, 'curated pool (O+S)')]:
        ax.plot(s.index, s.values, lw=0.4, color=c, alpha=0.30)
        m = s.rolling('12h').median()
        ax.plot(m.index, m.values, lw=1.8, color=c, label=lb)
    ax.plot(tb.index, tb.values, lw=1.2, color='#c98500',
            label='target-boosted (raw)')
    ax.axvline(te, color='#e34948', ls='--', lw=1.4)
    ax.axvspan(te - timedelta(days=2), te, color='#eda100', alpha=0.18)
    first = tb[(tb >= 0.8) & (tb.index < te)].index[0]
    ax.annotate('first alert\n3.4 days before eruption',
                xy=(first, 0.82), xytext=(first - timedelta(days=12), 0.86),
                fontsize=9, color=INK,
                arrowprops=dict(arrowstyle='->', color=INK2, lw=1))
    ax.text(te + timedelta(hours=8), 0.98, 'eruption\nDec 9, 2019',
            color='#e34948', fontsize=9, va='top')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('consensus')
    ax.set_title('(a) Whakaari, November–December 2019 — trained only on '
                 'data before this window', loc='left', fontsize=10.5,
                 color=INK)
    style_ax(ax, ygrid=True)
    ax.legend(loc='upper left', frameon=False, fontsize=8.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    # --- (b) Ruapehu, July 2009 ---
    ax = axes[1]
    te = datetime(2009, 7, 13, 6, 30)
    res = pd.read_pickle(os.path.join(REPO, r'forecasts\phase_g\FWVZ__e2.pkl'))
    ser = res['ser_us']
    t0, t1 = ser.index[0], ser.index[-1]
    full = _master_window(os.path.join(
        REPO, r'forecasts\phase_b\FWVZ__full\consensus_master.pkl'), t0, t1)
    cur = _master_window(os.path.join(
        REPO, r'forecasts\phase_b\FWVZ__no_WIZ\consensus_master.pkl'), t0, t1)
    for s, c, lb in [(full, C1, 'full pool'),
                     (cur, C2, 'curated pool (K+O+S)'),
                     (ser, C3, 'refined trees (ser_us)')]:
        ax.plot(s.index, s.values, lw=0.4, color=c, alpha=0.30)
        m = s.rolling('12h').median()
        ax.plot(m.index, m.values, lw=1.8, color=c, label=lb)
    ax.axvline(te, color='#e34948', ls='--', lw=1.4)
    ax.axvspan(te - timedelta(days=2), te, color='#eda100', alpha=0.18)
    ax.text(te + timedelta(hours=8), 0.98, 'eruption\nJul 13, 2009',
            color='#e34948', fontsize=9, va='top')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('consensus')
    ax.set_title('(b) Ruapehu, June–July 2009 — pool selected and trees '
                 'refined using only the 2006–2007 eruptions', loc='left',
                 fontsize=10.5, color=INK)
    style_ax(ax, ygrid=True)
    ax.legend(loc='upper left', frameon=False, fontsize=8.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    fig.suptitle('What the duty officer would have seen — forecasts using '
                 'only prior information', x=0.005, ha='left', fontsize=12.5,
                 fontweight='bold', y=1.005)
    fig.text(0.005, 0.965, 'Thin traces: raw 10-minute consensus; bold: '
             '12-hour rolling median; orange band: the 2-day alert window '
             'before each eruption.', fontsize=9, color=INK2)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, 'F7_operational_view')


# ============================================================
# F8 — sustained-alert rule: detection vs false alarms
# ============================================================
def fig8():
    df = pd.read_csv(os.path.join(REPO, r'forecasts\phase_b\alert_rule_table.csv'))
    targets = ['FWVZ', 'KRVZ', 'WIZ']
    tnames = {'FWVZ': 'Ruapehu (3 eruptions)', 'KRVZ': 'Tongariro (2 eruptions)',
              'WIZ': 'Whakaari (5 eruptions)'}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for ax, t in zip(axes, targets):
        sub = df[df.target == t]
        frac = sub.detected / sub.n_eruptions
        wiz = sub.pool.str.contains('WIZ|^W|full') if t != 'WIZ' \
            else pd.Series(False, index=sub.index)
        if t != 'WIZ':
            wiz = sub.pool.map(lambda p: 'WIZ' in pool_of_variant(t, p))
        colors = [C2 if w else C1 for w in wiz]
        ax.scatter(sub.false_alarms_per_year, frac, c=colors, s=48,
                   edgecolor=SURF, linewidth=0.8, zorder=3)
        # annotate only the standouts: best detection fraction, ties broken
        # by fewer false alarms / longer lead
        top = sub[sub.detected > 0].sort_values(
            ['detected', 'false_alarms_per_year'], ascending=[False, True])
        keep = top.head(2)
        if len(top) > 2:  # also flag the longest-lead detector if distinct
            lead_best = top.sort_values('mean_lead_days', ascending=False).head(1)
            keep = pd.concat([keep, lead_best]).drop_duplicates('pool')
        for k, (_, r) in enumerate(keep.iterrows()):
            nice = r.pool.replace('cur_', '').replace('no_', 'no ')
            dx = (-26, 26, 0)[k % 3]
            ax.annotate(f'{nice}\nlead {r.mean_lead_days:.0f}d',
                        (r.false_alarms_per_year, r.detected / r.n_eruptions),
                        xytext=(dx, 8), textcoords='offset points',
                        ha='center', fontsize=7.5, color=INK2)
        ax.set_xlim(0, 13)
        ax.set_ylim(-0.05, 0.62)
        ax.set_yticks([0, 0.2, 0.4, 0.6])
        ax.set_yticklabels(['0%', '20%', '40%', '60%'])
        ax.set_title(tnames[t], loc='left', fontsize=10, color=INK)
        ax.set_xlabel('false alarms per year')
        style_ax(ax, ygrid=True)
    axes[0].set_ylabel('eruptions detected (sustained alert in final 10 days)')
    fig.suptitle('A sustained-alert rule only works at Whakaari — short '
                 'precursors elsewhere need the 2-day window metric',
                 x=0.005, ha='left', fontsize=12.5, fontweight='bold', y=1.09)
    fig.text(0.005, 1.015,
             'Causal rule: alert when the 12-h median exceeds the trailing '
             '30-day q99 for 6+ hours. Blue = Whakaari-free pool, orange = '
             'contains Whakaari.', fontsize=9, color=INK2)
    fig.tight_layout()
    save(fig, 'F8_alert_rule_tradeoff')


# ============================================================
# F9 — case study: Whakaari Dec 2019, Ontake-pool sustained alert
# ============================================================
def fig9():
    import matplotlib.dates as mdates
    from datetime import datetime, timedelta
    te = datetime(2019, 12, 9, 1, 11)
    YTOP = 0.8           # focus on the meaningful consensus band

    # 2 years + 30-day threshold warm-up of the Ontake-only pool consensus
    ctx0 = te - timedelta(days=2 * 365.25)
    files = [os.path.join(REPO, rf'forecasts\phase_b_wiz\WIZ_{y}.pkl')
             for y in (2017, 2018, 2019)]
    df = pd.concat([pd.read_pickle(f) for f in files])
    df = df[~df.index.duplicated()].sort_index()
    o_all = df['O'][(df.index >= ctx0 - timedelta(days=32)) &
                    (df.index <= te + timedelta(days=4))]

    # causal alert rule (same as alert_rule_analysis.py): 12-h median above
    # the trailing 30-day q99 (shifted 1 h) for >= 6 consecutive hours
    o1h = o_all.resample('1h').median().dropna()
    med1h = o1h.rolling(12, min_periods=6).median()
    thr1h = o1h.rolling(720, min_periods=240).quantile(0.99).shift(1)
    active = (med1h > thr1h) & thr1h.notna()
    episodes, onset_, run = [], None, 0
    for t, a in active.items():
        if a:
            run += 1
            if run == 6:
                onset_ = t - timedelta(hours=5)
        else:
            if onset_ is not None:
                episodes.append((onset_, t))
                onset_ = None
            run = 0
    if onset_ is not None:
        episodes.append((onset_, active.index[-1]))
    episodes = [(a, b) for a, b in episodes if a >= ctx0]
    pre = [(a, b) for a, b in episodes if a < te and b > te - timedelta(days=10)]
    n_fa = len(episodes) - len(pre)

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.6))

    # --- (a) two-year context with every alert episode shaded ---
    ax = axes[0]
    ov = o_all[o_all.index >= ctx0]
    ax.plot(ov.index, ov.values, lw=0.3, color=C1, alpha=0.30)
    m = ov.rolling('5D').median()
    ax.plot(m.index, m.values, lw=1.5, color='#104281',
            label='Ontake-only pool, 5-day median')
    ax.plot(thr1h.index, thr1h.values, lw=1.0, ls=':', color=INK2,
            label='causal alert threshold (trailing 30-day q99)')
    for a, b in episodes:
        ax.axvspan(a, b, color='#eda100', alpha=0.35)
    ax.axvline(te, color='#e34948', ls='--', lw=1.5)
    ax.text(te - timedelta(days=18), YTOP * 0.97, 'eruption', color='#e34948',
            fontsize=9, va='top', ha='right')
    ax.set_xlim(ctx0, te + timedelta(days=4))
    ax.set_ylim(0, YTOP)
    ax.set_ylabel('consensus')
    style_ax(ax, ygrid=True)
    ax.legend(loc='upper left', frameon=False, fontsize=8.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.set_title(f'(a) the two years before: {len(episodes)} alert episodes '
                 f'(orange), {n_fa} of them false alarms '
                 f'(≈{n_fa/2:.0f} per year) — the last one precedes the '
                 'eruption', loc='left', fontsize=10.5, color=INK)

    # --- (b) final-month zoom (fixed Nov-background threshold, as before) ---
    ax = axes[1]
    w0 = te - timedelta(days=30.4)
    o = o_all[(o_all.index >= w0) & (o_all.index <= te + timedelta(days=4))]
    res = pd.read_pickle(os.path.join(REPO, r'forecasts\phase_g\WIZ__e4.pkl'))
    tb = res['tboost']
    bg = o[o.index < te - timedelta(days=10)]
    thr = bg.quantile(0.99)
    med = o.rolling('12h').median()
    above = (med[med.index >= te - timedelta(days=10)] > thr)
    run, onset = 0, None
    for t, a in above.items():
        run = run + 1 if a else 0
        if run >= 36:
            onset = t - timedelta(hours=6)
            break
    ax.plot(o.index, o.values, lw=0.4, color=C1, alpha=0.35)
    ax.plot(med.index, med.values, lw=1.9, color='#104281',
            label='Ontake-only pool, 12-h median')
    ax.plot(tb.index, tb.values, lw=1.1, color='#c98500',
            label='target-boosted (raw, prospective)')
    ax.axhline(thr, color=INK2, lw=1, ls=':',
               label=f'alert threshold (own Nov background q99 = {thr:.2f})')
    ax.axvline(te, color='#e34948', ls='--', lw=1.5)
    ax.axvspan(onset, te, color='#eda100', alpha=0.15)
    ax.annotate(f'sustained alert begins\n'
                f'{(te-onset).total_seconds()/86400:.1f} days before eruption',
                xy=(onset, thr + 0.02),
                xytext=(onset - timedelta(days=8), 0.70), fontsize=9.5,
                color=INK, arrowprops=dict(arrowstyle='->', color=INK2, lw=1))
    ax.text(te + timedelta(hours=8), YTOP * 0.97, 'eruption\nDec 9, 2019',
            color='#e34948', fontsize=9, va='top')
    ax.set_ylim(0, YTOP)
    ax.set_ylabel('consensus')
    style_ax(ax, ygrid=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.legend(loc='upper left', frameon=False, fontsize=9)
    ax.set_title('(b) the final month', loc='left', fontsize=10.5, color=INK)

    fig.suptitle('Whakaari, December 2019: a nine-day sustained alert — and '
                 'what the same rule costs in false alarms', x=0.005,
                 ha='left', fontsize=12.5, fontweight='bold', y=1.005)
    fig.text(0.005, 0.968,
             'Ontake-only pool consensus. Pool highlighted retrospectively; '
             'the target-boosted trace in (b) is fully prospective.',
             fontsize=8.6, color=INK2, va='top')
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    save(fig, 'F9_wiz2019_case_study')


# ============================================================
# F10 — optimal alert rule + operating curve
# ============================================================
def fig10():
    import matplotlib.dates as mdates
    from datetime import datetime, timedelta
    te = datetime(2019, 12, 9, 1, 11)
    YTOP = 0.8
    Q, LB_D, MW_H, SU_H = 0.995, 180, 24, 24   # sweep optimum

    files = sorted(glob(os.path.join(REPO, r'forecasts\phase_b_wiz', 'WIZ_*.pkl')))
    df = pd.concat([pd.read_pickle(f) for f in files])
    df = df[~df.index.duplicated()].sort_index()
    con = df['O'].resample('1h').median().dropna()
    med = con.rolling(MW_H, min_periods=MW_H // 2).median()
    thr = con.rolling(LB_D * 24, min_periods=240).quantile(Q).shift(1)
    active = (med > thr) & thr.notna()
    episodes, onset, run = [], None, 0
    for t, a in active.items():
        if a:
            run += 1
            if run == SU_H:
                onset = t - timedelta(hours=SU_H - 1)
        else:
            if onset is not None:
                episodes.append((onset, t))
                onset = None
            run = 0
    if onset is not None:
        episodes.append((onset, active.index[-1]))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4),
                             gridspec_kw={'width_ratios': [2.1, 1]})

    # --- (a) two-year view under the optimal rule ---
    ax = axes[0]
    ctx0 = te - timedelta(days=2 * 365.25)
    eps2 = [(a, b) for a, b in episodes if b >= ctx0]
    ov = df['O'][(df.index >= ctx0) & (df.index <= te + timedelta(days=4))]
    ax.plot(ov.index, ov.values, lw=0.3, color=C1, alpha=0.30)
    m5 = ov.rolling('5D').median()
    ax.plot(m5.index, m5.values, lw=1.5, color='#104281',
            label='Ontake-only pool, 5-day median')
    tt = thr[(thr.index >= ctx0) & (thr.index <= te + timedelta(days=4))]
    ax.plot(tt.index, tt.values, lw=1.0, ls=':', color=INK2,
            label=f'optimal threshold (trailing {LB_D}-day q{Q})')
    # 10-day warning period following each alert trigger (lighter fill)
    for a, b in eps2:
        ax.axvspan(a, a + timedelta(days=10), color='#eda100', alpha=0.15,
                   lw=0)
    for a, b in eps2:
        ax.axvspan(a, b, color='#eda100', alpha=0.45, lw=0)
    ax.axvline(te, color='#e34948', ls='--', lw=1.5)
    ax.text(te - timedelta(days=18), YTOP * 0.97, 'eruption',
            color='#e34948', fontsize=9, va='top', ha='right')
    pre = [o for o, b in eps2 if o < te and b > te - timedelta(days=10)]
    if pre:
        lead = (te - min(pre)).total_seconds() / 86400
        ax.annotate(f'alert {lead:.1f} days\nbefore eruption',
                    xy=(min(pre), 0.62), xytext=(te - timedelta(days=300), 0.68),
                    fontsize=9, color=INK,
                    arrowprops=dict(arrowstyle='->', color=INK2, lw=1))
    nfa = len(eps2) - len(pre)
    ax.set_xlim(ctx0, te + timedelta(days=4))
    ax.set_ylim(0, YTOP)
    ax.set_ylabel('consensus')
    style_ax(ax, ygrid=True)
    from matplotlib.patches import Patch
    h, l = ax.get_legend_handles_labels()
    h += [Patch(facecolor='#eda100', alpha=0.45, label='alert episode'),
          Patch(facecolor='#eda100', alpha=0.15,
                label='10-day warning period after trigger')]
    ax.legend(handles=h, loc='upper left', frameon=False, fontsize=8.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.set_title(f'(a) the optimal rule over the same two years: '
                 f'{nfa} false alarm{"s" if nfa != 1 else ""}, and the '
                 'pre-eruption alert survives', loc='left', fontsize=10.5,
                 color=INK)

    # --- (b) operating curve from the sweep ---
    ax = axes[1]
    sw = pd.read_csv(os.path.join(REPO, r'forecasts\phase_b',
                                  'alert_rule_sweep.csv'))
    frac = sw.detected / 5.
    ax.scatter(sw.false_alarms_per_year, frac, s=18, color=MUT, alpha=0.45,
               edgecolor='none', label='all 240 rule settings', zorder=2)
    # Pareto frontier: fewest FA for each detection level
    front = sw.groupby('detected').false_alarms_per_year.min().reset_index()
    front = front[front.detected > 0].sort_values('detected')
    fx = list(front.false_alarms_per_year) + [13]
    fy = list(front.detected / 5.)
    ax.step(fx, fy + [fy[-1]], where='post', color=C2, lw=2,
            label='best achievable (frontier)', zorder=3)
    ax.scatter(front.false_alarms_per_year, front.detected / 5., color=C2,
               s=52, zorder=4, edgecolor=SURF, linewidth=0.8)
    # original heuristic + optimum
    ax.scatter([10.8], [0.4], color=C1, s=60, zorder=5, edgecolor=SURF,
               linewidth=0.8, label='original heuristic')
    ax.annotate('optimum for 2019\n(1.1 FA/yr, 8.8-d lead)',
                xy=(1.1, 0.2), xytext=(2.6, 0.06), fontsize=8, color=INK2,
                arrowprops=dict(arrowstyle='->', color=INK2, lw=0.9))
    ax.set_xlim(0, 13)
    ax.set_ylim(-0.04, 0.72)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_yticklabels(['0/5', '1/5', '2/5', '3/5'])
    ax.set_xlabel('false alarms per year')
    ax.set_ylabel('eruptions detected')
    style_ax(ax, ygrid=True)
    ax.legend(loc='lower right', frameon=False, fontsize=8)
    ax.set_title('(b) detection vs false-alarm frontier', loc='left',
                 fontsize=10.5, color=INK)

    fig.suptitle('Tuning the alert rule: an order of magnitude fewer false '
                 'alarms at the same nine-day lead', x=0.005, ha='left',
                 fontsize=12.5, fontweight='bold', y=1.07)
    fig.text(0.005, 0.995,
             'Rule parameters swept in-sample over 2010-2020 (quantile, '
             'lookback, median window, sustain); the Dec-2019 lead is '
             'robust across 134/240 settings (8.6-9.5 days).',
             fontsize=8.6, color=INK2, va='top')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, 'F10_optimal_alert_rule')


# ============================================================
# Paper composites (main figures M1-M4; M2 = F3 unchanged)
# ============================================================
def m1():
    """M1 = F1 + F2: the curation rule and its out-of-pool generalization."""
    panels = []
    for target, f in [('FWVZ', r'forecasts\phase_b\ablation_FWVZ.csv'),
                      ('KRVZ', r'forecasts\phase_b\ablation_KRVZ.csv')]:
        df = pd.read_csv(os.path.join(REPO, f))
        df['pool'] = df.variant.map(lambda v: pool_of_variant(target, v))
        panels.append((target, df))
    dfw = pd.read_csv(os.path.join(REPO, r'forecasts\phase_b_wiz\ablation_WIZ.csv'))
    dfw['pool'] = dfw.sources.map(lambda s: s.split('+'))
    panels.append(('WIZ', dfw))

    fig = plt.figure(figsize=(12.5, 9.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.35,
                          wspace=0.28)
    names = {'FWVZ': 'Ruapehu (FWVZ)', 'KRVZ': 'Tongariro (KRVZ)',
             'WIZ': 'Whakaari (WIZ)'}
    for i, (target, df) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        df = df.copy()
        df['label'] = df.pool.map(pool_label)
        df['wiz'] = df.pool.map(lambda p: 'WIZ' in p)
        df = df.sort_values('auc').reset_index(drop=True)
        colors = [C2 if w else C1 for w in df.wiz]
        ax.hlines(range(len(df)), 0.4, df.auc, color=GRID, lw=0.7, zorder=1)
        ax.scatter(df.auc, range(len(df)), c=colors, s=36, zorder=3)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df.label, fontsize=7.5)
        best = df.iloc[-1]
        ax.annotate(f'{best.auc:.3f}', (best.auc, len(df) - 1),
                    xytext=(4, 0), textcoords='offset points', va='center',
                    fontsize=8, color=INK)
        ax.set_xlim(0.4, 1.02)
        ax.set_title(f'({chr(97+i)}) target: {names[target]}', loc='left',
                     fontsize=10, color=INK)
        style_ax(ax, xgrid=True)
        ax.set_xlabel('eruption AUC (target out-of-sample)', fontsize=8.5)

    # (d) external validation
    ax = fig.add_subplot(gs[1, :])
    df = pd.read_csv(os.path.join(
        REPO, r'forecasts\phase_b_external\external_validation.csv'))
    piv = df.pivot(index='target', columns='ensemble', values='auc')
    nev = df.groupby('target').n_events.first()
    piv = piv.sort_values('KOS')
    names2 = {'PN7A': 'Pavlof (PN7A)', 'PVV': 'Pavlof (PVV)',
              'VNSS': 'Veniaminof', 'BELO': 'Bezymianny', 'COP': 'Copahue',
              'MBGH': 'Montserrat', 'VRLE': 'VRLE', 'VTUN': 'VTUN'}
    y = np.arange(len(piv))
    for i, (t, r) in enumerate(piv.iterrows()):
        ax.plot([r['all5'], r['KOS']], [i, i], color=GRID, lw=2, zorder=1)
    ax.scatter(piv['all5'], y, color=C2, s=48, zorder=3,
               label='all 5 sources (generalized model)')
    ax.scatter(piv['KOS'], y, color=C1, s=48, zorder=3,
               label='curated pool K+O+S (no Whakaari)')
    ax.set_yticks(y)
    ax.set_yticklabels([f'{names2.get(t, t)}  (n={nev[t]})' for t in piv.index],
                       fontsize=8.5)
    style_ax(ax, xgrid=True)
    ax.set_xlabel('eruption AUC — 31 eruptions never seen by any model',
                  fontsize=9)
    ax.set_xlim(0.1, 1.0)
    ax.legend(loc='upper left', frameon=False, fontsize=8.5)
    ax.set_title('(d) external validation: the curated pool wins on 8 of 8 '
                 'out-of-pool volcanoes', loc='left', fontsize=10, color=INK)

    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([], [], marker='o', ls='', color=C1, label='pool without Whakaari'),
        Line2D([], [], marker='o', ls='', color=C2, label='pool contains Whakaari')],
        loc='upper right', bbox_to_anchor=(0.995, 1.045), frameon=False,
        fontsize=9, ncol=2)
    fig.suptitle('Source-pool curation: every Whakaari-free pool beats every '
                 'Whakaari-containing pool', x=0.005, ha='left', fontsize=13,
                 fontweight='bold', y=1.045)
    fig.text(0.005, 1.0,
             '(a–c) all 15 source-pool subsets per target, leave-target-'
             'volcano-out (W=Whakaari, F=Ruapehu, K=Tongariro, O=Ontake, '
             'S=St Helens); (d) trained ensembles applied unchanged to '
             'volcanoes outside the pool.', fontsize=9, color=INK2)
    fig.tight_layout()
    save(fig, 'M1_curation_and_generalization')


def m3():
    """M3 = F5 + F4: when does adaptation to the target pay?"""
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8),
                             gridspec_kw={'width_ratios': [1, 1.15]})

    # (a) ladder (F5)
    ax = axes[0]
    r1 = pd.read_csv(os.path.join(REPO, r'forecasts\phase_c\phase_c_results.csv'))
    r2 = pd.read_csv(os.path.join(
        REPO, r'forecasts\phase_c\phase_c_results_variants.csv'))
    rw = pd.read_csv(os.path.join(
        REPO, r'forecasts\phase_c\phase_c_results_WIZ.csv'))
    allr = pd.concat([r1, r2, rw])
    allr = allr[allr.base == 'full']
    base = {t: allr[(allr.target == t) &
                    (allr.method == 'base (no refinement)')].auc.iloc[0]
            for t in ['FWVZ', 'KRVZ', 'WIZ']}
    nerup = {'KRVZ': 2, 'FWVZ': 3, 'WIZ': 5}
    methods = ['strut', 'ser', 'strut_us', 'ser_us']
    mlabels = ['STRUT', 'SER', 'STRUT (per-tree undersample)',
               'SER (per-tree undersample)']
    mcolors = [C1, C2, C3, C4]
    ax.axhline(0, color=BASE, lw=1)
    ax.text(5.55, 0.006, 'helps', fontsize=8, color=INK2, va='bottom')
    ax.text(5.55, -0.012, 'harms', fontsize=8, color=INK2, va='top')
    for m, ml, c in zip(methods, mlabels, mcolors):
        xs, ys = [], []
        for t in ['KRVZ', 'FWVZ', 'WIZ']:
            row = allr[(allr.target == t) & (allr.method == m)]
            if len(row):
                xs.append(nerup[t])
                ys.append(row.auc.iloc[0] - base[t])
        ax.plot(xs, ys, color=c, lw=2, marker='o', ms=6.5, label=ml,
                markeredgecolor=SURF, markeredgewidth=1.2)
    ax.set_xticks([2, 3, 5])
    ax.set_xticklabels(['2\n(Tongariro)', '3\n(Ruapehu)', '5\n(Whakaari)'],
                       fontsize=8.5)
    ax.set_xlim(1.6, 5.95)
    ax.set_xlabel('recorded eruptions at the target volcano', fontsize=9)
    ax.set_ylabel('AUC change from refining on target data', fontsize=9)
    style_ax(ax, ygrid=True)
    ax.legend(loc='lower right', bbox_to_anchor=(0.99, 0.30), frameon=False,
              fontsize=7.8)
    ax.set_title('(a) fine-tuning is unreliable below five eruptions',
                 loc='left', fontsize=10, color=INK)

    # (b) prospective scoreboard (F4)
    ax = axes[1]
    df = pd.read_csv(os.path.join(REPO, r'forecasts\phase_g\phase_g_summary.csv'))
    pmethods = ['full', 'curated', 'ser_us', 'tboost']
    pmlabels = ['full pool', 'curated pool', 'refined trees (ser_us)',
                'target-boosted (tboost)']
    targets = ['FWVZ', 'KRVZ', 'WIZ']
    tlabels = ['Ruapehu\n3 eruptions', 'Tongariro\n2 eruptions',
               'Whakaari\n5 eruptions']
    x = np.arange(len(targets))
    w = 0.19
    for j, (m, ml, c) in enumerate(zip(pmethods, pmlabels, mcolors)):
        vals = [df[(df.target == t) & (df.method == m)].auc.iloc[0]
                for t in targets]
        pos = x + (j - 1.5) * w
        ax.bar(pos, vals, width=w - 0.025, color=c, edgecolor=SURF,
               linewidth=1.5, label=ml)
        for xi, v in zip(pos, vals):
            ax.text(xi, v + 0.012, f'{v:.2f}', ha='center', fontsize=7.3,
                    color=INK2)
    for t, xi in [('FWVZ', 0), ('KRVZ', 1)]:
        v = df[(df.target == t) & (df.method == 'tboost')].auc.iloc[0]
        ax.text(xi + 1.5 * w, v - 0.05, '*', ha='center', fontsize=12,
                color=SURF, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tlabels, fontsize=8.5)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel('pseudo-prospective eruption AUC', fontsize=9)
    style_ax(ax, ygrid=True)
    ax.legend(ncol=2, loc='upper left', frameon=False, fontsize=7.8)
    ax.set_title('(b) prospective test: curation always helps; boosting '
                 'needs history', loc='left', fontsize=10, color=INK)

    fig.suptitle('When does adapting to the target pay?', x=0.005, ha='left',
                 fontsize=13, fontweight='bold', y=1.07)
    fig.text(0.005, 1.0,
             '(a) leave-one-eruption-out refinement gain vs eruption count; '
             '(b) all target information restricted to before each eruption, '
             'scored on unrest-month windows (* = score carried by the '
             'no-prior-eruption fallback).', fontsize=9, color=INK2)
    fig.tight_layout()
    save(fig, 'M3_adaptation_when_it_pays')


def m4():
    """M4 = F10a + F9b + F10b: the operational alert system."""
    import matplotlib.dates as mdates
    from datetime import datetime, timedelta
    te = datetime(2019, 12, 9, 1, 11)
    YTOP = 0.8
    Q, LB_D, MW_H, SU_H = 0.995, 180, 24, 24

    files = sorted(glob(os.path.join(REPO, r'forecasts\phase_b_wiz', 'WIZ_*.pkl')))
    df = pd.concat([pd.read_pickle(f) for f in files])
    df = df[~df.index.duplicated()].sort_index()
    con = df['O'].resample('1h').median().dropna()
    med = con.rolling(MW_H, min_periods=MW_H // 2).median()
    thr = con.rolling(LB_D * 24, min_periods=240).quantile(Q).shift(1)
    active = (med > thr) & thr.notna()
    episodes, onset, run = [], None, 0
    for t, a in active.items():
        if a:
            run += 1
            if run == SU_H:
                onset = t - timedelta(hours=SU_H - 1)
        else:
            if onset is not None:
                episodes.append((onset, t))
                onset = None
            run = 0
    if onset is not None:
        episodes.append((onset, active.index[-1]))

    fig = plt.figure(figsize=(12.8, 8.6))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.3)

    # (a) two-year optimal-rule view
    ax = fig.add_subplot(gs[0, :])
    ctx0 = te - timedelta(days=2 * 365.25)
    eps2 = [(a, b) for a, b in episodes if b >= ctx0]
    ov = df['O'][(df.index >= ctx0) & (df.index <= te + timedelta(days=4))]
    ax.plot(ov.index, ov.values, lw=0.3, color=C1, alpha=0.30)
    m5 = ov.rolling('5D').median()
    ax.plot(m5.index, m5.values, lw=1.5, color='#104281',
            label='Ontake-only pool, 5-day median')
    tt = thr[(thr.index >= ctx0) & (thr.index <= te + timedelta(days=4))]
    ax.plot(tt.index, tt.values, lw=1.0, ls=':', color=INK2,
            label=f'optimal threshold (trailing {LB_D}-day q{Q})')
    for a, b in eps2:
        ax.axvspan(a, a + timedelta(days=10), color='#eda100', alpha=0.15, lw=0)
    for a, b in eps2:
        ax.axvspan(a, b, color='#eda100', alpha=0.45, lw=0)
    ax.axvline(te, color='#e34948', ls='--', lw=1.5)
    ax.text(te - timedelta(days=18), YTOP * 0.97, 'eruption',
            color='#e34948', fontsize=9, va='top', ha='right')
    pre = [o for o, b in eps2 if o < te and b > te - timedelta(days=10)]
    if pre:
        lead = (te - min(pre)).total_seconds() / 86400
        ax.annotate(f'alert {lead:.1f} days\nbefore eruption',
                    xy=(min(pre), 0.62), xytext=(te - timedelta(days=300), 0.68),
                    fontsize=9, color=INK,
                    arrowprops=dict(arrowstyle='->', color=INK2, lw=1))
    nfa = len(eps2) - len(pre)
    from matplotlib.patches import Patch
    h, _l = ax.get_legend_handles_labels()
    h += [Patch(facecolor='#eda100', alpha=0.45, label='alert episode'),
          Patch(facecolor='#eda100', alpha=0.15,
                label='10-day warning period after trigger')]
    ax.legend(handles=h, loc='upper left', frameon=False, fontsize=8)
    ax.set_xlim(ctx0, te + timedelta(days=4))
    ax.set_ylim(0, YTOP)
    ax.set_ylabel('consensus')
    style_ax(ax, ygrid=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.set_title(f'(a) optimised alert rule, 2018–2019: {nfa} false '
                 'alarms and a nine-day pre-eruption alert', loc='left',
                 fontsize=10, color=INK)

    # (b) final month
    ax = fig.add_subplot(gs[1, :2])
    w0 = te - timedelta(days=30.4)
    o = df['O'][(df.index >= w0) & (df.index <= te + timedelta(days=4))]
    res = pd.read_pickle(os.path.join(REPO, r'forecasts\phase_g\WIZ__e4.pkl'))
    tb = res['tboost']
    bg = o[o.index < te - timedelta(days=10)]
    thrb = bg.quantile(0.99)
    medb = o.rolling('12h').median()
    above = (medb[medb.index >= te - timedelta(days=10)] > thrb)
    run, onset = 0, None
    for t, a in above.items():
        run = run + 1 if a else 0
        if run >= 36:
            onset = t - timedelta(hours=6)
            break
    ax.plot(o.index, o.values, lw=0.4, color=C1, alpha=0.35)
    ax.plot(medb.index, medb.values, lw=1.8, color='#104281',
            label='Ontake-only pool, 12-h median')
    ax.plot(tb.index, tb.values, lw=1.0, color='#c98500',
            label='target-boosted (raw, prospective)')
    ax.axhline(thrb, color=INK2, lw=1, ls=':')
    ax.axvline(te, color='#e34948', ls='--', lw=1.5)
    ax.axvspan(onset, te, color='#eda100', alpha=0.15)
    ax.annotate(f'sustained alert begins\n'
                f'{(te-onset).total_seconds()/86400:.1f} days before eruption',
                xy=(onset, thrb + 0.02),
                xytext=(onset - timedelta(days=9), 0.68), fontsize=9,
                color=INK, arrowprops=dict(arrowstyle='->', color=INK2, lw=1))
    ax.text(te + timedelta(hours=8), YTOP * 0.97, 'eruption\nDec 9, 2019',
            color='#e34948', fontsize=8.5, va='top')
    ax.set_ylim(0, YTOP)
    ax.set_ylabel('consensus')
    style_ax(ax, ygrid=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.legend(loc='upper left', frameon=False, fontsize=8)
    ax.set_title('(b) the final month before the December 2019 eruption',
                 loc='left', fontsize=10, color=INK)

    # (c) frontier
    ax = fig.add_subplot(gs[1, 2])
    sw = pd.read_csv(os.path.join(REPO, r'forecasts\phase_b',
                                  'alert_rule_sweep.csv'))
    frac = sw.detected / 5.
    ax.scatter(sw.false_alarms_per_year, frac, s=16, color=MUT, alpha=0.45,
               edgecolor='none', label='all 240 rule settings', zorder=2)
    front = sw.groupby('detected').false_alarms_per_year.min().reset_index()
    front = front[front.detected > 0].sort_values('detected')
    fx = list(front.false_alarms_per_year) + [13]
    fy = list(front.detected / 5.)
    ax.step(fx, fy + [fy[-1]], where='post', color=C2, lw=2,
            label='best achievable', zorder=3)
    ax.scatter(front.false_alarms_per_year, front.detected / 5., color=C2,
               s=44, zorder=4, edgecolor=SURF, linewidth=0.8)
    ax.scatter([10.8], [0.4], color=C1, s=52, zorder=5, edgecolor=SURF,
               linewidth=0.8, label='original heuristic')
    ax.set_xlim(0, 13)
    ax.set_ylim(-0.04, 0.72)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_yticklabels(['0/5', '1/5', '2/5', '3/5'])
    ax.set_xlabel('false alarms per year', fontsize=9)
    ax.set_ylabel('eruptions detected', fontsize=9)
    style_ax(ax, ygrid=True)
    ax.legend(loc='lower right', frameon=False, fontsize=7.2)
    ax.set_title('(c) detection vs false-alarm\nfrontier', loc='left',
                 fontsize=10, color=INK)

    fig.suptitle('An operational alert system for Whakaari, built from '
                 'foreign data', x=0.005, ha='left', fontsize=13,
                 fontweight='bold', y=1.03)
    fig.text(0.005, 0.99,
             'Rule: alert when the 12/24-h consensus median exceeds a '
             'trailing-background quantile for a sustained period (strictly '
             'causal). Rule parameters tuned in-sample (see text); the '
             'Dec-2019 lead is robust across 134/240 settings.',
             fontsize=8.6, color=INK2, va='top')
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    save(fig, 'M4_operational_alerts')


if __name__ == '__main__':
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    fig8()
    fig9()
    fig10()
    m1()
    m3()
    m4()
    print('figure pack complete ->', FIG)
