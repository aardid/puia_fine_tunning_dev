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


if __name__ == '__main__':
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    print('figure pack complete ->', FIG)
