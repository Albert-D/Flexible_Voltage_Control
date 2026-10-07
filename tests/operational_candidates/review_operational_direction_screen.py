"""Cache-only figures and review notebook; no simulation or policy changes."""
from pathlib import Path
import gzip
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from config import Config

ROOT = 'operational_direction_screen_2026-09-15'
DATA = Path(Config.data_path)
CACHE = DATA / 'cache' / ROOT
IMAGES = DATA / 'images' / ROOT
METHODS = ('Linear', 'Safe-DDPG', 'RLC-FT')
COLORS = ('#B76F67', '#6F8FAC', '#6D8F5E')
FILLS = ('#D2877E', '#9FB4CD', '#98B486')
STYLES = ('--', '-.', '-')
CONTROL = [17, 20, 29, 44, 52]
CASES = ('corridor_to32_z2.5_midday', 'corridor_to53_z2.5_evening')


def load(path):
    with gzip.open(path, 'rb') as stream:
        return pickle.load(stream)


def longest(mask):
    changes = np.diff(np.r_[False, mask, False].astype(int))
    lengths = np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)
    return int(lengths.max()) if lengths.size else 0


def metrics(result, start=0, stop=None):
    v = result['all_states'][start:stop]
    bad = (v < .95) | (v > 1.05)
    fraction = bad.sum(axis=1) / np.isfinite(v).sum(axis=1)
    duration = .1 * longest(fraction > .1)
    return dict(peak=float(np.nanmax(v)), minimum=float(np.nanmin(v)),
                duration=duration, exposure=.1*float((fraction > .1).sum()),
                controlled_duration=.1*longest(bad[:, CONTROL].any(axis=1)),
                cost=float(-result['rewards'][start:stop].sum()),
                passes=bool(np.nanmax(v) <= 1.10 and duration < 5),
                q_peak=float(np.abs(result['actions'][start:stop]).max()))


def style():
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 9,
                         'axes.titlesize': 10, 'axes.labelsize': 9,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})


def save(fig, name):
    IMAGES.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf'):
        fig.savefig(IMAGES / f'{name}.{extension}', dpi=210, facecolor='white')
    plt.close(fig)


def tables():
    rows = []
    for folder in ('screen_initial', 'screen_pv18_neighbors', 'screen_corridors', 'screen_candidate_neighbors', 'screen_candidate_local_refinement'):
        path = CACHE / folder / 'summaries.json'
        if path.exists():
            for row in json.loads(path.read_text()):
                rows.append(dict(batch=folder, **row))
    frame = pd.json_normalize(rows)
    frame.to_csv(CACHE / 'all_screening_results.csv', index=False)
    full = []
    for case in CASES:
        for method in ('No control', *METHODS):
            for variant in ('original', 'matched_margin'):
                path = CACHE / 'full' / f'{case}__{method}__{variant}.pkl.gz'
                if not path.exists():
                    continue
                payload = load(path)
                if 'result' not in payload:
                    full.append(dict(case_id=case, method=method, variant=variant, window='failed', **{'error': payload['summary']['error']}))
                    continue
                r = payload['result']; k = r['case']['event']
                for window, a, b in (('full_day', 0, None), ('event_60min', k, k+600), ('remaining_day', k, None)):
                    full.append(dict(case_id=case, method=method, variant=variant, window=window, **metrics(r, a, b)))
    pd.DataFrame(full).to_csv(CACHE / 'full_day_metrics.csv', index=False)
    return frame, pd.DataFrame(full)


def candidate(case):
    records = {}
    for method in ('No control', *METHODS):
        path = CACHE / 'full' / f'{case}__{method}__original.pkl.gz'
        if not path.exists():
            return
        records[method] = load(path)
    if any('result' not in p for p in records.values()):
        return
    rr = {m: p['result'] for m, p in records.items()}
    profiles = load(DATA / 'cache/load_branch_restoration_2026-09-15/combined.pkl.gz')['profiles']
    k = rr['RLC-FT']['case']['event']; h = np.arange(14400) / 600
    for name, start, stop in (('event_60min', k, k+600), ('full_day', 0, None)):
        fig, axes = plt.subplots(2, 3, figsize=(10, 6.3))
        fig.subplots_adjust(left=.085, right=.985, top=.83, bottom=.16, wspace=.50, hspace=.72)
        a, b, c, d, e, f = axes.flat
        for values, label, color in ((.88*profiles['p'], 'Load P', '#6F8FAC'),
                                    (.88*profiles['q'], 'Load Q', '#C78F70'),
                                    (1.75*profiles['pv_p'], 'PV P', '#6D8F5E')):
            a.plot(h, values, label=label, color=color, lw=1.05)
        a.legend(ncol=3, frameon=False, fontsize=8, loc='lower left', bbox_to_anchor=(-.05, 1.23), handlelength=1.2, columnspacing=.9)
        a.set(title='Operating profiles', ylabel='Power (MW / Mvar)')
        extrema = []
        for ax, method, color in ((b, 'No control', '#777777'), (c, 'RLC-FT', COLORS[2])):
            v = rr[method]['all_states']
            low, high = np.nanmin(v, axis=1), np.nanmax(v, axis=1)
            ax.fill_between(h, low, high, color=color, alpha=.15, lw=0)
            ax.plot(h, low, color=color, lw=.9)
            ax.plot(h, high, color=color, lw=.9)
            for limit in (.95, 1.05):
                ax.axhline(limit, color='#777777', lw=.7, ls=':')
            ax.set_title(method)
            extrema.extend([low.min(), high.max()])
        b.set_ylabel('Voltage envelope (p.u.)')
        for ax in (b, c):
            ax.set_ylim(min(.94, min(extrema)-.01), max(1.06, max(extrema)+.01))
        fig.text(.665, .946, 'Envelope: all 56 modeled buses', ha='center', fontsize=9)
        for ax in (a, b, c):
            ax.axvline(k/600, color='#9B4E4A', ls=':', lw=.85)
            ax.set(xlim=(0, 24), xticks=[0, 6, 12, 18, 24], xlabel='Time (h)')
        mm = {m: metrics(rr[m], start, stop) for m in METHODS}
        for ax, key, title, ylabel, limit in ((d, 'peak', 'Peak voltage', 'Voltage (p.u.)', 1.10),
                (e, 'duration', 'Longest voltage violation', 'Duration (min)', 5),
                (f, 'cost', 'Total objective cost', 'Normalized cost', None)):
            vals = np.array([mm[m][key] for m in METHODS])
            if key == 'cost': vals = vals/vals[0]
            floor = .95 if key == 'peak' else min(.9, vals.min()-.05) if key == 'cost' else 0
            ceiling = max(vals.max(), limit or vals.max())
            span = max(ceiling-floor, .04)
            ax.set_ylim(floor-.035*span if key == 'duration' else floor, ceiling+span*.27)
            for j, value in enumerate(vals):
                ax.vlines(j, floor, value, color=COLORS[j], lw=1.2)
                ax.scatter(j, value, s=49, facecolor=FILLS[j], edgecolor=COLORS[j], zorder=4)
                ax.annotate(f'{value:.1f}' if key == 'duration' else f'{value:.3f}', (j, value), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=9,
                            bbox=dict(facecolor='white', edgecolor='none', pad=.7, alpha=.95))
            if limit is not None:
                ax.axhline(limit, color='#A64F4A', ls='--', lw=.85)
            else:
                ax.axhline(1, color='#999999', ls=':', lw=.75)
            ax.set(xlim=(-.45, 2.45), title=title, ylabel=ylabel)
            ax.set_xticks(range(3), METHODS, rotation=15, ha='right')
        for letter, ax in zip('abcdef', axes.flat):
            ax.text(-.23, 1.08, letter, transform=ax.transAxes, fontsize=12, weight='bold')
            ax.set_axisbelow(True); ax.grid(axis='y', color='#e5e5e5', lw=.5)
        window = 'first hour after change' if start else 'full 24 hours'
        fig.text(.5, .065, f'Hypothetical supply-path impedance change | Metrics: {window}', ha='center', fontsize=9)
        fig.text(.5, .031, 'Duration: >10% of modeled buses outside 0.95-1.05 p.u.; dashed references: 1.10 p.u. and 5 min.', ha='center', fontsize=8)
        save(fig, f'{case}_{name}_2x3')
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    fig.subplots_adjust(left=.075, right=.985, top=.81, bottom=.20, wspace=.44)
    sl = slice(k-10, k+50); t = (np.arange(14400)[sl]-k)*.1
    for method, color, ls in zip(METHODS, COLORS, STYLES):
        v = rr[method]['all_states'][sl]
        series = (v.min(axis=1), v.max(axis=1), 100*((v<.95)|(v>1.05)).mean(axis=1))
        for ax, values in zip(axes, series):
            ax.plot(t, values, color=color, ls=ls, lw=1.35, label=method)
    for ax, title, ylabel, limit in zip(axes, ('Minimum bus voltage', 'Maximum bus voltage', 'Buses outside safe range'),
                                      ('Voltage (p.u.)', 'Voltage (p.u.)', 'Fraction (%)'), (.95, 1.10, 10)):
        ax.set(title=title, ylabel=ylabel, xlabel='Time from change (min)')
        ax.axhline(limit, color='#A64F4A', lw=.8, ls=':')
        ax.axvline(0, color='#999999', lw=.7, ls=':')
    fig.legend(handles=axes[0].get_legend_handles_labels()[0], labels=METHODS, ncol=3, frameon=False, loc='upper center')
    save(fig, f'{case}_event_detail')


def screen_maps(frame):
    for batch, group in frame.groupby('batch', sort=False):
        ids = list(dict.fromkeys(group.case_id))
        values = np.full((len(ids), 3), 2.)
        labels = {}
        for i, case in enumerate(ids):
            for j, method in enumerate(METHODS):
                row = group[(group.case_id == case) & (group.method == method)]
                if row.empty: continue
                r = row.iloc[0]
                if r.status != 'ok': labels[i,j] = 'PF failed'; continue
                values[i,j] = 0 if r.passes else 1
                labels[i,j] = f'{r.peak:.3f} | {r.fraction_duration_min:.1f}'
        height = max(3.5, .30*len(ids)+1.9)
        fig, ax = plt.subplots(figsize=(9.5, height))
        fig.subplots_adjust(left=.47, right=.98, top=1-1.05/height, bottom=.06)
        ax.imshow(values, cmap=ListedColormap(['#e3eddd', '#f0dcd8', '#dddddd']), vmin=0, vmax=2, aspect='auto')
        ax.set_xticks(range(3), METHODS); ax.xaxis.tick_top()
        ax.set_yticks(range(len(ids)), [x.replace('corridor_', '').replace('path_', '') for x in ids])
        ax.tick_params(length=0, pad=6)
        for (i,j), label in labels.items(): ax.text(j, i, label, ha='center', va='center', fontsize=8)
        fig.suptitle('Peak (p.u.) | longest duration (min)\nGreen: passes both screening references; red: does not; gray: solver failure', fontsize=10)
        save(fig, batch+'_all_cases')


def notebook():
    import nbformat as nbf
    path = Path(__file__).resolve().with_name('test_operational_direction_screen.ipynb')
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell('# Operational direction screening\n\nCache-only review. Original data and policies; RLC-FT scale = 1.0. All screened cases, including failures, are retained. Corridor changes are hypothetical equivalent electrical changes, not verified SCE switching assets. References apply to modeled buses, not measured customer meters. Full-day and first-hour metrics are kept separate.'),
                nbf.v4.new_code_cell('import review_operational_direction_screen as review\nfrom IPython.display import display, Image\nreview.style()\nscreen, full = review.tables()\ndisplay(screen.groupby(["batch", "status"]).size().rename("runs").to_frame())'),
                nbf.v4.new_markdown_cell('## Full-day confirmation\n\nThe no-control comparison and voltage minima prevent a favorable pass/fail summary from hiding transient undervoltage or an unnecessarily difficult controller state.'),
                nbf.v4.new_code_cell('display(full.round(4))')]
    names = [f'{case}_{window}_2x3' for case in CASES for window in ('event_60min', 'full_day')]
    names += [f'{case}_event_detail' for case in CASES]
    names += [f'{batch}_all_cases' for batch in ('screen_initial', 'screen_pv18_neighbors', 'screen_corridors', 'screen_candidate_neighbors', 'screen_candidate_local_refinement')]
    for name in names:
        nb.cells.append(nbf.v4.new_code_cell(f'display(Image(filename=str(review.IMAGES / "{name}.png")))'))
    nb.cells.append(nbf.v4.new_markdown_cell('## Interpretation boundaries\n\n- A finite-grid feasibility witness only establishes that one post-event operating point has an admissible reactive-power allocation.\n- All-node violation duration is not identical to the five-bus stability objective.\n- Matched-margin baselines use complete new daily histories, not an unmatched cached initial state.\n- The best visible example was found by screening; neighboring settings and negative examples are shown, not held-out confirmation of broad superiority.'))
    nb.cells.append(nbf.v4.new_code_cell('import json\nimport pandas as pd\nfrozen = json.loads((review.CACHE / "frozen_input/summaries.json").read_text())\ndisplay(pd.json_normalize(frozen)[["case_id", "peak", "minimum", "fraction_duration_min", "cost", "normalized_input_change"]])'))
    nb.metadata.kernelspec = dict(display_name='Python 3', language='python', name='python3')
    nbf.write(nb, path)
    return path


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--notebook', action='store_true'); args = parser.parse_args()
    style(); frame, _ = tables(); screen_maps(frame)
    for case in CASES: candidate(case)
    if args.notebook:
        from nbclient import NotebookClient
        import nbformat
        path = notebook(); nb = nbformat.read(path, as_version=4)
        NotebookClient(nb, timeout=180, kernel_name='python3', resources={'metadata': {'path': str(path.parent)}}).execute()
        nbformat.write(nb, path)
    print('COMPLETE ' + str(IMAGES))


if __name__ == '__main__': main()
