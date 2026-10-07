"""Cache-only summaries and figures for the operational path-shift study."""
from pathlib import Path
import sys
import json
import argparse
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as experiment
import review_operational_direction_screen as legacy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = experiment.OUT
CACHE = OUT / 'cache'
FIGURES = OUT / 'figures'
METHODS = legacy.METHODS


def tables():
    all_rows = []
    for folder in sorted(OUT.glob('screen_*')):
        path = folder / 'all_summaries.json'
        if path.exists():
            all_rows.extend(dict(batch=folder.name, **row) for row in json.loads(path.read_text()))
    frame = pd.json_normalize(all_rows)
    frame.to_csv(OUT / 'screening_results.csv', index=False)
    selected = []
    path = OUT / 'selected_cases.json'
    if path.exists():
        selected = json.loads(path.read_text())
    full_rows = []
    for case in selected:
        for method in ('No control', *METHODS):
            path = CACHE / 'full' / f'{case["id"]}__{method}__original.pkl.gz'
            if not path.exists():
                continue
            p = legacy.load(path)
            if 'result' not in p:
                full_rows.append(dict(p['summary'], window='failed'))
                continue
            for window, a, b in (('full_day', 0, None), ('event_60min', case['event'], case['event']+600)):
                full_rows.append(dict(case_id=case['id'], method=method, window=window,
                                      **legacy.metrics(p['result'], a, b)))
    full = pd.DataFrame(full_rows)
    full.to_csv(OUT / 'full_metrics.csv', index=False)
    return frame, full


def screening_figure(frame):
    for batch, group in frame.groupby('batch', sort=False):
        manifest = json.loads((OUT / batch / 'manifest.json').read_text())
        ids = [c['id'] for c in manifest['cases'] if c['id'] in set(group.case_id)]
        fig, axes = plt.subplots(1, 3, figsize=(12, max(4.2, .31*len(ids)+1.6)), sharey=True)
        fig.subplots_adjust(left=.28, right=.98, top=.88, bottom=.1, wspace=.28)
        for ax, metric, title, limit in zip(axes,
                ('peak', 'fraction_duration_min', 'cost'),
                ('Peak voltage (p.u.)', 'Longest violation (min)', 'Event cost / Linear'), (1.1, 5., 1.)):
            for j, (method, color, marker) in enumerate(zip(METHODS, legacy.COLORS, ('s', '^', 'o'))):
                xx, yy = [], []
                for i, case in enumerate(ids):
                    rows = group[(group.case_id == case) & (group.method == method)]
                    if rows.empty or rows.iloc[0].status != 'ok':
                        ax.text(.02, i+.2*(j-1), 'PF failed', transform=ax.get_yaxis_transform(),
                                color=color, fontsize=6, va='center')
                        continue
                    value = float(rows.iloc[0][metric])
                    if metric == 'cost':
                        lin = group[(group.case_id == case) & (group.method == 'Linear')]
                        if lin.empty or lin.iloc[0].status != 'ok':
                            continue
                        value /= float(lin.iloc[0].cost)
                    xx.append(value); yy.append(i+.2*(j-1))
                ax.scatter(xx, yy, color=color, marker=marker, s=23, label=method, zorder=3)
            ax.axvline(limit, color='#9B4E4A', ls='--', lw=.8)
            ax.set_title(title)
            ax.set_axisbelow(True); ax.grid(axis='y', color='#eeeeee')
        axes[0].set_yticks(range(len(ids)), ids, fontsize=8)
        axes[0].invert_yaxis()
        fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=3, frameon=False)
        fig.text(.5, .03, 'All screened cases; event window only. Failed power flows are not treated as physical collapse.', ha='center', fontsize=8)
        save(fig, batch + '_comparison')


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    for suffix in ('pdf', 'png'):
        fig.savefig(FIGURES / f'{name}.{suffix}', dpi=200, facecolor='white')
    plt.close(fig)


def events(selected):
    for case in selected:
        records = {}
        for method in METHODS:
            path = CACHE / 'screen' / f'{case["id"]}__{method}__original.pkl.gz'
            if not path.exists():
                continue
            p = legacy.load(path)
            if 'result' in p:
                records[method] = p['result']
        if len(records) != 3:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))
        fig.subplots_adjust(left=.09, right=.98, top=.86, bottom=.12, hspace=.65, wspace=.48)
        for method, color, ls in zip(METHODS, legacy.COLORS, legacy.STYLES):
            r = records[method]; v = r['all_states']; t = np.arange(len(v))*.1
            active = np.isfinite(v).sum(axis=1)
            fraction = 100*((v<.95)|(v>1.05)).sum(axis=1)/active
            ys = (np.nanmin(v, axis=1), np.nanmax(v, axis=1), fraction,
                  np.max(np.abs(r['actions']), axis=1), np.cumsum(-r['rewards']),
                  100*((v[:, legacy.CONTROL]<.95)|(v[:, legacy.CONTROL]>1.05)).mean(axis=1))
            for ax, y in zip(axes.flat, ys):
                ax.plot(t, y, color=color, ls=ls, lw=1.3, label=method)
        titles = ('Minimum bus voltage', 'Maximum bus voltage', 'Network buses outside safe range',
                  'Maximum reactive action', 'Cumulative event cost', 'Control buses outside safe range')
        ylabels = ('Voltage (p.u.)', 'Voltage (p.u.)', 'Fraction (%)', 'Reactive power (Mvar)', 'Objective cost', 'Fraction (%)')
        for i, (ax, title, ylabel) in enumerate(zip(axes.flat, titles, ylabels)):
            ax.set(title=title, ylabel=ylabel, xlabel='Time after change (min)')
            ax.set_axisbelow(True); ax.grid(color='#e8e8e8', lw=.5)
            ax.text(-.24, 1.12, 'abcdef'[i], transform=ax.transAxes, weight='bold', fontsize=12)
        for ax, value in zip(axes.flat, (.95, 1.1, 10.)):
            ax.axhline(value, color='#9B4E4A', ls=':', lw=.8)
        fig.legend(*axes.flat[0].get_legend_handles_labels(), ncol=3, loc='upper center', frameon=False)
        save(fig, case['id'] + '_event_diagnostics')


def full_figures(selected):
    legacy.CACHE = CACHE
    legacy.IMAGES = FIGURES
    for case in selected:
        legacy.candidate(case['id'])


def notebook():
    import nbformat as nbf
    from nbclient import NotebookClient
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell('# Operational path shifts\n\nCache-only review of all screened scenarios and selected full-day cases. Fixed controllers; RLC-FT scale=1. Synthetic tie connections and equivalent impedance changes are hypotheses, not verified SCE asset events. The thresholds apply to modeled-node proxies.'),
                nbf.v4.new_code_cell(f'from pathlib import Path\nimport sys\nfolder = Path({str(HERE)!r})\nsys.path.insert(0, str(folder))\nimport review\nfrom IPython.display import display, Image\nscreen, full = review.tables()\ndisplay(screen[["case_id", "method", "status", "peak", "minimum", "fraction_duration_min", "cost"]])\ndisplay(full)')]
    for path in sorted(FIGURES.glob('*.png')):
        nb.cells.append(nbf.v4.new_markdown_cell('## ' + path.stem.replace('_', ' ')))
        nb.cells.append(nbf.v4.new_code_cell(f'display(Image(filename={str(path)!r}))'))
    nb.metadata.kernelspec = dict(display_name='Python 3', language='python', name='python3')
    NotebookClient(nb, timeout=180, kernel_name='python3', resources={'metadata': {'path': str(HERE)}}).execute()
    nbf.write(nb, HERE / 'review.ipynb')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--notebook', action='store_true'); args = parser.parse_args()
    legacy.style()
    frame, full = tables()
    screening_figure(frame)
    selected_file = OUT / 'selected_cases.json'
    if selected_file.exists():
        selected = json.loads(selected_file.read_text())
        extra_file = OUT / 'review_cases.json'
        event_cases = json.loads(extra_file.read_text()) if extra_file.exists() else selected
        events(event_cases); full_figures(selected)
    if args.notebook:
        notebook()
    print(str(FIGURES), flush=True)


if __name__ == '__main__':
    main()
