"""Cache-only presentation; does not change the frozen simulation protocol."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.ticker import FuncFormatter
import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
ROOT = Path('D:/Code/Python/Flexible_Voltage_Control/experiments/operational_path_shift/20260916_robustness_validation')
METHODS = ['Linear', 'Safe-DDPG', 'RLC-FT']
COLORS = ['#B76F67', '#6F8FAC', '#6D8F5E']
INDEPENDENT_N = 60
NOTEBOOK_NAME = 'robustness_validation_review.ipynb'


def main():
    data = pd.read_csv(ROOT/'all_metrics.csv')
    inv = json.loads((ROOT/'topology_inventory.json').read_text())
    labels = {r['line']: '-'.join(map(str, r['buses'])) for r in inv['all_switches']}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 11, 'axes.labelsize': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    figures = []
    for phase, n in [('local', 21), ('independent', INDEPENDENT_N)]:
        frame = data[data.phase.eq(phase)]
        if len(frame) != 3*n:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.65), gridspec_kw={'width_ratios': [1, 1, 1.3]})
        fig.subplots_adjust(left=.065, right=.985, bottom=.23, top=.82, wspace=.43)
        rng = np.random.default_rng(19)
        for ax, key, limit, title, ylabel in [
            (axes[0], 'peak', 1.1, 'Peak voltage', 'Voltage (p.u.)'),
            (axes[1], 'duration', 5., 'Longest voltage violation', 'Duration (min)')]:
            for i, method in enumerate(METHODS):
                vals = frame.loc[frame.method.eq(method) & frame.status.eq('ok'), key].to_numpy()
                ax.scatter(i+rng.uniform(-.16, .16, len(vals)), vals, s=22,
                           color=COLORS[i], alpha=.55, edgecolors='none', zorder=3)
                if len(vals):
                    lo, med, hi = np.quantile(vals, [.25, .5, .75])
                    ax.vlines(i, lo, hi, lw=2, color='#333333', zorder=4)
                    ax.plot([i-.11, i+.11], [med, med], color='#222222', lw=2, zorder=5)
            ax.axhline(limit, color='#AD5149', ls=(0, (4, 3)), lw=1, zorder=2)
            ax.set(xticks=range(3), xticklabels=METHODS, xlim=(-.45, 2.45), ylabel=ylabel, title=title)
            ax.tick_params(axis='x', labelsize=9, length=0, pad=7)
            ax.grid(axis='y', color='#E7E7E7', lw=.7)
            ax.set_axisbelow(True)
        axes[0].set_ylim(top=max(1.106, frame.peak.max()+.004))
        axes[1].set_yscale('symlog', linthresh=1)
        ymax = max(8, frame.duration.max()*1.35)
        axes[1].set_yticks([v for v in [0, 1, 5, 10, 30, 60, 120, 300, 600] if v < ymax])
        axes[1].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:g}'))
        axes[1].set_ylim(0, ymax)
        ax = axes[2]
        lines = [line for line in [54, 14, 31] if line in frame.line.unique()]
        rgba = np.ones((3, len(lines), 4))
        for i, method in enumerate(METHODS):
            for j, line in enumerate(lines):
                g = frame[frame.method.eq(method) & frame.line.eq(line)]
                passed = int(g.loc[g.status.eq('ok'), 'passes'].sum())
                rgba[i, j] = to_rgba(COLORS[i], .12+.30*passed/len(g))
                ax.text(j, i, f'{passed}/{len(g)}\n{100*passed/len(g):.0f}%',
                        ha='center', va='center', fontsize=10, linespacing=1.6)
        ax.imshow(rgba, aspect='auto', interpolation='none')
        ax.set(xticks=range(len(lines)), xticklabels=[labels[x] for x in lines],
               yticks=range(3), yticklabels=METHODS, xlabel='Maintenance branch (bus IDs)',
               title='Passing both reference checks')
        ax.tick_params(length=0, pad=7, labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for y in [.5, 1.5]:
            ax.axhline(y, color='white', lw=3)
        for x in np.arange(len(lines)-1)+.5:
            ax.axvline(x, color='white', lw=3)
        for letter, ax in zip('abc', axes):
            ax.text(-.17, 1.12, letter, transform=ax.transAxes, fontsize=13, fontweight='bold')
        fig.suptitle('Local sensitivity: 21 design points' if phase == 'local' else f'Independent validation: {n} paired scenarios', y=.99, fontsize=12)
        fig.text(.065, .035, 'Dots: individual scenarios; black marks: median and IQR. Dashed lines: 1.10 p.u. and 5 min references.', fontsize=9, color='#444444')
        for ext in ['png', 'pdf']:
            fig.savefig(ROOT/'figures'/f'{phase}_robustness_clean.{ext}', dpi=240)
        plt.close(fig)
        figures.append(ROOT/'figures'/f'{phase}_robustness_clean.png')
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell('# Operational robustness validation\n\nFixed trained controllers; no gain retuning. Local sensitivity and independent validation are separate. The duration criterion uses >10% of energized modeled buses outside 0.95-1.05 p.u.; offline buses are excluded. References: peak <=1.10 p.u. and duration <5 min. These are modeled-node checks, not customer-meter certification or a mathematical stability proof. Duration uses a linear scale below 1 min and logarithmic scale above it.'),
                nbf.v4.new_code_cell(f'from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\nroot = Path({str(ROOT)!r})\ndisplay(pd.read_csv(root/"summary.csv"))\npaired = pd.read_csv(root/"paired_metrics.csv")\ndisplay(paired.groupby(["phase", "baseline"])[["peak_reduction", "duration_reduction", "cost_ratio"]].agg(["min", "median", "max"]))')]
    for path in figures:
        nb.cells.append(nbf.v4.new_code_cell(f'display(Image(filename={str(path)!r}))'))
    nb.metadata.kernelspec = dict(display_name='Python 3', language='python', name='python3')
    NotebookClient(nb, timeout=180, kernel_name='python3', resources={'metadata': {'path': str(HERE)}}).execute()
    nbf.write(nb, HERE/NOTEBOOK_NAME)


if __name__ == '__main__':
    main()
