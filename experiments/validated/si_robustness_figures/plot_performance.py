"""SI Fig. 1: original 123-bus statistics with the main-text curved-arrow style."""

import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Patch
import numpy as np


WIDTH, HEIGHT = 370.6102117061, 158
ANCHOR = "# Supplementary figure: paired 123-bus comparison with component-wise transient costs."
MAIN_ANCHOR = "# Paper figure: paired 56-bus comparison with component-wise transient costs."


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_cell(path, anchor):
    book = json.loads(path.read_text(encoding="utf-8"))
    matches = [(i, c) for i, c in enumerate(book["cells"])
               if "".join(c.get("source", [])).startswith(anchor)]
    assert len(matches) == 1
    index, cell = matches[0]
    source = "".join(cell["source"])
    return source, {"path": str(path), "sha256": sha(path), "cell_one_based": index + 1,
                    "source_sha256": hashlib.sha256(source.encode()).hexdigest()}


def evidence(repo, data_root):
    source, record = selected_cell(repo / "test_123_performance.ipynb", ANCHOR)
    tree = ast.parse(source)
    statements = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "plt.rcParams.update":
            break
        if isinstance(node, ast.ImportFrom) and node.module == "config":
            continue
        statements.append(node)
    context = {"Config": SimpleNamespace(data_path=str(data_root))}
    exec(compile(ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[])),
                 str(repo / "test_123_performance.ipynb") + "#statistics_only", "exec"), context)
    main, main_record = selected_cell(repo / "test_56bus_performance.ipynb", MAIN_ANCHOR)
    for node in ast.parse(main).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ["COLORS", "EDGES"]:
                    context[target.id] = ast.literal_eval(node.value)
    return context, record, main_record


def render(context, output):
    methods, colors, edges = [context[n] for n in ["METHODS", "COLORS", "EDGES"]]
    stats, reductions = context["normalized_stats"], context["reduction_stats"]
    plt.rcParams.update({"font.family": "Arial", "font.size": 6.5,
                         "axes.labelsize": 7.3, "axes.titlesize": 7.3,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "axes.linewidth": .6, "pdf.fonttype": 42,
                         "figure.facecolor": "white", "text.color": "#171717"})
    fig = plt.figure(figsize=(WIDTH / 72, HEIGHT / 72))
    x = np.arange(3, dtype=float)
    for i, metric in enumerate(context["series"]):
        left, bottom, width, height = 35 + 85 * i, 34, 75, 93
        ax = fig.add_axes([left / WIDTH, bottom / HEIGHT, width / WIDTH, height / HEIGHT])
        med = np.asarray([stats[metric][m]["median"] for m in methods])
        q25 = np.asarray([stats[metric][m]["q25"] for m in methods])
        q75 = np.asarray([stats[metric][m]["q75"] for m in methods])
        err = np.vstack([med - q25, q75 - med])
        err[:, 0] = 0
        ax.bar(x, med, yerr=err, width=.38, color=[colors[m] for m in methods],
               edgecolor=[edges[m] for m in methods], linewidth=.45, capsize=1.8,
               error_kw={"elinewidth": .65, "ecolor": "#4F4F4F", "capthick": .65}, zorder=3)
        short_y = .5 * (med[1] + q75[1])
        for start, end, curve, edge, arrow_style in [
            ((.30, 1.055), (2.08, q75[2] + .045), -.34, "Linear",
             "Simple,tail_width=.55,head_width=4.4,head_length=4.8"),
            ((1.10, short_y), (1.92, q75[2] + .025), -.25, "Safe-DDPG",
             "Simple,tail_width=.53,head_width=4.1,head_length=4.6"),
        ]:
            ax.add_patch(FancyArrowPatch(start, end, connectionstyle=f"arc3,rad={curve}",
                                         arrowstyle=arrow_style, mutation_scale=1,
                                         color=edges[edge], linewidth=0, alpha=.94, zorder=5))
        ax.text(.85, 1.01 + .25 * q75[2], f"\N{MINUS SIGN}{reductions[metric]['Linear']:.1f}%",
                ha="center", va="center", rotation=-15, rotation_mode="anchor",
                fontsize=6.8, fontweight="bold", color=edges["Linear"], zorder=8)
        ax.text(1.30 if i in [0, 2] else 1.32,
                .5 * (short_y + q75[2]) + (.2675 if i in [0, 2] else .2225),
                f"\N{MINUS SIGN}{reductions[metric]['Safe-DDPG']:.1f}%",
                ha="center", va="center", rotation=-22 if i in [0, 2] else -15,
                rotation_mode="anchor", fontsize=6.8, fontweight="bold",
                color=edges["Safe-DDPG"], zorder=8)
        ax.text(0, 1.012, "1.00", ha="center", va="bottom", fontsize=6.5)
        ax.axhline(1, color="#858585", linestyle=(0, (4, 3)), linewidth=.55, zorder=1)
        ax.set(xlim=(-.5, 2.5), ylim=(0, 1.34), xticks=[], yticks=np.arange(0, 1.21, .2))
        ax.tick_params(axis="y", width=.6, length=2.5, pad=2.2, labelleft=i == 0)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#D9DDE1", linewidth=.45, alpha=.72)
        ax.spines[["top", "right"]].set_visible(False)
        fig.text((left + width / 2) / WIDTH, 144 / HEIGHT, metric, fontsize=7.3,
                 ha="center", va="center")
        fig.text((left - 8) / WIDTH, 139 / HEIGHT, chr(97 + i), fontsize=9,
                 fontweight="bold", ha="left", va="center")
        if i == 0:
            ax.set_ylabel("Normalized value (Linear = 1)", labelpad=5)
    handles = [Patch(facecolor=colors[m], edgecolor=edges[m], linewidth=.45, label=m) for m in methods]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.53, 12 / HEIGHT),
               ncol=3, frameon=False, handlelength=1.35, columnspacing=1.6, fontsize=6.8)
    stem = output / "fig01_123bus_performance_preview"
    fig.savefig(stem.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return stem.with_suffix(".pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--originals", type=Path, help="Immutable pre-restyling PDF directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    originals = args.output / "originals"
    originals.mkdir(exist_ok=True)
    original = (args.originals if args.originals else args.paper / "fig") / "performance_normalized_reduction_vs_linear_123bus.pdf"
    backup = originals / original.name
    if backup.exists():
        assert sha(backup) == sha(original)
    else:
        shutil.copy2(original, backup)
    reference = args.paper / "fig/performance_normalized_reduction_vs_linear_5000.pdf"
    if not (originals / reference.name).exists():
        shutil.copy2(reference, originals / reference.name)
    context, source, style = evidence(args.repo, args.data_root)
    target = render(context, args.output)
    hashes = {str(args.data_root / "cache/notebook_results" / f"{n}.pkl.gz"):
              sha(args.data_root / "cache/notebook_results" / f"{n}.pkl.gz")
              for n in ["123bus_linear_metrics", "123bus_safe_ddpg_metrics", "123bus_rlcft_metrics"]}
    record = {"status": "preview_only_not_adopted", "source": source, "style_reference": style,
              "main_pdf_sha256": sha(reference), "original_pdf_sha256": sha(original),
              "source_code_sha256": sha(__file__), "input_sha256": hashes,
              "normalized_stats": context["normalized_stats"], "paired_reduction_percent": context["reduction_stats"],
              "series_sha256": {metric: {method: hashlib.sha256(np.asarray(array, dtype="<f8").tobytes()).hexdigest()
                                          for method, array in values.items()} for metric, values in context["series"].items()},
              "dimensions_pt": [WIDTH, HEIGHT], "output_sha256": sha(target),
              "typography_pt": {"ticks": 6.5, "reduction_labels": 6.8, "legend": 6.8,
                                 "titles": 7.3, "axis": 7.3, "panel_letters": 9},
              "note": "Only style/layout changed. Original 123-bus action-cost weight 0.1, medians, IQRs and paired reductions retained."}
    assert sha(backup) == sha(original)
    (args.output / "fig01_manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(target), "reductions": record["paired_reduction_percent"]}, indent=2))


if __name__ == "__main__":
    main()
