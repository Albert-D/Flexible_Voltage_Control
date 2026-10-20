"""Restyle SI recovery distributions and saved voltage trajectories only."""

import argparse
import base64
import copy
import gzip
import hashlib
import json
import pickle
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


WIDTH = 370.6102117061
METHODS = ["Linear", "Safe-DDPG", "RLC-FT"]
COLORS = {"Linear": "#C87C75", "Safe-DDPG": "#7D9ABA", "RLC-FT": "#779B6E"}
STYLES = {"Linear": (0, (4, 2)), "Safe-DDPG": (0, (1, 1.6)), "RLC-FT": "-"}
MARKERS = {"Linear": "s", "Safe-DDPG": "^", "RLC-FT": "o"}
ASSETS = ["recovery_over_time_distribution_56_123.pdf", "nature_enhanced_trajectory.pdf"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(array):
    return hashlib.sha256(np.asarray(array, dtype="<f8").tobytes()).hexdigest()


def decode(array):
    if isinstance(array, dict) and "bdata" in array:
        result = np.frombuffer(base64.b64decode(array["bdata"]), dtype=array["dtype"])
        if "shape" in array:
            result = result.reshape(tuple(int(s) for s in str(array["shape"]).split(",")))
        return result
    return np.asarray(array, dtype=float)


def notebook_cell(path, anchor):
    book = json.loads(path.read_text(encoding="utf-8"))
    matches = [(i, c) for i, c in enumerate(book["cells"])
               if anchor in "".join(c.get("source", []))]
    if len(matches) != 1:
        raise ValueError((path, anchor, len(matches)))
    index, cell = matches[0]
    return cell, {"path": str(path), "sha256": sha(path), "zero_based_cell": index,
                  "cell_id": cell.get("id"), "anchor": anchor,
                  "cell_source_sha256": hashlib.sha256("".join(cell["source"]).encode()).hexdigest()}


def keep_original(source, target):
    if target.exists():
        if sha(source) != sha(target):
            raise RuntimeError(f"Backup is immutable: {target}")
    else:
        shutil.copy2(source, target)


def setup():
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 6.5, "axes.labelsize": 7.3,
        "axes.titlesize": 7.8, "legend.fontsize": 6.8,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.linewidth": 0.65, "xtick.major.width": 0.65,
        "ytick.major.width": 0.65, "xtick.major.size": 2.6,
        "ytick.major.size": 2.6, "xtick.major.pad": 2.5,
        "ytick.major.pad": 2.5, "text.color": "#222222",
        "axes.labelcolor": "#222222", "xtick.color": "#222222",
        "ytick.color": "#222222", "axes.edgecolor": "#333333",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "path.simplify": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
    })


def axes_at(fig, height, left, bottom, width, panel_height):
    ax = fig.add_axes([left / WIDTH, bottom / height, width / WIDTH, panel_height / height])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E4E6E7", linewidth=0.45)
    return ax


def header(fig, height, letter, title, left, center, y):
    fig.text((left - 9) / WIDTH, y / height, letter, fontsize=9,
             fontweight="bold", ha="left", va="center")
    fig.text(center / WIDTH, y / height, title, fontsize=7.8,
             ha="center", va="center")


def handles(markers=False):
    return [Line2D([], [], color=COLORS[m], linestyle=STYLES[m],
                   linewidth=1.25 if m == "RLC-FT" else 1.05,
                   marker=MARKERS[m] if markers else None, markersize=2.2,
                   label=m) for m in METHODS]


def export(fig, output, stem):
    pdf = output / f"{stem}.pdf"
    fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(output / f"{stem}.png", dpi=300)
    plt.close(fig)
    return {pdf.name: sha(pdf), f"{stem}.png": sha(output / f"{stem}.png")}


def recovery(args, manifest):
    notebook = args.repo / "test_recovery_over_time_distribution.ipynb"
    _, source = notebook_cell(notebook, "# Paper figure: comparison with external baselines only")
    manifest["sources"]["recovery"] = source
    specs = {"56bus": [("56bus_linear", "base_succ_list"),
                        ("56bus_safe_ddpg", "safe_succ_list"), ("56bus_rlcft", "success_list")],
             "123bus": [("123bus_linear_metrics", "base_succ_list"),
                         ("123bus_safe_ddpg_metrics", "safe_succ_list"),
                         ("123bus_rlcft_metrics", "success_list")]}
    csv_path = args.data_root / "images/recovery_over_time_distribution/recovery_over_time_distribution_curves.csv"
    csv = pd.read_csv(csv_path)
    manifest["inputs"][str(csv_path)] = sha(csv_path)
    steps = np.arange(201)
    curves = {}
    checks = {}
    for system, method_specs in specs.items():
        curves[system] = {}
        checks[system] = {}
        for method, (basename, key) in zip(METHODS, method_specs):
            path = args.data_root / "cache/notebook_results" / f"{basename}.pkl.gz"
            manifest["inputs"][str(path)] = sha(path)
            with gzip.open(path, "rb") as stream:
                payload = pickle.load(stream)
            success = np.asarray(payload.get("data", payload)[key]).reshape(-1, 2)
            times = np.full(5000, 200.0)
            recovered = np.zeros(5000, dtype=bool)
            ids = success[:, 0].astype(int)
            valid = (ids >= 0) & (ids < 5000)
            times[ids[valid]] = success[valid, 1]
            recovered[ids[valid]] = True
            values = np.asarray([100.0 * np.mean(recovered & (times <= t)) for t in steps])
            saved = csv[(csv.system == system) & (csv.method == method)].sort_values("step")
            np.testing.assert_array_equal(saved.step.to_numpy(), steps)
            np.testing.assert_allclose(saved.recovered_percent.to_numpy(), values, atol=1e-12, rtol=0)
            assert np.all(np.diff(values) >= 0)
            curves[system][method] = (steps, values)
            checks[system][method] = {"curve_sha256": array_sha(values),
                                     "n_scenarios": 5000,
                                     "not_recovered_at_horizon": int((~recovered).sum()),
                                     "at_50_steps_percent": float(values[50]),
                                     "at_200_steps_percent": float(values[-1])}
    height = 185
    fig = plt.figure(figsize=(WIDTH / 72, height / 72))
    for left, system, letter, title in [(43, "56bus", "a", "56-bus system"),
                                         (225, "123bus", "b", "123-bus system")]:
        ax = axes_at(fig, height, left, 34, 136, 115)
        for method in METHODS:
            x, y = curves[system][method]
            line, = ax.plot(x, y, color=COLORS[method], linestyle=STYLES[method],
                            linewidth=1.35 if method == "RLC-FT" else 1.1,
                            zorder=4 if method == "RLC-FT" else 3)
            np.testing.assert_array_equal(line.get_ydata(), y)
        if args.recovery_xscale == "log":
            ax.set_xscale("log")
            limits = (1, 200)
            ticks = [1, 10, 100] if args.recovery_log_ticks == "decades" else [1, 2, 5, 10, 20, 50, 100, 200]
        elif args.recovery_xscale == "symlog":
            ax.set_xscale("symlog", linthresh=1, linscale=0.5, base=10)
            limits, ticks = (0, 200), [0, 1, 2, 5, 10, 20, 50, 100, 200]
        else:
            limits, ticks = (0, 200), [0, 50, 100, 150, 200]
        ax.set(xlim=limits, ylim=(-2, 102), xticks=ticks,
               yticks=[0, 20, 40, 60, 80, 100],
               xlabel="Control steps (log scale)" if args.recovery_xscale == "log" else "Control steps")
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        if args.recovery_xscale == "log" and args.recovery_log_ticks == "decades":
            ax.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=30))
            ax.xaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(
                lambda value, position: "200" if np.isclose(value, 200) else ""))
            ax.tick_params(axis="x", which="minor", length=1.6, width=0.5,
                           labelsize=6.5, pad=3.5)
        else:
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        if system == "56bus":
            ax.set_ylabel("Scenarios recovered (%)")
        else:
            ax.tick_params(labelleft=False)
        header(fig, height, letter, title, left, left + 68, 158)
    fig.legend(handles=handles(), loc="center", bbox_to_anchor=(0.5, 177 / height),
               ncol=3, frameon=False, handlelength=2.6, handletextpad=0.6,
               columnspacing=2.1, borderaxespad=0)
    manifest["checks"]["recovery"] = checks
    manifest["checks"]["recovery_display"] = {
        "xscale": args.recovery_xscale, "display_xlim": list(limits), "ticks": ticks,
        "all_201_original_samples_preserved": True,
        "zero_step_visible": args.recovery_xscale != "log",
        "symlog_linthresh": 1 if args.recovery_xscale == "symlog" else None,
        "symlog_linscale": 0.5 if args.recovery_xscale == "symlog" else None,
        "log_tick_style": args.recovery_log_ticks if args.recovery_xscale == "log" else None,
        "endpoint_label": 200 if args.recovery_xscale == "log" and args.recovery_log_ticks == "decades" else None,
    }
    suffix = "" if args.recovery_xscale == "linear" else f"_{args.recovery_xscale}"
    manifest["outputs"].update(export(fig, args.output, f"fig02_recovery_distribution{suffix}_preview"))
    if args.reference_run:
        reference = args.reference_run / "originals/fig02_reconstructed_from_cache.pdf"
        manifest["inputs"][str(reference)] = sha(reference)
        return curves
    # Reconstruct the historical geometry using only the audited plotting cell.
    context = {"__name__": "__style_reference__"}
    cell, _ = notebook_cell(notebook, "# Paper figure: comparison with external baselines only")
    original_source = "".join(cell["source"])
    original_source = original_source.replace("from config import Config", "")
    original_source = original_source.replace("OUTPUT_DIR = Path(Config.data_path) / \"images\" / \"recovery_over_time_distribution\"",
                                               "OUTPUT_DIR = Path(reference_output)")
    original_source = original_source.replace('output_stem = OUTPUT_DIR / "recovery_over_time_distribution_56_123"',
                                               'output_stem = OUTPUT_DIR / "fig02_reconstructed_from_cache"')
    from types import SimpleNamespace
    context.update(Config=SimpleNamespace(data_path=str(args.data_root)),
                   reference_output=str(args.output / "originals"))
    with plt.rc_context(matplotlib.rcParamsDefault):
        exec(compile(original_source, str(notebook) + "#paper_plot_only", "exec"), context)
    plt.close("all")
    return curves


def trajectory(args, manifest):
    notebook = args.repo / "new_trajectory.ipynb"
    cell, source = notebook_cell(notebook, "# Figure: Enhanced Nature-Style Voltage Trajectory Comparison")
    saved = [o.get("data", {}).get("application/vnd.plotly.v1+json") for o in cell.get("outputs", [])]
    payload = next(p for p in saved if p)
    assert len(payload["data"]) == 27
    manifest["sources"]["trajectory"] = source
    manifest["checks"]["trajectory_arrays"] = [
        {"name": t.get("name", "min-max band"), "axes": t.get("xaxis", "x"),
         "point_count": len(decode(t["x"])), "x_sha256": array_sha(decode(t["x"])),
         "y_sha256": array_sha(decode(t["y"]))} for t in payload["data"]]
    height = 350
    fig = plt.figure(figsize=(WIDTH / 72, height / 72))
    titles = ["Bus 18", "Bus 21", "Bus 30", "Bus 45", "Bus 53", "Mean voltage"]
    for i, title in enumerate(titles):
        left = 43 if i % 2 == 0 else 225
        bottom = [236, 135, 34][i // 2]
        ax = axes_at(fig, height, left, bottom, 136, 77)
        axis_key = "x" if i == 0 else f"x{i + 1}"
        traces = [t for t in payload["data"] if t.get("xaxis", "x") == axis_key]
        bands = [t for t in traces if t.get("fill") == "toself"]
        if i == 5:
            assert len(bands) == 3
            for band, method in zip(bands, ["RLC-FT", "Linear", "Safe-DDPG"]):
                ax.fill(decode(band["x"]), decode(band["y"]), color=COLORS[method],
                        alpha=0.1, linewidth=0, zorder=1)
        limit = next(t for t in traces if t.get("name") == "Upper limit")
        ax.plot(decode(limit["x"]), decode(limit["y"]), color="#555555", linewidth=0.75, zorder=2)
        for method in METHODS:
            trace = next(t for t in traces if t.get("name") == method)
            x, y = decode(trace["x"]), decode(trace["y"])
            line, = ax.plot(x, y, color=COLORS[method], linestyle=STYLES[method],
                            linewidth=1.15 if method == "RLC-FT" else 1.0,
                            marker=MARKERS[method] if i < 5 else None, markersize=2.1,
                            markeredgewidth=0.3, zorder=4 if method == "RLC-FT" else 3)
            np.testing.assert_array_equal(line.get_xdata(), x)
            np.testing.assert_array_equal(line.get_ydata(), y)
        ykey = "yaxis" if i == 0 else f"yaxis{i + 1}"
        lo, hi = payload["layout"][ykey]["range"]
        ticks = np.arange(np.ceil(lo / .3) * .3, hi + 1e-8, .3)
        ax.set(xlim=(0, 15), ylim=(lo, hi), xticks=[0, 5, 10, 15], yticks=ticks)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
        if i % 2 == 0:
            ax.set_ylabel("Voltage (kV)")
        if i // 2 == 2:
            ax.set_xlabel("Control steps")
        else:
            ax.tick_params(labelbottom=False)
        header(fig, height, chr(97 + i), title, left, left + 68, bottom + 86)
    key = handles(markers=True) + [Line2D([], [], color="#555555", linewidth=.75, label="Upper limit")]
    fig.legend(handles=key, loc="center", bbox_to_anchor=(.5, 337 / height), ncol=4,
               frameon=False, handlelength=2.3, handletextpad=.55, columnspacing=1.3,
               borderaxespad=0)
    manifest["outputs"].update(export(fig, args.output, "fig03_voltage_trajectory_preview"))
    # Plotly's saved display data already contain the original interpolation.
    import plotly.graph_objects as go
    historical = copy.deepcopy(payload)
    historical.get("layout", {}).get("template", {}).get("data", {}).pop("heatmapgl", None)
    go.Figure(historical).write_image(
        args.output / "originals/fig03_reconstructed_from_saved_traces.pdf", scale=4,
        width=payload["layout"]["width"], height=payload["layout"]["height"])
    manifest["checks"]["trajectory"] = {"saved_trace_count": 27, "method_curves": 18,
                                          "reference_lines": 6, "min_max_bands": 3,
                                          "display_xlim": [0, 15], "y_ranges_unchanged": True,
                                          "new_smoothing_or_resampling": False,
                                          "simulations_executed": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--originals", type=Path,
                        help="Immutable original PDF directory for reproduction after SI adoption.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", choices=["all", "recovery", "trajectory"], default="all")
    parser.add_argument("--recovery-xscale", choices=["linear", "log", "symlog"], default="linear")
    parser.add_argument("--recovery-log-ticks", choices=["dense", "decades"], default="decades",
                        help="Decade-spaced main labels, small minor ticks and a labeled 200-step endpoint.")
    parser.add_argument("--reference-run", type=Path,
                        help="Reuse an already checked historical reconstruction for recovery-only styling.")
    args = parser.parse_args()
    if args.reference_run and args.figure != "recovery":
        parser.error("--reference-run is supported for --figure recovery only")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "originals").mkdir(exist_ok=True)
    manifest = {"scope": "SI style-only previews; not adopted", "width_pt": WIDTH,
                "paper_assets": {}, "paper_targets": {}, "paper_before": {},
                "sources": {}, "inputs": {}, "checks": {}, "outputs": {},
                "typography_pt": {"ticks": 6.5, "legend": 6.8, "axes": 7.3,
                                  "titles": 7.8, "panel_letters": 9},
                "colors": COLORS, "source_sha256": sha(__file__)}
    manifest["figure_selection"] = args.figure
    manifest["reference_run"] = str(args.reference_run) if args.reference_run else None
    assets = ASSETS if args.figure == "all" else [ASSETS[0 if args.figure == "recovery" else 1]]
    for name in assets:
        paper_target = args.paper / "fig" / name
        original = args.originals / name if args.originals else paper_target
        keep_original(original, args.output / "originals" / name)
        manifest["paper_assets"][str(original)] = sha(original)
        suffix = "" if args.recovery_xscale == "linear" else f"_{args.recovery_xscale}"
        candidate = (f"fig02_recovery_distribution{suffix}_preview.pdf" if name == ASSETS[0]
                     else "fig03_voltage_trajectory_preview.pdf")
        manifest["paper_targets"][str(paper_target)] = {"candidate": candidate,
                                                       "original_sha256": sha(original)}
        manifest["paper_before"][str(paper_target)] = sha(paper_target)
    if args.figure in ["all", "recovery"]:
        setup()
        recovery(args, manifest)
    if args.figure in ["all", "trajectory"]:
        setup()
        trajectory(args, manifest)
    for path, expected in manifest["paper_assets"].items():
        assert sha(path) == expected
    for path, expected in manifest["paper_before"].items():
        assert sha(path) == expected
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"outputs": manifest["outputs"],
                      "recovery_display": manifest["checks"].get("recovery_display"),
                      "trajectory": manifest["checks"].get("trajectory")}, indent=2))


if __name__ == "__main__":
    main()
