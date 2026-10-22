"""Render SI Fig. 10/12 from unchanged historical data; never copy into the paper."""

import argparse
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
from matplotlib.ticker import FuncFormatter
from matplotlib.text import Text
import numpy as np
import pandas as pd


WIDTH = 370.6102117061 / 72
METHODS = ("Linear", "Safe-DDPG", "RLC-FT")
COLORS = {"Linear": "#C87C75", "Safe-DDPG": "#89A6C4", "RLC-FT": "#779B6E"}
STYLES = {"Linear": (0, (4, 2)), "Safe-DDPG": (0, (2, 1.5)), "RLC-FT": "-"}
MARKERS = {"Linear": "s", "Safe-DDPG": "^", "RLC-FT": "o"}
BUS_COLORS = ["#577D9F", "#C87C75", "#739968", "#B59878", "#9380A7"]
INK = "#242A2D"
LIMIT = "#879187"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(array):
    return hashlib.sha256(np.asarray(array, dtype="<f8").tobytes()).hexdigest()


def configure():
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 7,
        "axes.labelsize": 7.3, "axes.titlesize": 7.8,
        "axes.edgecolor": INK, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.linewidth": 0.65, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "xtick.major.pad": 2.5,
        "ytick.major.pad": 2.5, "legend.fontsize": 6.8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "path.simplify": False,
    })


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E3E7E5", linewidth=0.45)


def header(fig, ax, letter, title, y, center=False):
    box = ax.get_position()
    fig.text(box.x0 - 0.045, y, letter, fontsize=9, weight="bold", va="center")
    fig.text(box.x0 + (box.width / 2 if center else 0), y, title,
             fontsize=7.8, weight="medium", va="center",
             ha="center" if center else "left", linespacing=1.15)


def save(fig, output, stem):
    # Fixed physical width is part of the typography contract; no tight cropping.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if box.width and box.height:
            assert box.x0 >= -1 and box.y0 >= -1, (stem, text.get_text(), box)
            assert box.x1 <= fig.bbox.width + 1 and box.y1 <= fig.bbox.height + 1, (stem, text.get_text(), box)
    fig.savefig(output / f"{stem}.pdf")
    fig.savefig(output / f"{stem}.png", dpi=300)
    plt.close(fig)


def stress_preview(data_root, output):
    source = data_root / "images/56bus/admittance_stress"
    summary_path = source / "56bus_admittance_stress_summary_100.csv"
    raw_path = source / "56bus_admittance_stress_raw_100.csv"
    summary = pd.read_csv(summary_path)
    raw = pd.read_csv(raw_path)
    factors = [1.0, 1.5, 2.0, 3.0, 5.0]
    assert len(summary) == 15
    np.testing.assert_array_equal(sorted(summary.stress_factor.unique()), factors)
    plotted = []
    fig = plt.figure(figsize=(WIDTH, 2.55))
    axes = [fig.add_axes([x, .23, .25, .52]) for x in [.075, .3975, .72]]
    for ax in axes:
        style_axes(ax)
        ax.set_xlim(.8, 5.25)
        ax.set_xticks(factors, ["1.0", "1.5", "2.0", "3.0", "5.0"])
    for j, method in enumerate(METHODS):
        rows = summary[summary.method == method].sort_values("stress_factor")
        np.testing.assert_array_equal(rows.n, [100] * 5)
        x = rows.stress_factor.to_numpy()
        for _, row in rows.iterrows():
            samples = raw[(raw.method == method) & (raw.stress_factor == row.stress_factor)]
            assert len(samples) == 100
            for key, raw_key in [("recovery", "recovery_time"), ("objective", "objective_cost")]:
                actual = np.quantile(samples[raw_key], [.25, .5, .75])
                expected = [row[f"{key}_q25"], row[f"{key}_median"], row[f"{key}_q75"]]
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)
            np.testing.assert_allclose((samples.status == "recovered").mean(), row.recovered_within_horizon)
        for ax, metric in zip(axes[:2], ["recovery", "objective"]):
            y = rows[f"{metric}_median"].to_numpy()
            low = rows[f"{metric}_q25"].to_numpy()
            high = rows[f"{metric}_q75"].to_numpy()
            ax.errorbar(x, y, yerr=np.vstack([y-low, high-y]),
                        color=COLORS[method], linestyle=STYLES[method],
                        marker=MARKERS[method], markersize=3, markeredgewidth=.8,
                        markerfacecolor="white", linewidth=1.25, elinewidth=.7,
                        capsize=1.8, capthick=.7, zorder=3+j)
            plotted.append({"method": method, "metric": metric, "x": x.tolist(),
                            "median": y.tolist(), "q25": low.tolist(), "q75": high.tolist()})
        rate = 100 * rows.recovered_within_horizon.to_numpy()
        line, = axes[2].plot(x, rate, color=COLORS[method], linestyle=STYLES[method],
                            marker=MARKERS[method], markersize=3, markeredgewidth=.8,
                            markerfacecolor="white", linewidth=1.25, zorder=3+j)
        np.testing.assert_array_equal(line.get_ydata(), rate)
        plotted.append({"method": method, "metric": "recovered_percent", "x": x.tolist(), "y": rate.tolist()})
    axes[0].set_ylim(0, 210)
    axes[0].set_yticks([0, 50, 100, 150, 200])
    axes[0].set_ylabel("Steps", labelpad=4)
    axes[1].set_ylim(0, 6300)
    axes[1].set_yticks([0, 2000, 4000, 6000], ["0", "2,000", "4,000", "6,000"])
    axes[2].set_ylim(0, 105)
    axes[2].set_yticks([0, 25, 50, 75, 100])
    axes[2].set_ylabel("Scenarios (%)", labelpad=4)
    titles = ["Voltage recovery\ntime", "Total objective\ncost", "Recovered within\nhorizon"]
    for ax, letter, title in zip(axes, "abc", titles):
        header(fig, ax, letter, title, .825, center=True)
    handles = [Line2D([], [], color=COLORS[m], ls=STYLES[m], marker=MARKERS[m],
                      ms=3, mfc="white", mew=.8, lw=1.25, label=m) for m in METHODS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .995),
               ncol=3, frameon=False, handlelength=2.3, columnspacing=2)
    fig.text(.525, .075, "Admittance scaling factor", ha="center", fontsize=7.3)
    save(fig, output, "fig10_admittance_stress_preview")
    return {"inputs": {str(p): digest(p) for p in [summary_path, raw_path]},
            "raw_summary_agree": True, "plotted_values": plotted,
            "aggregation": "Unchanged median and IQR; original recovered-within-horizon fraction."}


def load_operational(repo):
    notebook = repo / "experiments/validated/real_world_evaluation/test_real_world.ipynb"
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    cell = nb["cells"][10]
    spec = next(o["data"]["application/vnd.plotly.v1+json"]
                for o in cell["outputs"] if "application/vnd.plotly.v1+json" in o.get("data", {}))
    cache = repo / "data/reference/imperfect_results.pkl.gz"
    with gzip.open(cache, "rb") as stream:
        data = pickle.load(stream)
    assert data["topology_change_step"] == 7000
    assert data["loss_rate"] == .5 and data["max_delay"] == 100
    buses = (np.asarray(data["injection_bus"]) + 1).tolist()
    assert buses == [18, 21, 30, 45, 53]
    series = []
    for info, offset in [("perfect_info", 0), ("imperfect_info", 7)]:
        states = np.asarray(data[info]["state_list"])
        for i, bus in enumerate(buses):
            tr = spec["data"][offset+i]
            x, y = np.asarray(tr["x"]), np.asarray(tr["y"])
            np.testing.assert_array_equal(x, np.arange(14421))
            np.testing.assert_array_equal(y, states[:14421, i])
            z = spec["data"][14 + 2*i + (info == "imperfect_info")]
            np.testing.assert_array_equal(z["x"], x[6850:7150])
            np.testing.assert_array_equal(z["y"], y[6850:7150])
            series.append({"info": info, "bus": bus, "x": x, "y": y})
    return spec, series, {"inputs": {str(p): digest(p) for p in [notebook, cache]},
                          "notebook_cell": 11, "cache_matches_all_saved_traces": True,
                          "samples_per_daily_trace": 14421, "zoom_steps": [6850, 7149],
                          "event_step": 7000,
                          "trace_hashes": [{"bus": s["bus"], "info": s["info"],
                                             "x": array_digest(s["x"]), "y": array_digest(s["y"])} for s in series]}


def operational_preview(repo, output, reconstruct):
    spec, series, record = load_operational(repo)
    fig = plt.figure(figsize=(WIDTH, 4.35))
    axes = [fig.add_axes([.105, .595, .38, .28]),
            fig.add_axes([.60, .595, .38, .28]),
            fig.add_axes([.105, .105, .875, .285])]
    for ax in axes:
        style_axes(ax)
        for limit in [.95, 1.05]:
            ax.axhline(limit, color=LIMIT, lw=.7, ls=(0, (4, 3)), zorder=1)
        ax.axvline(7000, color="#A56862", lw=.75, ls=(0, (2, 2.5)), zorder=2)
    for ax, info, letter, title in zip(axes[:2], ["perfect_info", "imperfect_info"],
                                      "ab", ["Perfect information", "Imperfect information"]):
        for s, color in zip([s for s in series if s["info"] == info], BUS_COLORS):
            ls = "-" if info == "perfect_info" else (0, (3, 1.8))
            line, = ax.plot(s["x"], s["y"], color=color, lw=1.0, ls=ls)
            np.testing.assert_array_equal(line.get_ydata(), s["y"])
        ax.set_xlim(0, 14420)
        ax.set_ylim(.92, 1.112)
        ax.set_yticks([.95, 1.00, 1.05, 1.10])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
        ax.set_xticks([0, 5000, 10000, 14420], ["0", "5,000", "10,000", "14,420"])
        ax.get_xticklabels()[-1].set_ha("right")
        header(fig, ax, letter, title, .905)
    axes[0].set_ylabel("Bus voltage (p.u.)", labelpad=5)
    fig.text(.54, .525, "Time step", ha="center", fontsize=7.3)
    zoom = axes[2]
    for s in series:
        color = BUS_COLORS[[18, 21, 30, 45, 53].index(s["bus"])]
        line, = zoom.plot(s["x"][6850:7150], s["y"][6850:7150], color=color,
                          ls="-" if s["info"] == "perfect_info" else (0, (3, 1.8)), lw=1.1)
        np.testing.assert_array_equal(line.get_ydata(), s["y"][6850:7150])
    zoom.set_xlim(6850, 7149)
    zoom.set_ylim(spec["layout"]["yaxis3"]["range"])
    zoom.set_xticks([6850, 6900, 6950, 7000, 7050, 7100],
                   ["6,850", "6,900", "6,950", "7,000", "7,050", "7,100"])
    zoom.set_yticks([1.00, 1.02, 1.04, 1.06, 1.08, 1.10])
    zoom.set_xlabel("Time step", labelpad=5)
    zoom.set_ylabel("Bus voltage (p.u.)", labelpad=5)
    header(fig, zoom, "c", "Response around PV connection", .475)
    handles = [Line2D([], [], color=c, lw=1.5, label=f"Bus {bus}")
               for c, bus in zip(BUS_COLORS, [18, 21, 30, 45, 53])]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, 1.0),
               ncol=5, frameon=False, handlelength=1.7, columnspacing=1.25)
    keys = [Line2D([], [], color=INK, lw=1.1, ls=ls, label=label)
            for label, ls in [("Perfect information", "-"),
                              ("Imperfect information", (0, (3, 1.8)))]]
    fig.legend(handles=keys, loc="center", bbox_to_anchor=(.54, .427),
               ncol=2, frameon=False, handlelength=2.5, columnspacing=2)
    save(fig, output, "fig12_imperfect_information_preview")
    if reconstruct:
        import plotly.graph_objects as go
        # Removed Plotly-5 template entry is unrelated to these scatter traces.
        spec["layout"]["template"]["data"].pop("heatmapgl", None)
        go.Figure(spec).write_image(output / "originals/fig12_reconstructed_from_traces.pdf",
                                   width=2160, height=900, scale=1.5)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True,
                        help="Preserved original PDFs, independent of the current paper assets.")
    parser.add_argument("--reconstruct-original", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "originals").mkdir(exist_ok=True)
    originals = [args.paper / "fig/56bus_admittance_stress_summary_100.pdf",
                 args.paper / "fig/imperfect_info_zoom_loss50_delay100.pdf"]
    paper_hashes = {str(p): digest(p) for p in originals}
    original_hashes = {str(p): digest(args.reference_dir / p.name) for p in originals}
    for p in originals:
        reference = args.reference_dir / p.name
        dest = args.output / "originals" / p.name
        if dest.exists():
            assert digest(dest) == digest(reference), "Preserved original differs; use a separate output directory."
        else:
            shutil.copy2(reference, dest)
    configure()
    manifest = {"status": "preview_only_not_adopted", "source": str(Path(__file__).resolve()),
                "source_sha256": digest(__file__), "display_width_pdf_pt": WIDTH * 72,
                "original_paper_assets": original_hashes,
                "fig10": stress_preview(args.data_root, args.output),
                "fig12": operational_preview(args.repo, args.output, args.reconstruct_original)}
    assert paper_hashes == {str(p): digest(p) for p in originals}
    manifest["paper_assets_unchanged_by_render"] = True
    manifest["reference_dir"] = str(args.reference_dir.resolve())
    manifest["preview_outputs"] = {p.name: digest(p) for p in args.output.glob("*preview.*")}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "originals_unchanged": True,
                      "fig10_raw_summary_check": True, "fig12_trace_cache_check": True}, indent=2))


if __name__ == "__main__":
    main()
