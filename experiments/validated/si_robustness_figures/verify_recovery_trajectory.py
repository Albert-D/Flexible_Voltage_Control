"""Verify data provenance, historical plot geometry and preview typography."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pdfplumber


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_paths(old, new, groups, tolerance):
    assert len(old) == len(new)
    for a, b in zip(old, new):
        assert len(a["pts"]) == len(b["pts"])
        assert a["stroking_color"] == b["stroking_color"]
        assert (a["fill"], a["stroke"]) == (b["fill"], b["stroke"])
    residuals = []
    for begin, end in groups:
        a = np.concatenate([c["pts"] for c in old[begin:end]])
        b = np.concatenate([c["pts"] for c in new[begin:end]])
        design = np.c_[a, np.ones(len(a))]
        transform = np.linalg.lstsq(design, b, rcond=None)[0]
        residual = float(np.max(np.abs(design @ transform - b)))
        assert residual < tolerance, (begin, end, residual)
        residuals.append(residual)
    return {"matching_path_count": len(old),
            "matching_vertices": sum(len(c["pts"]) for c in old),
            "per_axes_max_affine_residual_pt": residuals}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--require-adopted", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
    report = {"originals_unchanged": True, "input_hashes_unchanged": True,
              "manuscript_not_replaced": True, "typography": {}, "historical_geometry": {}}
    for path, expected in manifest["paper_assets"].items():
        assert sha(path) == expected
        assert sha(args.output / "originals" / Path(path).name) == expected
    states = {}
    for path, info in manifest.get("paper_targets", {}).items():
        current = sha(path)
        candidate = sha(args.output / info["candidate"])
        states[Path(path).name] = ("adopted" if current == candidate else
                                  "original" if current == info["original_sha256"] else "another_revision")
        if args.require_adopted:
            assert states[Path(path).name] == "adopted"
    if args.require_adopted:
        assert states, "Adoption verification requires a manifest with explicit paper targets."
    report["paper_asset_states"] = states
    if states:
        report["manuscript_not_replaced"] = all(v == "original" for v in states.values())
    for path, expected in manifest["inputs"].items():
        assert sha(path) == expected
    for source in manifest["sources"].values():
        assert sha(source["path"]) == source["sha256"]
    for name, expected in manifest["outputs"].items():
        assert sha(args.output / name) == expected
        if not name.endswith(".pdf"):
            continue
        with pdfplumber.open(args.output / name) as pdf:
            page = pdf.pages[0]
            np.testing.assert_allclose(page.width, manifest["width_pt"], atol=1e-6)
            visible = [c for c in page.chars if c["text"].strip()]
            upright = [c for c in visible if c["upright"]]
            sizes = sorted({round(c["size"], 4) for c in upright})
            assert min(sizes) >= 6.49
            outside = [c["text"] for c in visible if c["x0"] < -.2 or c["x1"] > page.width+.2
                       or c["top"] < -.2 or c["bottom"] > page.height+.2]
            assert not outside, (name, outside)
            report["typography"][name] = {"width_pt": page.width, "height_pt": page.height,
                                         "upright_font_sizes_pt": sizes,
                                         "fonts": sorted({c["fontname"] for c in visible}),
                                         "text_inside_page": True}
    roots = args.output / "originals"
    reference_run = manifest.get("reference_run")
    if reference_run:
        reference = Path(reference_run)
        baseline = json.loads((reference / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["checks"]["recovery"] == baseline["checks"]["recovery"]
        report["same_recovery_data_as_linear_preview"] = True
        roots = reference / "originals"
    recovery_colors = [0.7411764706, (.3568627451, .5529411765, .7215686275),
                       (.1058823529, .6196078431, .4666666667)]
    curves = []
    if "recovery" in manifest["checks"]:
        for name in ["recovery_over_time_distribution_56_123.pdf", "fig02_reconstructed_from_cache.pdf"]:
            with pdfplumber.open(roots / name) as pdf:
                curves.append([c for c in pdf.pages[0].curves if c["stroke"]
                               and c["stroking_color"] in recovery_colors and len(c["pts"]) > 10])
        assert len(curves[0]) == 6
        report["historical_geometry"]["fig02"] = compare_paths(*curves, [(0, 3), (3, 6)], 1e-6)
    trajectory_colors = [(0., .4471, .698), (.6, .6, .6), (.8353, .3686, 0.)]
    curves = []
    if "trajectory" in manifest["checks"]:
        for name in ["nature_enhanced_trajectory.pdf", "fig03_reconstructed_from_saved_traces.pdf"]:
            with pdfplumber.open(roots / name) as pdf:
                curves.append([c for c in pdf.pages[0].curves if c["stroking_color"] in trajectory_colors
                               and c["top"] > 200])
        starts = [i for i, c in enumerate(curves[0]) if c["stroke"]
                  and c["stroking_color"] == trajectory_colors[0]]
        assert len(starts) == 6
        starts[-1] -= 1  # The first min-max polygon precedes the mean curve.
        starts.append(len(curves[0]))
        report["historical_geometry"]["fig03"] = compare_paths(
            *curves, list(zip(starts[:-1], starts[1:])), 2e-4)
        report["historical_geometry"]["fig03"]["interpretation"] = (
            "Saved traces reproduce every historical method path, marker and min-max polygon; "
            "new Kaleido export has a common canvas padding transform and subpixel rounding.")
    report["recovery_statistics"] = manifest["checks"].get("recovery")
    report["recovery_display"] = manifest["checks"].get("recovery_display")
    report["trajectory_checks"] = manifest["checks"].get("trajectory")
    report["compiled_si_check"] = "pending approval/adoption; preview designed at exact insertion width"
    (args.report or args.output / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
