"""Check SI Fig. 10/12 candidates, preserved originals and optional adoption."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pdfplumber


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path,
                        help="Separate report path; preserves the original preview verification.")
    parser.add_argument("--require-adopted", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
    asset_stems = {
        "56bus_admittance_stress_summary_100.pdf": "fig10_admittance_stress_preview",
        "imperfect_info_zoom_loss50_delay100.pdf": "fig12_imperfect_information_preview",
    }
    paper_states = {}
    for path, expected in manifest["original_paper_assets"].items():
        name = Path(path).name
        assert digest(args.output / "originals" / name) == expected
        candidate = args.output / f"{asset_stems[name]}.pdf"
        assert digest(candidate) == manifest["preview_outputs"][candidate.name]
        current = digest(path)
        paper_states[name] = ("adopted" if current == digest(candidate)
                              else "original" if current == expected else "another_revision")
        if args.require_adopted:
            assert paper_states[name] == "adopted"
    report = {"original_backups_unchanged": True, "paper_assets": paper_states, "typography": {}}
    for stem in ["fig10_admittance_stress_preview", "fig12_imperfect_information_preview"]:
        with pdfplumber.open(args.output / f"{stem}.pdf") as pdf:
            page = pdf.pages[0]
            np.testing.assert_allclose(page.width, 370.6102117061, atol=1e-5)
            visible = [c for c in page.chars if c["text"].strip()]
            upright = [c for c in visible if c["upright"]]
            minimum = min(c["size"] for c in upright)
            assert minimum >= 6.0
            outside = [c["text"] for c in visible if c["x0"] < -.2 or c["x1"] > page.width+.2
                       or c["top"] < -.2 or c["bottom"] > page.height+.2]
            assert not outside, (stem, outside)
            report["typography"][stem] = {"width_pt": float(page.width),
                                          "minimum_upright_font_pt": float(minimum),
                                          "text_within_page": True,
                                          "fonts": sorted({c["fontname"] for c in visible})}
    paths = [args.output / "originals/imperfect_info_zoom_loss50_delay100.pdf",
             args.output / "originals/fig12_reconstructed_from_traces.pdf"]
    curves = []
    for path in paths:
        with pdfplumber.open(path) as pdf:
            curves.append([c for c in pdf.pages[0].curves if len(c["pts"]) > 10])
    assert len(curves[0]) == len(curves[1]) == 104
    for old, new in zip(*curves):
        assert len(old["pts"]) == len(new["pts"])
        assert old["stroking_color"] == new["stroking_color"]
    residuals = []
    # One layout transform per axes, not a separate transform per voltage trace.
    for start, stop in [(0, 5), (5, 10), (10, 104)]:
        a = np.concatenate([np.asarray(c["pts"]) for c in curves[0][start:stop]])
        b = np.concatenate([np.asarray(c["pts"]) for c in curves[1][start:stop]])
        design = np.c_[a, np.ones(len(a))]
        transform = np.linalg.lstsq(design, b, rcond=None)[0]
        residual = float(np.max(np.abs(design @ transform - b)))
        assert residual < 1e-5, (start, stop, residual)
        residuals.append(residual)
    report["fig12_original_vs_reconstructed"] = {
        "matching_path_count": 104,
        "matching_vertices": sum(len(c["pts"]) for c in curves[0]),
        "per_axes_max_affine_residual_pt": residuals,
        "interpretation": "Stored voltage traces reproduce the adopted PDF's curve geometry; export layout differs."}
    (args.report or args.output / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
