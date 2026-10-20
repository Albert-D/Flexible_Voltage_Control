"""Check the Fig. 1 statistics and Fig. 4 preserved paths after PDF export."""

import argparse
import json
from pathlib import Path

import numpy as np
import pdfplumber

from plot_output_cost import original_objects, sha, verify


COLORS = [tuple(int(value[i:i + 2], 16) / 255 for i in (1, 3, 5))
          for value in ("#D2877E", "#9FB4CD", "#98B486")]
METHODS = ["Linear", "Safe-DDPG", "RLC-FT"]


def performance_geometry(path):
    with pdfplumber.open(path) as pdf:
        page = pdf.pages[0]
        bars = [r for r in page.rects if r["fill"]
                and isinstance(r["non_stroking_color"], (list, tuple))
                and any(np.allclose(r["non_stroking_color"], c, rtol=0, atol=1e-5)
                        for c in COLORS)][:12]
        assert len(bars) == 12
        medians = [b["height"] / bars[i // 3 * 3]["height"]
                   for i, b in enumerate(bars)]
        stems = [o for o in page.lines + page.curves if o["stroke"]
                 and o["width"] < .0001 and o["height"] > .1
                 and np.allclose(o["stroking_color"], 79 / 255, rtol=0, atol=1e-6)]
        assert len(stems) == 8
        iqrs = []
        for index, stem in enumerate(stems):
            base = bars[index // 2 * 3]
            iqrs.append([(stem[k] - base["y0"]) / base["height"]
                         for k in ("y0", "y1")])
        return np.asarray(medians), np.asarray(iqrs)


def check_typography(path):
    with pdfplumber.open(path) as pdf:
        page = pdf.pages[0]
        chars = [ch for ch in page.chars if ch["text"].strip()]
        smallest = min(ch["size"] for ch in chars if ch["upright"])
        assert smallest >= 6.49
        assert all(ch["x0"] >= 0 and ch["x1"] <= page.width
                   and ch["top"] >= 0 and ch["bottom"] <= page.height for ch in chars)
        return {"minimum_upright_font_pt": smallest, "all_text_within_page": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--require-adopted", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root = args.root
    m1 = json.loads((root / "fig01/fig01_manifest.json").read_text())
    old1 = root / "fig01/originals/performance_normalized_reduction_vs_linear_123bus.pdf"
    new1 = root / "fig01/fig01_123bus_performance_preview.pdf"
    original, preview = performance_geometry(old1), performance_geometry(new1)
    stats = list(m1["normalized_stats"].values())
    expected = (np.asarray([row[m]["median"] for row in stats for m in METHODS]),
                np.asarray([[row[m][q] for q in ("q25", "q75")]
                            for row in stats for m in METHODS[1:]]))
    for old, new, reference in zip(original, preview, expected):
        np.testing.assert_allclose(old, new, rtol=0, atol=1e-7)
        np.testing.assert_allclose(new, reference, rtol=0, atol=1e-7)
    assert sha(old1) == m1["original_pdf_sha256"]
    assert sha(new1) == m1["output_sha256"]
    assert sha(args.paper / "fig" / old1.name) == sha(new1 if args.require_adopted else old1)
    for path, digest in m1["input_sha256"].items():
        assert sha(path) == digest

    m4 = json.loads((root / "fig04/fig04_manifest.json").read_text())
    old4 = root / "fig04/originals/combined_plots_nature.pdf"
    new4 = root / "fig04/fig04_output_cost_preview.pdf"
    assert sha(old4) == m4["original_pdf_sha256"]
    assert sha(new4) == m4["output_sha256"]
    assert sha(args.paper / "fig" / old4.name) == sha(new4 if args.require_adopted else old4)
    with pdfplumber.open(old4) as pdf:
        checks4 = verify(new4, original_objects(pdf.pages[0]), m4["source_plot_boxes"])
    report = {"status": "adopted_verified" if args.require_adopted else "preview_verified_not_adopted",
              "paper_assets_match": "approved_previews" if args.require_adopted else "historical_originals",
              "original_backups_unchanged": True,
              "fig01": {"all_12_bar_ratios_match": True,
                        "all_8_iqr_intervals_match": True,
                        "cache_hashes_unchanged": True, **check_typography(new1)},
              "fig04": {**checks4, **check_typography(new4)}}
    report_path = args.report if args.report else root / ("adopted_verification.json" if args.require_adopted else "verification.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
