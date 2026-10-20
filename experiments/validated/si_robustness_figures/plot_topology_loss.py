"""Restyle SI Fig. 11 while preserving its original PDF curve/band geometry."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pdfplumber
from reportlab.lib.colors import HexColor, Color
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


WIDTH, HEIGHT = 370.6102117061, 208.0
BOX = (40.0, 34.0, 359.0, 162.0)  # left, bottom, right, top
COLORS = ("#779B6E", "#C87C75")
INK, GRID, LIMIT = "#242A2D", "#E3E7E5", "#6F7773"
OLD_COLORS = ((.1059, .6196, .4667), (.851, .3725, .0078))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_geometry(source):
    with pdfplumber.open(source) as pdf:
        page = pdf.pages[0]
        assert len(pdf.pages) == 1 and (page.width, page.height) == (2700, 1350)
        curves = [c for c in page.curves if len(c["pts"]) > 10]
        assert [len(c["pts"]) for c in curves] == [32, 42, 13, 22]
        for i, curve in enumerate(curves):
            np.testing.assert_allclose(curve["stroking_color"], OLD_COLORS[i % 2])
            assert curve["fill"] == (i < 2)
            assert curve["stroke"] == (i >= 2)
            assert all(item[0] in ("m", "l", "h") for item in curve["path"])
        frame = page.rects[1]
        old_box = [frame[key] for key in ("x0", "top", "x1", "bottom")]
        np.testing.assert_allclose(old_box, [148.5, 150, 2580, 1230], atol=.001)
        ticks = sorted([l["top"] for l in page.lines if l["stroke"]
                        and abs(l["width"] - frame["width"]) < .001
                        and l["height"] < .001], reverse=True)
        assert len(ticks) == 10
        # The original dashed threshold was exported as filled dash rectangles.
        limit_y = (page.curves[4]["top"] + page.curves[4]["bottom"]) / 2
        return curves, old_box, ticks, limit_y


def transform(point, old_box):
    x0, top, x1, bottom = old_box
    left, low, right, high = BOX
    x, y = point
    return (left + (x-x0) / (x1-x0) * (right-left),
            high - (y-top) / (bottom-top) * (high-low))


def draw_original_path(c, original, old_box, stroke, fill):
    path = c.beginPath()
    for command in original["path"]:
        if command[0] == "h":
            path.close()
        else:
            point = transform(command[1], old_box)
            (path.moveTo if command[0] == "m" else path.lineTo)(*point)
    c.drawPath(path, stroke=stroke, fill=fill)


def legend_item(c, x, y, label, color, dashed=False, band=False, limit=False):
    if band:
        c.saveState()
        c.setFillColor(HexColor(color))
        c.setFillAlpha(.16)
        c.rect(x, y-2.5, 19, 5, fill=1, stroke=0)
        c.restoreState()
    c.setStrokeColor(HexColor(color))
    c.setLineWidth(.8 if limit else 1.3)
    c.setDash([3, 1.8] if dashed else [])
    c.line(x, y, x+19, y)
    c.setDash([])
    c.setFillColor(HexColor(INK))
    c.setFont("Arial", 6.8)
    c.drawString(x+24, y-2.25, label)
    return 24 + pdfmetrics.stringWidth(label, "Arial", 6.8)


def render(source, target, font_dir):
    pdfmetrics.registerFont(TTFont("Arial", str(font_dir / "arial.ttf")))
    pdfmetrics.registerFont(TTFont("Arial-Bold", str(font_dir / "arialbd.ttf")))
    curves, old_box, ticks, limit_y = load_geometry(source)
    c = canvas.Canvas(str(target), pagesize=(WIDTH, HEIGHT), invariant=1, pageCompression=1)
    c.setTitle("Voltage recovery on Bus 30 under communication loss")
    c.setAuthor("RLC-FT")
    left, bottom, right, top = BOX
    c.setFillColor(Color(1, 1, 1))
    c.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)

    c.setFillColor(HexColor(INK))
    c.setFont("Arial-Bold", 8.0)
    c.drawString(left, 195, "Voltage recovery on Bus 30")
    x = left
    x += legend_item(c, x, 180, "Perfect information (0% loss)", COLORS[0], band=True) + 15
    x += legend_item(c, x, 180, "50% packet loss", COLORS[1], dashed=True, band=True) + 15
    assert x + legend_item(c, x, 180, "Safe lower limit", LIMIT, dashed=True, limit=True) < WIDTH-6

    c.setStrokeColor(HexColor(GRID))
    c.setLineWidth(.45)
    for tick in ticks:
        y = transform((old_box[0], tick), old_box)[1]
        c.line(left, y, right, y)

    # One affine layout transform for every original mean and range boundary.
    # Preserve the original axes clipping, including paths extending beyond step 18.
    c.saveState()
    clip = c.beginPath()
    clip.rect(left, bottom, right-left, top-bottom)
    c.clipPath(clip, stroke=0, fill=0)
    for i in range(2):
        c.setFillColor(HexColor(COLORS[i]))
        c.setFillAlpha(.16)
        draw_original_path(c, curves[i], old_box, stroke=0, fill=1)
    c.setFillAlpha(1)
    for i in range(2):
        c.setStrokeColor(HexColor(COLORS[i]))
        c.setLineWidth(1.35)
        c.setLineJoin(1)
        c.setLineCap(0)
        c.setDash([] if i == 0 else [3, 1.8])
        draw_original_path(c, curves[i+2], old_box, stroke=1, fill=0)
    c.setStrokeColor(HexColor(LIMIT))
    c.setLineWidth(.75)
    c.setDash([3, 2])
    y = transform((old_box[0], limit_y), old_box)[1]
    c.line(left, y, right, y)
    c.restoreState()

    c.setStrokeColor(HexColor(INK))
    c.setFillColor(HexColor(INK))
    c.setLineWidth(.65)
    c.line(left, bottom, right, bottom)
    c.line(left, bottom, left, top)
    c.setFont("Arial", 6.5)
    for value, tick in zip(np.arange(80, 100, 2) / 100, ticks):
        y = transform((old_box[0], tick), old_box)[1]
        c.line(left-2.5, y, left, y)
        c.drawRightString(left-5, y-2.2, f"{value:.2f}")
    for step in range(0, 19, 2):
        x = left + step/18 * (right-left)
        c.line(x, bottom, x, bottom-2.5)
        c.drawCentredString(x, bottom-11, str(step))
    c.setFont("Arial", 7.3)
    c.drawCentredString((left+right)/2, 7, "Control step")
    c.saveState()
    c.translate(12, (bottom+top)/2)
    c.rotate(90)
    c.drawCentredString(0, 0, "Bus voltage (p.u.)")
    c.restoreState()
    c.showPage()
    c.save()
    return curves, old_box


def verify(target, originals, old_box):
    with pdfplumber.open(target) as pdf:
        page = pdf.pages[0]
        curves = [p for p in page.curves if len(p["pts"]) > 10]
        assert len(curves) == 4
        errors = []
        for old, new in zip(originals, curves):
            assert [v[0] for v in old["path"]] == [v[0] for v in new["path"]]
            expected = np.asarray([transform(p, old_box) for p in old["pts"]])
            actual = np.asarray([(x, HEIGHT-y) for x, y in new["pts"]])
            assert expected.shape == actual.shape
            error = float(np.max(np.abs(expected-actual)))
            assert error < .001, error
            errors.append(error)
        visible = [char for char in page.chars if char["text"].strip()]
        assert min(char["size"] for char in visible if char["upright"]) >= 6.49
        assert all(char["x0"] >= 0 and char["x1"] <= WIDTH
                   and char["top"] >= 0 and char["bottom"] <= HEIGHT for char in visible)
        return {"preserved_path_vertices": [len(p["pts"]) for p in curves],
                "maximum_affine_error_pdf_pt": max(errors),
                "same_path_commands": True,
                "data_recomputed": False,
                "same_x_extent_steps": [0, 18],
                "same_y_extent": True,
                "minimum_upright_font_pt": min(char["size"] for char in visible if char["upright"]),
                "text_within_page": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font-dir", type=Path, default=Path("C:/Windows/Fonts"))
    args = parser.parse_args()
    source_hash = digest(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    backups = args.output / "originals"
    backups.mkdir(exist_ok=True)
    backup = backups / args.source.name
    if backup.exists():
        assert digest(backup) == source_hash
    else:
        shutil.copy2(args.source, backup)
    target = args.output / "fig11_topology_loss_preview.pdf"
    originals, old_box = render(backup, target, args.font_dir)
    checks = verify(target, originals, old_box)
    assert digest(args.source) == source_hash == digest(backup)
    manifest = {"status": "preview_only_not_adopted", "source_pdf": str(args.source.resolve()),
                "source_pdf_sha256": source_hash, "backup": str(backup.resolve()),
                "source_code": str(Path(__file__).resolve()), "source_code_sha256": digest(__file__),
                "output": str(target.resolve()), "output_sha256": digest(target),
                "dimensions_pdf_pt": [WIDTH, HEIGHT], "source_plot_box": old_box,
                "target_plot_box": BOX, "verification": checks,
                "provenance": "Original PDF vector paths, not recovered raw simulation arrays.",
                "style_changes": ["Arial at manuscript width", "muted condition colors",
                                  "external legend", "solid/dashed condition keys",
                                  "lighter min-max bands", "simplified title and axes"]}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
