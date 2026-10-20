"""Restyle Fig. 9 using its exact original raster curves, not model replay."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

WIDTH, HEIGHT = 370.6102117061, 350.
LEFT, RIGHT, GAP = 32., WIDTH - 8., 16.
PANEL_WIDTH = (RIGHT - LEFT - 4*GAP) / 5
PANEL_HEIGHT = 68.
ROW_TOPS = (278., 187., 96.)
BUS_IDS = (18, 21, 30, 45, 53)
SI_LABELS = ("0.01", "0.1", "0.5")
INK, GREEN, RED, ZERO = "#242A2D", "#779B6E", "#C87C75", "#DDE3DF"
SOURCE_COLORS = ((255, 127, 14), (31, 119, 180))
STROKE_FILTER_SIZE = 3

# Labels read from the verified original PNGs, ordered top-to-bottom.
Y_TICKS = (
    ((2, 1, 0, -1, -2), (1.5, 1, .5, 0, -.5, -1, -1.5),
     (1, .5, 0, -.5, -1), (1, .5, 0, -.5, -1), (.8, .6, .4, .2, 0, -.2, -.4, -.6)),
    ((1.5, 1, .5, 0, -.5, -1), (1.5, 1, .5, 0, -.5, -1, -1.5),
     (1.5, 1, .5, 0, -.5, -1, -1.5), (1.5, 1, .5, 0, -.5, -1, -1.5),
     (.8, .6, .4, .2, 0, -.2, -.4, -.6)),
    ((1.5, 1, .5, 0, -.5, -1), (3, 2, 1, 0, -1, -2, -3),
     (.8, .6, .4, .2, 0, -.2, -.4, -.6), (.8, .6, .4, .2, 0, -.2, -.4, -.6),
     (.8, .6, .4, .2, 0, -.2, -.4, -.6)),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def axis_edges(array):
    xs = np.flatnonzero((array.max(axis=2) < 100).sum(axis=0) > len(array)*.55)
    groups = np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1)
    return [int(np.median(group)) for group in groups if len(group)]


def hex_rgb(value):
    return np.asarray([int(value[i:i+2], 16) for i in (1, 3, 5)], dtype=float)


def alpha_for_color(array, color):
    color = np.asarray(color, dtype=float)
    pixels = array.astype(float)
    direction = color - color.mean()
    alpha = ((pixels - pixels.mean(axis=2, keepdims=True)) @ direction) / np.dot(direction, direction)
    neutral = pixels.mean(axis=2) - alpha*color.mean()
    reconstructed = neutral[..., None] + alpha[..., None]*color
    error = np.max(np.abs(reconstructed - array), axis=2)
    return np.where((error < 1.25) & (alpha > .008) & (alpha < 1.03), np.clip(alpha, 0, 1), 0)


def original_legend_frame(raw):
    pixels = raw.astype(float)
    gray = (pixels.max(axis=2)-pixels.min(axis=2) < 2) & (pixels.mean(axis=2) > 200) & (pixels.mean(axis=2) < 230)
    gray[:int(len(raw)*.55)] = False
    gray[:, :3] = False
    gray[:, -3:] = False
    gray[-3:] = False
    rows = np.flatnonzero(gray.sum(axis=1) > raw.shape[1]*.45)
    assert len(rows) == 2, rows
    top, bottom = int(rows.min()), int(rows.max())
    cols = np.flatnonzero(gray[top:bottom+1].sum(axis=0) > (bottom-top)*.5)
    assert len(cols) == 2, cols
    return int(cols.min()), top, int(cols.max()), bottom


def tick_rows(array, left):
    black = array.max(axis=2) < 100
    ys = np.flatnonzero(black[:, left-5:left-1].sum(axis=1) >= 3)
    groups = np.split(ys, np.flatnonzero(np.diff(ys) > 1) + 1)
    return [int(np.median(group)) for group in groups if len(group)]


def panel_layers(array, row, col, output):
    edges = axis_edges(array)
    left, right = edges[2*col:2*col+2]
    black = array.max(axis=2) < 100
    vertical = np.flatnonzero(black[:, left])
    top, bottom = int(vertical.min()), int(vertical.max())
    ticks = tick_rows(array, left)
    labels = Y_TICKS[row][col]
    assert len(ticks) == len(labels), (row, col, ticks, labels)
    zero = ticks[labels.index(0)]
    raw = array[top:bottom+1, left:right+1]
    legend_frame = original_legend_frame(raw)
    layers = []
    for method, source_color, target_color in zip(("RLC-FT", "Linear"), SOURCE_COLORS, (GREEN, RED)):
        alpha = alpha_for_color(raw, source_color)
        yy, xx = np.indices(alpha.shape)
        # In all original panels, data at the left are nonnegative; the only
        # colored pixels below zero there are the lower-left legend samples.
        legend_region = (xx < .38*(right-left)) & (yy > zero-top+4)
        alpha[legend_region] = 0
        support = alpha > 0
        ll, lt, lr, lb = legend_frame
        old_overlay = (xx >= ll-1) & (xx <= lr+1) & (yy >= lt-1) & (yy <= lb+1)
        # Undo the legacy legend's 0.8 white/gray overlay only at existing
        # curve pixels; no missing points or paths are inferred.
        alpha[old_overlay] = np.minimum(1, 5*alpha[old_overlay])
        alpha_u8 = np.round(alpha*255).astype(np.uint8)
        np.testing.assert_array_equal(support, alpha_u8 > 0)
        assert np.count_nonzero(alpha_u8) > 60, (row, col, method)
        base_alpha = alpha_u8.copy()
        # Expand the rendered stroke symmetrically by one source pixel,
        # retaining all original curve positions instead of refitting data.
        alpha_u8 = np.asarray(Image.fromarray(base_alpha).filter(ImageFilter.MaxFilter(STROKE_FILTER_SIZE)))
        assert np.all(alpha_u8 >= base_alpha)
        rgba = np.zeros((*alpha.shape, 4), dtype=np.uint8)
        rgba[..., :3] = hex_rgb(target_color).astype(np.uint8)
        rgba[..., 3] = alpha_u8
        image = Image.fromarray(rgba)
        file = output / "curve_layers" / f"row{row+1}_bus{BUS_IDS[col]}_{method.lower()}.png"
        file.parent.mkdir(parents=True, exist_ok=True)
        image.save(file)
        layers.append({"method": method, "file": str(file), "sha256": digest(file),
                       "alpha_sha256": hashlib.sha256(alpha_u8.tobytes()).hexdigest(),
                       "source_curve_support_sha256": hashlib.sha256(support.tobytes()).hexdigest(),
                       "base_alpha_sha256": hashlib.sha256(base_alpha.tobytes()).hexdigest(),
                       "stroke_filter_size": STROKE_FILTER_SIZE,
                       "visible_source_pixels": int(np.count_nonzero(alpha_u8))})
    selected = [value for value in labels if value in
                ((2, 0, -2) if row == 0 and col == 0 else
                 (3, 0, -3) if row == 2 and col == 1 else
                 (.6, 0, -.6) if col == 4 or (row == 2 and col >= 2) else (1, 0, -1))]
    return {"source_pixel_frame": [left, top, right, bottom], "legacy_legend_frame": list(legend_frame),
            "ticks": [{"value": value, "source_y": ticks[labels.index(value)]} for value in selected],
            "zero_source_y": zero, "layers": layers}


def text(c, x, y, value, size=6.5, font="Arial", align="left"):
    c.setFillColor(HexColor(INK))
    c.setFont(font, size)
    width = pdfmetrics.stringWidth(value, font, size)
    actual_left = x - width/2 if align == "center" else x - width if align == "right" else x
    assert actual_left >= 0 and actual_left + width <= WIDTH, (value, actual_left, width)
    assert y >= 0 and y + size <= HEIGHT, (value, y)
    {"left": c.drawString, "right": c.drawRightString, "center": c.drawCentredString}[align](x, y, value)


def key(c, x, y, value, color, dashed=False):
    c.saveState()
    c.setStrokeColor(HexColor(color))
    c.setLineWidth(1.1)
    if dashed:
        c.setDash([4, 1.8, .9, 1.8])
    c.line(x, y, x+18, y)
    c.restoreState()
    text(c, x+23, y-2.2, value, 6.8)


def draw_panel(c, panel, row, col):
    left = LEFT + col*(PANEL_WIDTH + GAP)
    top = ROW_TOPS[row]
    bottom = top - PANEL_HEIGHT
    source_left, source_top, source_right, source_bottom = panel["source_pixel_frame"]
    source_w, source_h = source_right - source_left, source_bottom - source_top
    y_at = lambda source_y: bottom + (source_bottom - source_y)*PANEL_HEIGHT/source_h
    c.setStrokeColor(HexColor(ZERO))
    c.setLineWidth(.45)
    c.line(left, y_at(panel["zero_source_y"]), left+PANEL_WIDTH, y_at(panel["zero_source_y"]))
    c.saveState()
    path = c.beginPath()
    path.rect(left, bottom, PANEL_WIDTH, PANEL_HEIGHT)
    c.clipPath(path, stroke=0, fill=0)
    # Pixel centers, including the source-axis boundary, retain their exact
    # normalized coordinates. Text is redrawn separately as vector glyphs.
    dx, dy = PANEL_WIDTH/source_w, PANEL_HEIGHT/source_h
    for layer in reversed(panel["layers"]):
        c.drawImage(ImageReader(layer["file"]), left-dx/2, bottom-dy/2,
                    width=PANEL_WIDTH+dx, height=PANEL_HEIGHT+dy, mask="auto")
    c.restoreState()
    c.setStrokeColor(HexColor(INK))
    c.setLineWidth(.65)
    c.line(left, bottom, left+PANEL_WIDTH, bottom)
    c.line(left, bottom, left, top)
    for tick in panel["ticks"]:
        y = y_at(tick["source_y"])
        c.line(left-2.5, y, left, y)
        label = f"{tick['value']:g}".replace("-", "\u2212")
        text(c, left-4.5, y-2.2, label, align="right")
    # Legacy Train.py uses 12*voltage and Matplotlib's 5% x margin.
    for value in (10, 12, 14):
        x = left + PANEL_WIDTH*(value - 9.366)/(14.514 - 9.366)
        c.line(x, bottom, x, bottom-2.5)
        if row == 2:
            text(c, x, bottom-11, str(value), align="center")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-record", type=Path, required=True)
    parser.add_argument("--original-pdf", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stem", default="kappa")
    parser.add_argument("--font-dir", type=Path, default=Path("C:/Windows/Fonts"))
    args = parser.parse_args()
    record = json.loads(args.source_record.read_text(encoding="utf-8"))
    source_pdf = args.original_pdf or Path(record["paper_pdf"])
    assert digest(source_pdf) == record["paper_pdf_sha256"]
    args.output.mkdir(parents=True, exist_ok=True)
    originals = args.output / "originals"
    originals.mkdir(exist_ok=True)
    for path in [source_pdf, *[Path(row["source"]["path"]) for row in record["rows"]]]:
        target = originals / path.name
        if target.exists():
            assert digest(target) == digest(path), target
        else:
            shutil.copy2(path, target)
    for name, filename in [("Arial", "arial.ttf"), ("Arial-Bold", "arialbd.ttf"), ("Arial-Italic", "ariali.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(args.font_dir / filename)))
    panels = []
    for row, item in enumerate(record["rows"]):
        path = Path(item["source"]["path"])
        assert digest(path) == item["source_sha256"]
        array = np.asarray(Image.open(path).convert("RGB"))
        panels.append([panel_layers(array, row, col, args.output) for col in range(5)])
    pdf = args.output / f"{args.stem}.pdf"
    c = canvas.Canvas(str(pdf), pagesize=(WIDTH, HEIGHT), invariant=1)
    c.setTitle("Stability slope bound and learned control policies")
    key_width = 23 + pdfmetrics.stringWidth("RLC-FT", "Arial", 6.8) + 25 + 23 + pdfmetrics.stringWidth("Linear", "Arial", 6.8)
    x = (WIDTH - key_width)/2
    key(c, x, 337, "RLC-FT", GREEN)
    key(c, x + 23 + pdfmetrics.stringWidth("RLC-FT", "Arial", 6.8) + 25, 337, "Linear", RED, True)
    for col, bus in enumerate(BUS_IDS):
        text(c, LEFT + col*(PANEL_WIDTH+GAP) + PANEL_WIDTH/2, 312, f"Bus {bus}", 7.8, "Arial-Bold", "center")
    for row, label in enumerate(SI_LABELS):
        text(c, LEFT, ROW_TOPS[row]+11, "abc"[row], 9, "Arial-Bold")
        text(c, (LEFT+RIGHT)/2, ROW_TOPS[row]+11, "\u03ba = " + label, 8.2, "Arial-Italic", "center")
        for col in range(5):
            draw_panel(c, panels[row][col], row, col)
    text(c, (LEFT+RIGHT)/2, 3.5, "Voltage (kV)", 7.3, align="center")
    c.saveState()
    c.translate(8, (ROW_TOPS[0] + ROW_TOPS[2] - PANEL_HEIGHT)/2)
    c.rotate(90)
    c.setFont("Arial", 7.3)
    c.setFillColor(HexColor(INK))
    c.drawCentredString(0, 0, "Q (MVAr)")
    c.restoreState()
    c.save()
    from pypdf import PdfReader
    import pdfplumber

    originals = [np.asarray(Image.open(layer["file"]).convert("RGBA"))
                 for row in panels for panel in row for layer in panel["layers"]]
    pool = originals.copy()
    embedded = list(PdfReader(pdf).pages[0].images)
    for item in embedded:
        array = np.asarray(item.image.convert("RGBA"))
        matches = [i for i, original in enumerate(pool)
                   if array.shape == original.shape and np.array_equal(array, original)]
        assert matches, item.name
        pool.pop(matches[0])
    assert len(embedded) == 30 and not pool
    with pdfplumber.open(pdf) as document:
        page = document.pages[0]
        assert abs(page.width - WIDTH) < .001 and page.height == HEIGHT
        upright_sizes = sorted({round(char["size"], 2) for char in page.chars if char["upright"]})
        assert min(upright_sizes) >= 6.5
        assert page.extract_text().count("\u03ba =") == 3
    manifest = {"status": "generated candidate; adoption recorded separately", "output_pdf": str(pdf), "output_pdf_sha256": digest(pdf),
                "page_width_bp": WIDTH, "page_height_bp": HEIGHT, "fonts": {"ticks": 6.5, "legend": 6.8, "axis_labels": 7.3, "bus_headers": 7.8, "kappa_headers": 8.2},
                "display_kappa_labels": list(SI_LABELS),
                "stroke_style": {"symmetric_source_pixel_expansion": 1,
                                 "filter_size": STROKE_FILTER_SIZE,
                                 "additional_full_stroke_width_bp": [2*PANEL_WIDTH/201, 2*PANEL_HEIGHT/231],
                                 "source_curve_positions_retained": True},
                "label_decision": "User requested retention of current SI labels on 2026-10-13. Historical clamp evidence is retained in model_diagnostic.json and is not overwritten or treated as disproved.",
                "source_record": str(args.source_record), "source_record_sha256": digest(args.source_record),
                "source_pdf_sha256": digest(source_pdf), "panels": panels,
                "verification": {"pixel_identical_exported_curve_layers": len(embedded),
                                 "original_curve_pixel_support_preserved": True,
                                 "upright_font_sizes_pt": upright_sizes,
                                 "display_width_matches_si_linewidth": True},
                "curve_handling": "Original raster curve positions retained under affine layout transformation; original legend samples excluded, controller colors changed. Legacy legend-overlay fading restored at existing colored pixels; stroke footprint expanded symmetrically by one source pixel for readability. No inferred missing data points, digitization, interpolation, smoothing, scale change or model evaluation used for the figure. Axes/text are vector; curves remain raster.",
                "manuscript_changed": False}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"pdf": str(pdf), "panels": 15, "source_images": 3, "manuscript_changed": False}))


if __name__ == "__main__":
    main()
