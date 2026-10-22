"""SI Fig. 4: preserve published PDF paths while separating bus/method keys."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pdfplumber
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


WIDTH, HEIGHT = 370.6102117061, 230
BOXES = [(41, 35, 178, 175), (230, 35, 364, 175)]
BUS_OLD = [(0.3882, 0.4314, 0.9804), (0.9373, 0.3333, 0.2314), (0., .8, .5882)]
COST_OLD = [(0., .4471, .698), (.3333, .3333, .3333), (.6157, .5725, .0157)]
BUS_NEW = ["#7392B2", "#C18C74", "#9A86AD"]
COST_NEW = ["#779B6E", "#C87C75", "#7D9ABA"]
INK, GRID = "#222222", "#E4E6E7"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mapped(point, old_box, new_box):
    x0, top, x1, bottom = old_box
    left, low, right, high = new_box
    x, y = point
    return left + (x - x0) / (x1 - x0) * (right - left), high - (y - top) / (bottom - top) * (high - low)


def original_objects(page):
    records = []
    for obj in page.curves + page.rects + page.lines:
        old_color = obj.get("stroking_color")
        if old_color in BUS_OLD:
            panel, color = 0, BUS_NEW[BUS_OLD.index(old_color)]
        elif old_color in COST_OLD:
            panel, color = 1, COST_NEW[COST_OLD.index(old_color)]
        else:
            continue
        # Old legend marks lie outside the axes and are excluded by the same clipping.
        assert all(command[0] in ["m", "l", "h"] for command in obj["path"])
        records.append({"panel": panel, "color": color, "object": obj})
    return records


def path_commands(c, obj, old_box, new_box):
    path = c.beginPath()
    for command in obj["path"]:
        if command[0] == "h":
            path.close()
        else:
            point = mapped(command[1], old_box, new_box)
            (path.moveTo if command[0] == "m" else path.lineTo)(*point)
    return path


def key(c, box, y, labels, colors, dashes):
    widths = [15 + pdfmetrics.stringWidth(s, "Arial", 6.8) for s in labels]
    total = sum(widths) + 5 * (len(labels) - 1)
    assert total <= box[2] - box[0] + .2, total
    x = (box[0] + box[2] - total) / 2
    for label, color, dash, width in zip(labels, colors, dashes, widths):
        c.setStrokeColor(HexColor(color))
        c.setLineWidth(1.1 if label == "RLC-FT" else .9)
        c.setDash(dash)
        c.line(x, y, x + 12, y)
        c.setDash([])
        c.setFont("Arial", 6.8)
        c.setFillColor(HexColor(INK))
        c.drawString(x + 15, y - 2.2, label)
        x += width + 5


def render(source, target, fonts):
    pdfmetrics.registerFont(TTFont("Arial", str(fonts / "arial.ttf")))
    pdfmetrics.registerFont(TTFont("Arial-Bold", str(fonts / "arialbd.ttf")))
    with pdfplumber.open(source) as pdf:
        page = pdf.pages[0]
        assert (page.width, page.height) == (2700, 1350)
        old_boxes = [[page.rects[i][k] for k in ["x0", "top", "x1", "bottom"]] for i in [1, 2]]
        records = original_objects(page)
        grids = []
        for old in old_boxes:
            values = sorted(set(round(l["top"], 3) for l in page.lines
                                if abs(l["width"] - (old[2] - old[0])) < .01
                                and abs(l["x0"] - old[0]) < .01 and l["height"] < .01
                                and old[1] < l["top"] < old[3]))
            grids.append(values)
        assert [len(g) for g in grids] == [9, 4]
    c = canvas.Canvas(str(target), pagesize=(WIDTH, HEIGHT), invariant=1, pageCompression=1)
    c.setTitle("Representative reactive-power output and objective cost")
    for panel, (old, new) in enumerate(zip(old_boxes, BOXES)):
        left, bottom, right, top = new
        c.setStrokeColor(HexColor(GRID))
        c.setLineWidth(.45)
        for y in grids[panel]:
            yy = mapped((old[0], y), old, new)[1]
            c.line(left, yy, right, yy)
        c.saveState()
        clip = c.beginPath()
        clip.rect(left, bottom, right - left, top - bottom)
        c.clipPath(clip, stroke=0, fill=0)
        for i, record in enumerate(records):
            if record["panel"] != panel:
                continue
            obj = record["object"]
            c.setStrokeColor(HexColor(record["color"]))
            c.setFillColor(HexColor(record["color"]))
            c.setLineWidth(1.2 if obj["stroke"] else .8)
            c.setDash([])
            c.setLineCap(0)
            c.setLineJoin(1)
            c.addLiteral(f"/Figure <</MCID {i}>> BDC")
            c.drawPath(path_commands(c, obj, old, new), stroke=int(obj["stroke"]), fill=int(obj["fill"]))
            c.addLiteral("EMC")
        c.restoreState()
        c.setStrokeColor(HexColor(INK))
        c.setLineWidth(.65)
        c.line(left, bottom, right, bottom)
        c.line(left, bottom, left, top)
        c.setFillColor(HexColor(INK))
        c.setFont("Arial", 6.5)
        yticks = np.arange(1.8, .19, -.2) if panel == 0 else [25, 20, 15, 10]
        for y, tick in zip(grids[panel], yticks):
            yy = mapped((old[0], y), old, new)[1]
            c.line(left - 2.5, yy, left, yy)
            text = f"{tick:.1f}" if panel == 0 else str(tick)
            c.drawRightString(left - 5, yy - 2.2, text)
        xmax = 15 if panel == 0 else 20
        xticks = range(0, 15, 2) if panel == 0 else [0, 5, 10, 15, 20]
        for tick in xticks:
            xx = left + tick / xmax * (right - left)
            c.line(xx, bottom, xx, bottom - 2.5)
            c.drawCentredString(xx, bottom - 11, str(tick))
        c.setFont("Arial", 7.3)
        c.drawCentredString((left + right) / 2, 7, "Control steps")
        c.saveState()
        c.translate(left - 26, (bottom + top) / 2)
        c.rotate(90)
        c.drawCentredString(0, 0, "Reactive power (MVAr)" if panel == 0 else "Objective cost")
        c.restoreState()
        c.setFont("Arial", 7.8)
        c.drawCentredString((left + right) / 2, 211, "Reactive power output" if panel == 0 else "Objective cost")
        c.setFont("Arial-Bold", 9)
        c.drawString(left - 9, 211, "a" if panel == 0 else "b")
    key(c, BOXES[0], 193, ["Bus 30", "Bus 45", "Bus 53"], BUS_NEW, [[], [], []])
    key(c, BOXES[1], 193, ["Linear", "Safe-DDPG", "RLC-FT"],
        [COST_NEW[1], COST_NEW[2], COST_NEW[0]], [[3, 1.8], [3, 1.2, .9, 1.2], []])
    c.showPage()
    c.save()
    return records, old_boxes


def verify(target, records, old_boxes):
    with pdfplumber.open(target) as pdf:
        page = pdf.pages[0]
        objects = page.curves + page.rects + page.lines
        actual = {o["mcid"]: o for o in objects if isinstance(o.get("mcid"), int)}
        assert len(actual) == len(records), (len(actual), len(records))
        errors = []
        for i, record in enumerate(records):
            old, new = record["object"], actual[i]
            assert [p[0] for p in old["path"]] == [p[0] for p in new["path"]]
            assert (old["stroke"], old["fill"]) == (new["stroke"], new["fill"])
            for a, b in zip(old["path"], new["path"]):
                if a[0] == "h":
                    continue
                x, y = mapped(a[1], old_boxes[record["panel"]], BOXES[record["panel"]])
                error = float(np.max(np.abs(np.asarray([x, HEIGHT - y]) - b[1])))
                errors.append(error)
                assert error < .001, error
        chars = [ch for ch in page.chars if ch["text"].strip()]
        assert min(ch["size"] for ch in chars if ch["upright"]) >= 6.49
        assert all(ch["x0"] >= 0 and ch["x1"] <= WIDTH and ch["top"] >= 0
                   and ch["bottom"] <= HEIGHT for ch in chars)
        return {"preserved_vector_objects": len(records),
                "preserved_vertices": sum(len(r["object"]["pts"]) for r in records),
                "same_path_commands": True, "maximum_transform_error_pt": max(errors),
                "same_axes_clipping": True, "raw_simulation_arrays_recovered": False,
                "data_recomputed": False, "minimum_tick_legend_font_pt": 6.5}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font-dir", type=Path, default=Path("C:/Windows/Fonts"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    originals = args.output / "originals"
    originals.mkdir(exist_ok=True)
    backup = originals / args.source.name
    if backup.exists():
        assert sha(backup) == sha(args.source)
    else:
        shutil.copy2(args.source, backup)
    target = args.output / "fig04_output_cost_preview.pdf"
    records, boxes = render(backup, target, args.font_dir)
    checks = verify(target, records, boxes)
    notebook = args.repo / "experiments/validated/supplementary_evaluation/policy/test_policy_output.ipynb"
    book = json.loads(notebook.read_text(encoding="utf-8"))
    cell = book["cells"][3]
    record = {"status": "preview_only_not_adopted", "original_pdf_sha256": sha(args.source),
              "source_code_sha256": sha(__file__), "output_sha256": sha(target),
              "dimensions_pt": [WIDTH, HEIGHT], "source_plot_boxes": boxes, "target_plot_boxes": BOXES,
              "verification": checks,
              "code_family": {"notebook": str(notebook), "sha256": sha(notebook),
                              "plot_cell_one_based": 4, "simulation_cell_one_based": 3,
                              "plot_cell_sha256": hashlib.sha256("".join(cell["source"]).encode()).hexdigest(),
                              "historical_commit_checked": "3c4d36f",
                              "saved_traces_match_published_pdf": False},
              "provenance": "Published PDF vector paths, not reconstructed raw simulation arrays. Current and earliest tracked notebook store different negative-Q/cost traces.",
              "caption_on_adoption": "Panel a colors identify buses; solid/dash-dot/dashed lines identify RLC-FT/Safe-DDPG/Linear. Panel b uses green/blue/muted-red methods with the same line patterns."}
    assert sha(backup) == sha(args.source)
    (args.output / "fig04_manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(target), "verification": checks}, indent=2))


if __name__ == "__main__":
    main()
