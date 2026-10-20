"""Arrange selected panels from the existing vector PDFs without replotting data."""

import argparse
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path

from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import ArrayObject, ByteStringObject, FloatObject, NameObject, RectangleObject
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def path_sha(page):
    path_ops = {b"m", b"l", b"c", b"v", b"y", b"h", b"re", b"S", b"s",
                b"f", b"f*", b"B", b"B*", b"b", b"b*", b"n", b"W", b"W*"}
    paths = [(args, op) for args, op in page.get_contents().operations if op in path_ops]
    return hashlib.sha256(repr(paths).encode("utf-8")).hexdigest()


def text_width(block, fonts):
    """Measure the original embedded-font text, including PDF kerning."""
    cursor = 0.0
    line_x = 0.0
    right = 0.0
    font = None
    size = 0.0
    widths = {}
    for args, op in block:
        if op == b"Tf":
            font, size = args[0], float(args[1])
            descendant = fonts[font]["/DescendantFonts"][0].get_object()
            values = descendant["/W"]
            widths = {}
            i = 0
            while i < len(values):
                first, second = int(values[i]), values[i + 1]
                if isinstance(second, list):
                    widths.update({first + j: float(w) for j, w in enumerate(second)})
                    i += 2
                else:
                    widths.update({j: float(values[i + 2])
                                   for j in range(first, int(second) + 1)})
                    i += 3
        elif op == b"Td":
            line_x += float(args[0])
            cursor = line_x
        elif op in (b"TJ", b"Tj"):
            for item in (args[0] if op == b"TJ" else args):
                if isinstance(item, (str, bytes)):
                    raw = item.original_bytes if isinstance(item, str) else item
                    for j in range(0, len(raw), 2):
                        cid = int.from_bytes(raw[j:j + 2], "big")
                        cursor += widths.get(cid, 1000) * size / 1000
                else:
                    cursor -= float(item) * size / 1000
                right = max(right, cursor)
    return right


def single_column_text(page, clip, scale, panel):
    """Resize text blocks only; leave all curve/interval/marker paths intact."""
    content = page.get_contents()
    operations = content.operations
    fonts = page["/Resources"]["/Font"]
    out = []
    i = 0
    axis_left = 51.8854296 if panel in ("a", "c") else 229.6334424
    while i < len(operations):
        args, op = operations[i]
        if op != b"BT" or i == 0 or operations[i - 1][1] != b"cm":
            out.append((args, op))
            i += 1
            continue
        j = i + 1
        while operations[j][1] != b"ET":
            j += 1
        block = operations[i:j + 1]
        matrix = list(map(float, operations[i - 1][0]))
        x, y = matrix[4:]
        sizes = [float(a[1]) for a, operator in block if operator == b"Tf"]
        size = max(sizes)
        if not (clip.left <= x < clip.right and clip.bottom <= y < clip.top):
            out.extend(block)
            i = j + 1
            continue
        if size == 10:
            i = j + 1
            continue
        strings = [s for a, operator in block if operator == b"TJ"
                   for s in a[0] if isinstance(s, (str, bytes))]
        text = "".join((s.original_bytes if isinstance(s, str) else s).decode("utf-16-be")
                       for s in strings)
        lines = None
        if size == 6.8 and text == "No action penalty":
            lines = ("No action", "penalty")
        elif panel == "d" and size == 7 and text == "Full reward":
            lines = ("Full", "reward")
        if lines:
            font_args = next(a for a, operator in block if operator == b"Tf")
            block = [([], b"BT"), (font_args, b"Tf"),
                     ([FloatObject(0), FloatObject(3 if panel == "d" else 0)], b"Td")]
            for line_index, line in enumerate(lines):
                if line_index:
                    block.append(([FloatObject(0), FloatObject(-9)], b"Td"))
                block.append(([ArrayObject([ByteStringObject(line.encode("utf-16-be"))])], b"TJ"))
            block.append(([], b"ET"))
        target = 7.0 if size == 9 else (6.5 if size == 8 else 6.0)
        factor = max(1.0, target / (size * scale))
        length = text_width(block, fonts)
        vertical = abs(matrix[1]) > 0.5
        local_y = y - float(clip.bottom)
        if vertical or local_y < 20:
            anchor = 0.5
        elif x < axis_left - 2:
            anchor = 1.0
        elif panel == "d" and size == 7:
            anchor = 1.0
        else:
            anchor = 0.0
        dx = -(factor - 1) * length * anchor
        if panel == "d" and size == 7 and x < axis_left - 2:
            dx = axis_left - 4 - x - factor * length
        if not vertical:
            dx = max(float(clip.left) + 1 - x,
                     min(dx, float(clip.right) - 2 - x - factor * length))
        transform = [factor, 0, 0, factor, dx, 0]
        out += [([], b"q"), ([FloatObject(v) for v in transform], b"cm")]
        out.extend(block)
        out.append(([], b"Q"))
        i = j + 1
    content.operations = out
    page[NameObject("/Contents")] = content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True,
                        help="DejaVuSans-Bold.ttf for headings and panel letters.")
    parser.add_argument("--layout", choices=("2x3", "3x2"), default="2x3")
    parser.add_argument("--width", type=float, help="Output width in PDF points.")
    args = parser.parse_args()
    files = {arm: args.source / f"candidate_{arm}_2601.pdf" for arm in ("voltage", "action")}
    pages = {arm: PdfReader(path).pages[0] for arm, path in files.items()}
    width, source_height = map(float, (pages["voltage"].mediabox.width,
                                      pages["voltage"].mediabox.height))
    cell_height = 154.0
    if args.layout == "2x3":
        columns, top, gap, column_step = 3, 22.0, 32.0, 181.5
        width = 2 * 7 + 3 * 178.5 + 2 * 3
        height = top + 2 * cell_height + gap + 4
        output_width = args.width if args.width is not None else 372 * 72 / 72.27
        selections = [("voltage", panel) for panel in ("a", "b", "c")]
        selections += [("action", panel) for panel in ("a", "b", "d")]
    else:
        columns, top, gap, column_step = 2, 28.0, 12.0, 178.0
        height = top + 3 * cell_height + 2 * gap + 4
        output_width = args.width if args.width is not None else width
        selections = [("voltage", "a"), ("action", "a"),
                      ("voltage", "b"), ("action", "b"),
                      ("voltage", "c"), ("action", "d")]
    if output_width <= 0:
        parser.error("--width must be positive")
    scale = output_width / width
    writer = PdfWriter()
    target = writer.add_blank_page(width=width, height=height)
    original = {"a": (7, 8), "b": (185, 8),
                "c": (7, 177.388), "d": (185, 177.388)}
    overlay_buffer = BytesIO()
    overlay = canvas.Canvas(overlay_buffer, pagesize=(width, height))
    pdfmetrics.registerFont(TTFont("PanelBold", str(args.font)))
    if args.layout == "2x3":
        overlay.setFont("PanelBold", 8 / scale)
        overlay.drawCentredString(width / 2, height - 12, "Voltage penalties")
        overlay.drawCentredString(width / 2, height - cell_height - gap - 12,
                                 "Action penalty")
    else:
        overlay.setFont("PanelBold", 9)
        overlay.drawCentredString(99, height - 13, "Voltage penalties")
        overlay.drawCentredString(277, height - 13, "Action penalty")
    manifest = {"source_sha256": {str(p): sha(p) for p in files.values()},
                "renderer_sha256": sha(Path(__file__)),
                "font_sha256": sha(args.font), "layout": args.layout,
                "width_bp": output_width, "height_bp": height * scale,
                "panels": []}
    if args.layout == "2x3":
        manifest["typography_pt"] = {"ticks_and_legends": 6, "axis_labels": 6.5,
                                     "panel_titles": 7, "panel_letters": 7.5,
                                     "row_headings": 8}
    for index, (arm, panel) in enumerate(selections):
        row, column = divmod(index, columns)
        source_x, source_top = original[panel]
        clip = RectangleObject((source_x, source_height - source_top - cell_height,
                                source_x + 178.5, source_height - source_top))
        page = deepcopy(pages[arm])
        original_path_sha = path_sha(page)
        if args.layout == "2x3":
            single_column_text(page, clip, scale, panel)
            assert path_sha(page) == original_path_sha, "A plotted path changed"
        page.cropbox = clip
        page.trimbox = clip
        dest_x = 7 + column * column_step
        dest_y = height - top - row * (cell_height + gap) - cell_height
        target.merge_transformed_page(page, Transformation().translate(
            tx=dest_x - source_x, ty=dest_y - float(clip.bottom)))
        # Panel letters are renumbered; data paths are never reconstructed.
        letter_x = dest_x + (19.5676091201 if panel in ("a", "c") else 197.3156219732) - source_x
        if args.layout == "2x3":
            letter_x = dest_x
        else:
            overlay.setFillColorRGB(1, 1, 1)
            overlay.rect(letter_x - 1, dest_y + cell_height - 19, 10, 13, fill=1, stroke=0)
        overlay.setFillColorRGB(0, 0, 0)
        overlay.setFont("PanelBold", 7.5 / scale if args.layout == "2x3" else 10)
        overlay.drawString(letter_x, dest_y + cell_height - 16.8, "abcdef"[index])
        manifest["panels"].append({"label": "abcdef"[index], "source": str(files[arm]),
                                   "source_panel": panel, "crop_box": list(clip),
                                   "scale": scale, "unchanged_path_sha256": original_path_sha})
    overlay.save()
    overlay_buffer.seek(0)
    target.merge_page(PdfReader(overlay_buffer).pages[0])
    target.scale_by(scale)
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / f"reward_components_original_{args.layout}.pdf"
    with out.open("wb") as stream:
        writer.write(stream)
    manifest["output_sha256"] = sha(out)
    manifest["unchanged_learning_pdf"] = str(args.source / "candidate_learning.pdf")
    manifest["unchanged_learning_sha256"] = sha(args.source / "candidate_learning.pdf")
    (args.output / "composition_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
