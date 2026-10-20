"""Restyle SI Fig. 13 from its original vector paths after SVG cross-checks."""

import argparse
import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pdfplumber
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


WIDTH, HEIGHT = 370.6102117061, 372.0
COUNTS = (10, 5, 3, 1)
BUS_IDS = ((16, 19, 20, 22, 24, 31, 35, 46, 47, 54),
           (19, 22, 31, 46, 54), (18, 30, 53), (30,))
BOXES = ((34., 202., 177., 302.), (217., 202., 360., 302.),
         (34., 39., 177., 139.), (217., 39., 360., 139.))
BUS_COLORS = {
    16: '#6F91B2', 19: '#C87C75', 20: '#779B6E', 22: '#9B84AB',
    24: '#C4AA63', 31: '#5C9C9B', 35: '#BB819B', 46: '#526C88',
    47: '#AC8D71', 54: '#94AA62', 18: '#6F91B2', 30: '#779B6E',
    53: '#C87C75',
}
OLD_COLORS = ((31, 119, 180), (255, 127, 14), (44, 160, 44),
              (214, 39, 40), (148, 103, 189), (140, 86, 75),
              (227, 119, 194), (127, 127, 127), (188, 189, 34),
              (23, 190, 207), (211, 211, 211))
INK, GRID, BACKGROUND = '#242A2D', '#E6EAE8', '#C3C9C6'
SAFE, LIMIT, NOMINAL = '#F1F6EF', '#687A65', '#8D9691'
NS = {'s': 'http://www.w3.org/2000/svg'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rgb(color):
    return np.asarray([color] * 3 if isinstance(color, (float, int)) else color)


def svg_geometry(file):
    root = ET.parse(file).getroot()
    assert root.get('viewBox') == '0 0 700 500'
    xy = next(g for g in root.findall('.//s:g', NS) if g.get('class') == 'xy')
    clip_id = xy.get('clip-path')[5:-1]
    clip = next(g for g in root.findall('.//s:clipPath', NS) if g.get('id') == clip_id)[0]
    assert (float(clip.get('width')), float(clip.get('height'))) == (463, 360)
    traces = []
    for element in xy.findall('.//s:path', NS):
        if element.get('class') != 'js-line':
            continue
        path = element.get('d')
        assert not re.search('[^ML0-9.,+ eE-]', path), path
        numbers = re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?', path)
        points = np.asarray([float(v) for v in numbers]).reshape(-1, 2)
        color = re.search(r'stroke: rgb\((\d+), (\d+), (\d+)\)', element.get('style'))
        traces.append({'points': points, 'color': tuple(int(v) for v in color.groups())})
    labels = [int(t.text[4:]) for t in root.findall('.//s:text', NS)
              if re.fullmatch(r'Bus \d+', t.text or '')]
    assert len(traces) == 51
    return traces, labels


def original_panels(source, component_dir):
    with pdfplumber.open(source) as document:
        assert len(document.pages) == 1
        page = document.pages[0]
        assert (page.width, page.height) == (960, 540)
        trajectories = [p for p in page.curves + page.lines
                        if p['stroke'] and not p['fill'] and p['width'] > 100
                        and all(command[0] in ('m', 'l') for command in p['path'])
                        and any(np.max(np.abs(rgb(p['stroking_color']) - np.asarray(color) / 255))
                                < .0006 for color in OLD_COLORS)]
        panels = []
        for index, count in enumerate(COUNTS):
            left_column, top_row = index % 2 == 0, index < 2
            picture = next(r for r in page.rects if 370 < r['width'] < 380
                           and 268 < r['height'] < 271
                           and (r['x0'] < 480) == left_column
                           and (r['top'] < 100) == top_row)
            scale_x, scale_y = picture['width'] / 700, picture['height'] / 500
            origin_x = picture['x0'] + 96 * scale_x
            origin_y = picture['top'] + 60 * scale_y
            old_box = (origin_x, origin_y, origin_x + 463 * scale_x,
                       origin_y + 360 * scale_y)
            pool = [p for p in trajectories if (p['x0'] < 480) == left_column
                    and (p['pts'][-1][1] < 270) == top_row]
            assert len(pool) == 51, (count, len(pool))
            svg_file = component_dir / f'voltage_plot_{count}_buses.svg'
            svg_traces, labels = svg_geometry(svg_file)
            assert labels == list(BUS_IDS[index]), labels
            colored_ids = iter(labels)
            matches = []
            source_error = 0.
            # The components identify every published path. Render the PDF paths
            # themselves so SVG coordinate rounding cannot affect the preview.
            for svg_trace in svg_traces:
                expected = svg_trace['points'] * (scale_x, scale_y) + (origin_x, origin_y)
                candidates = [p for p in pool if len(p['pts']) == len(expected)
                              and np.max(np.abs(rgb(p['stroking_color'])
                                                - np.asarray(svg_trace['color']) / 255)) < .0006]
                assert candidates
                best = min(candidates, key=lambda p: np.max(np.abs(np.asarray(p['pts']) - expected)))
                residual = float(np.max(np.abs(np.asarray(best['pts']) - expected)))
                assert residual < .025, (count, residual)
                source_error = max(source_error, residual)
                pool.remove(best)
                controlled = svg_trace['color'] != (211, 211, 211)
                matches.append({'path': best['path'], 'points': best['pts'],
                                'bus': next(colored_ids) if controlled else None})
            assert not pool
            assert sum(p['bus'] is not None for p in matches) == count
            panels.append({'count': count, 'old_box': old_box, 'traces': matches,
                           'component': str(svg_file.resolve()), 'component_sha256': digest(svg_file),
                           'component_match_maximum_residual_pdf_pt': source_error})
        assert sum(len(p['traces']) for p in panels) == len(trajectories) == 204
        return panels


def transform(points, old_box, box):
    old_left, old_top, old_right, old_bottom = old_box
    left, bottom, right, top = box
    points = np.asarray(points)
    return np.column_stack((left + (points[:, 0] - old_left) / (old_right - old_left) * (right - left),
                            top - (points[:, 1] - old_top) / (old_bottom - old_top) * (top - bottom)))


def draw_trace(c, trace, old_box, box):
    new_points = iter(transform(trace['points'], old_box, box))
    path = c.beginPath()
    for item in trace['path']:
        point = next(new_points)
        (path.moveTo if item[0] == 'm' else path.lineTo)(*point)
    c.drawPath(path, stroke=1, fill=0)


def set_ink(c, font='Arial', size=6.5):
    c.setFillColor(HexColor(INK))
    c.setFont(font, size)


def global_key(c):
    labels = ('Uncontrollable buses', 'Safe range (11.4-12.6 kV)', 'Nominal (12 kV)')
    widths = [18 + pdfmetrics.stringWidth(text, 'Arial', 6.5) for text in labels]
    x = (WIDTH - sum(widths) - 2 * 14) / 2
    y = 357
    for index, (text, width) in enumerate(zip(labels, widths)):
        c.saveState()
        if index == 0:
            c.setStrokeColor(HexColor(BACKGROUND))
            c.setLineWidth(1.0)
            c.line(x, y, x + 12, y)
        elif index == 1:
            c.setFillColor(HexColor(SAFE))
            c.rect(x, y - 3, 12, 6, fill=1, stroke=0)
            c.setStrokeColor(HexColor(LIMIT))
            c.setLineWidth(.65)
            c.setDash([2, 1.4])
            c.line(x, y - 3, x + 12, y - 3)
            c.line(x, y + 3, x + 12, y + 3)
        else:
            c.setStrokeColor(HexColor(NOMINAL))
            c.setLineWidth(.6)
            c.setDash([1, 2])
            c.line(x, y, x + 12, y)
        c.restoreState()
        set_ink(c)
        c.drawString(x + 18, y - 2.2, text)
        x += width + 14


def bus_key(c, bus_ids, left, y):
    set_ink(c)
    c.drawString(left, y - 2.2, 'Bus')
    for i, bus in enumerate(bus_ids):
        column, row = i % 5, i // 5
        x, baseline = left + 19 + 24.5 * column, y - 10 * row
        c.setStrokeColor(HexColor(BUS_COLORS[bus]))
        c.setLineWidth(1.2)
        c.line(x, baseline, x + 7, baseline)
        set_ink(c)
        c.drawString(x + 9, baseline - 2.2, str(bus))


def draw_panel(c, panel, index):
    left, bottom, right, top = box = BOXES[index]
    title_y, legend_y = (337, 324) if index < 2 else (174, 161)
    set_ink(c, 'Arial-Bold', 9)
    c.drawString(left - 21, title_y, 'abcd'[index])
    set_ink(c, 'Arial-Bold', 7.8)
    noun = 'bus' if panel['count'] == 1 else 'buses'
    c.drawString(left, title_y, f"{panel['count']} controllable {noun}")
    bus_key(c, BUS_IDS[index], left, legend_y)
    y = lambda value: bottom + (value - 11) / 2 * (top - bottom)
    x = lambda value: left + (value + .5) / 51 * (right - left)
    c.setFillColor(HexColor(SAFE))
    c.rect(left, y(11.4), right - left, y(12.6) - y(11.4), fill=1, stroke=0)
    c.setStrokeColor(HexColor(GRID))
    c.setLineWidth(.4)
    for value in (11., 11.5, 12., 12.5, 13.):
        c.line(left, y(value), right, y(value))
    c.saveState()
    clip = c.beginPath()
    clip.rect(left, bottom, right - left, top - bottom)
    c.clipPath(clip, stroke=0, fill=0)
    c.setLineJoin(1)
    c.setLineCap(1)
    for trace in panel['traces']:
        color = BACKGROUND if trace['bus'] is None else BUS_COLORS[trace['bus']]
        c.setStrokeColor(HexColor(color))
        c.setStrokeAlpha(.8 if trace['bus'] is None else 1.)
        c.setLineWidth(.55 if trace['bus'] is None else 1.15)
        draw_trace(c, trace, panel['old_box'], box)
    c.setStrokeAlpha(1)
    c.setStrokeColor(HexColor(NOMINAL))
    c.setLineWidth(.55)
    c.setDash([1, 2])
    c.line(left, y(12), right, y(12))
    c.setStrokeColor(HexColor(LIMIT))
    c.setLineWidth(.65)
    c.setDash([2.6, 1.8])
    for value in (11.4, 12.6):
        c.line(left, y(value), right, y(value))
    c.restoreState()
    c.setStrokeColor(HexColor(INK))
    c.setLineWidth(.65)
    c.line(left, bottom, right, bottom)
    c.line(left, bottom, left, top)
    set_ink(c)
    for value in (11., 11.5, 12., 12.5, 13.):
        c.line(left - 2.3, y(value), left, y(value))
        c.drawRightString(left - 4.5, y(value) - 2.2, f'{value:.1f}')
    for step in (0, 10, 20, 30, 40, 50):
        c.line(x(step), bottom, x(step), bottom - 2.3)
        c.drawCentredString(x(step), bottom - 11, str(step))
    if index >= 2:
        set_ink(c, size=7.3)
        c.drawCentredString((left + right) / 2, bottom - 26, 'Control step')
    if index % 2 == 0:
        c.saveState()
        c.translate(8, (bottom + top) / 2)
        c.rotate(90)
        set_ink(c, size=7.3)
        c.drawCentredString(0, 0, 'Bus voltage (kV)')
        c.restoreState()


def render(panels, target, font_dir):
    pdfmetrics.registerFont(TTFont('Arial', str(font_dir / 'arial.ttf')))
    pdfmetrics.registerFont(TTFont('Arial-Bold', str(font_dir / 'arialbd.ttf')))
    c = canvas.Canvas(str(target), pagesize=(WIDTH, HEIGHT), invariant=1, pageCompression=1)
    c.setTitle('Voltage trajectories under sparse control')
    c.setAuthor('RLC-FT')
    global_key(c)
    for index, panel in enumerate(panels):
        draw_panel(c, panel, index)
    c.showPage()
    c.save()


def verify(panels, target):
    with pdfplumber.open(target) as document:
        page = document.pages[0]
        colors = [HexColor(value).rgb() for value in [BACKGROUND] + list(BUS_COLORS.values())]
        paths = [p for p in page.curves + page.lines if p['stroke'] and not p['fill']
                 and p['width'] > 80 and all(command[0] in ('m', 'l') for command in p['path'])
                 and any(np.max(np.abs(rgb(p['stroking_color']) - color)) < .00001 for color in colors)]
        assert len(paths) == 204, len(paths)
        residuals, panel_checks = [], []
        for index, panel in enumerate(panels):
            left, bottom, right, top = BOXES[index]
            pool = [p for p in paths if left - .01 < p['x0'] < right
                    and (p['pts'][-1][1] < HEIGHT - 175) == (index < 2)]
            assert len(pool) == 51, (index, len(pool))
            for old in panel['traces']:
                color = HexColor(BACKGROUND if old['bus'] is None else BUS_COLORS[old['bus']]).rgb()
                expected = transform(old['points'], panel['old_box'], BOXES[index])
                candidates = [p for p in pool if len(p['pts']) == len(expected)
                              and np.max(np.abs(rgb(p['stroking_color']) - color)) < .00001]
                assert candidates
                def error(p):
                    observed = np.asarray([(x, HEIGHT - y) for x, y in p['pts']])
                    return float(np.max(np.abs(observed - expected)))
                best = min(candidates, key=error)
                residual = error(best)
                assert residual < .001, residual
                assert [p[0] for p in old['path']] == [p[0] for p in best['path']]
                residuals.append(residual)
                pool.remove(best)
            panel_checks.append({'controllable_buses': panel['count'], 'bus_ids': list(BUS_IDS[index]),
                                 'trajectory_count': 51,
                                 'path_vertices': sum(len(p['points']) for p in panel['traces']),
                                 'component_svg': panel['component'],
                                 'component_svg_sha256': panel['component_sha256'],
                                 'component_match_maximum_residual_pdf_pt': panel['component_match_maximum_residual_pdf_pt']})
        visible = [c for c in page.chars if c['text'].strip()]
        assert min(c['size'] for c in visible if c['upright']) >= 6.49
        assert all(c['x0'] >= 0 and c['x1'] <= WIDTH and c['top'] >= 0 and c['bottom'] <= HEIGHT for c in visible)
        return {'panels': panel_checks, 'preserved_trajectories': len(paths),
                'preserved_path_vertices': sum(v['path_vertices'] for v in panel_checks),
                'maximum_affine_error_pdf_pt': max(residuals), 'same_path_commands': True,
                'axes_ranges_unchanged': {'control_step': [-.5, 50.5], 'voltage_kv': [11, 13]},
                'minimum_upright_font_pt': min(c['size'] for c in visible if c['upright']),
                'all_text_within_page': True, 'data_recomputed': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--components', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--font-dir', type=Path, default=Path('C:/Windows/Fonts'))
    args = parser.parse_args()
    source_hash = digest(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    originals = args.output / 'originals'
    originals.mkdir(exist_ok=True)
    backup = originals / args.source.name
    if backup.exists():
        assert digest(backup) == source_hash
    else:
        shutil.copy2(args.source, backup)
    for count in COUNTS:
        component = args.components / f'voltage_plot_{count}_buses.svg'
        saved = originals / component.name
        if saved.exists():
            assert digest(saved) == digest(component)
        else:
            shutil.copy2(component, saved)
    panels = original_panels(backup, originals)
    target = args.output / 'fig13_sparse_control_preview.pdf'
    render(panels, target, args.font_dir)
    checks = verify(panels, target)
    assert source_hash == digest(args.source) == digest(backup)
    manifest = {'status': 'preview_only_not_adopted', 'source_pdf': str(args.source.resolve()),
                'source_pdf_sha256': source_hash, 'preserved_original': str(backup.resolve()),
                'code': str(Path(__file__).resolve()), 'code_sha256': digest(__file__),
                'output_pdf': str(target.resolve()), 'output_sha256': digest(target),
                'dimensions_pdf_bp': [WIDTH, HEIGHT], 'target_plot_boxes': BOXES,
                'verification': checks,
                'provenance': 'Original published PDF paths; corresponding SVG components identify all paths. No raw simulation arrays reconstructed.',
                'changes': ['Four labeled panels at actual SI width', '6.5 pt ticks and legends',
                            '7.3 pt axes and 7.8 pt panel headings', 'External compact bus keys',
                            'Same bus uses the same color across panels', 'Remove dense point markers',
                            'Lighter gray uncontrolled trajectories', 'Shared reference key and safe-range shading']}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
