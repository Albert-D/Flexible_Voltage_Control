"""Reflow SI Fig. 8 with unchanged published policy and envelope paths."""

import argparse
import ast
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


WIDTH, HEIGHT = 370.6102117061, 382.
BUS_IDS = (18, 21, 30, 45, 53)
UPPER_BOXES = ((34., 277., 120., 345.), (154., 277., 240., 345.),
               (274., 277., 360., 345.), (34., 178., 120., 246.),
               (154., 178., 240., 246.))
LOWER_BOX = (34., 31., 360., 108.)
INK, GRID, ZERO = '#242A2D', '#E5E9E6', '#919B95'
GREEN, RED, BLUE, SAFE = '#779B6E', '#C87C75', '#7D9ABA', '#F1F6EF'
NS = {'s': 'http://www.w3.org/2000/svg'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def translate_values(text):
    match = re.fullmatch(r'translate\(([-+.\d]+),\s*([-+.\d]+)\)', text)
    assert match, text
    return np.asarray([float(v) for v in match.groups()])


def svg_points(path):
    assert not re.search('[^MLZ0-9.,+ eE-]', path), path
    values = re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?', path)
    points = np.asarray([float(v) for v in values]).reshape(-1, 2)
    if path.endswith('Z'):
        points = np.vstack((points, points[0]))
    return points


def svg_axes(file):
    root = ET.parse(file).getroot()
    viewbox = [float(v) for v in root.get('viewBox').split()]
    assert viewbox[:2] == [0, 0]
    axes = []
    for group in root.findall('.//s:g', NS):
        if not re.fullmatch(r'x\d*y\d*', group.get('class', '')):
            continue
        origin = translate_values(group.get('transform'))
        clip_id = group.get('clip-path')[5:-1]
        clip = next(g for g in root.findall('.//s:clipPath', NS) if g.get('id') == clip_id)[0]
        size = np.asarray([float(clip.get('width')), float(clip.get('height'))])
        paths = []
        for element in group.findall('.//s:path', NS):
            style = element.get('style', '')
            if element.get('class') not in ('js-line', 'js-fill'):
                continue
            if 'stroke-opacity: 0;' in style and element.get('class') == 'js-line':
                continue
            points = svg_points(element.get('d'))
            if len(points) < 3:
                continue
            filled = element.get('class') == 'js-fill'
            color = re.search(r'(?:fill|stroke): rgb\((\d+), (\d+), (\d+)\)', style)
            assert color
            paths.append({'points': points, 'filled': filled,
                          'color': np.asarray([int(v) for v in color.groups()]) / 255})
        axes.append({'origin': origin, 'size': size, 'paths': paths})
    return axes, np.asarray(viewbox[2:])


def saved_metadata(notebook):
    document = json.loads(notebook.read_text(encoding='utf-8'))
    by_id = {cell.get('id'): cell for cell in document['cells']}
    upper_cell, lower_cell = by_id['cf2b1c8e'], by_id['cb964cdb']
    figure = next(o['data']['application/vnd.plotly.v1+json'] for o in upper_cell['outputs']
                  if 'application/vnd.plotly.v1+json' in o.get('data', {}))
    lower_figure = next(o['data']['application/vnd.plotly.v1+json'] for o in lower_cell['outputs']
                        if 'application/vnd.plotly.v1+json' in o.get('data', {}))
    ranges = [figure['layout']['yaxis' + (str(i + 1) if i else '')]['range'] for i in range(5)]
    source = ''.join(lower_cell['source'])
    indices = {node.slice.value for node in ast.walk(ast.parse(source))
               if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
               and node.value.id == 'policy_net' and isinstance(node.slice, ast.Constant)}
    assert indices == {2}, indices
    setup = ''.join(document['cells'][0]['source'])
    assignment = next(node for node in ast.walk(ast.parse(setup)) if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'injection_bus' for target in node.targets))
    ids = ast.literal_eval(assignment.value.left.args[0])
    assert tuple(ids) == BUS_IDS
    upper_source = ast.parse(''.join(upper_cell['source']))
    sweep = next(node.value for node in ast.walk(upper_source) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == 'state_val' for target in node.targets))
    count = next(node.value.value for node in ast.walk(upper_source) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == 'N' for target in node.targets))
    assert isinstance(sweep, ast.BinOp) and isinstance(sweep.op, ast.Add)
    assert isinstance(sweep.right, ast.BinOp) and isinstance(sweep.right.op, ast.Mult)
    first, increment = sweep.left.value, sweep.right.left.value
    assert (first, increment, count) == (.8, .001, 400)
    physical_range = [12 * first, 12 * (first + (count - 1) * increment)]
    return ranges, lower_figure['layout']['yaxis']['range'], {
        'notebook': str(notebook.resolve()), 'notebook_sha256': digest(notebook),
        'upper_cell_id': 'cf2b1c8e', 'lower_cell_id': 'cb964cdb',
        'topology_agent_index': 2, 'topology_bus': ids[2],
        'historical_setup': '2025-02-18 Step-500 Seed-4 family, read only; no model executed',
        'preexisting_processing': 'The historical exported paths include moving-average and Savitzky-Golay preprocessing; no additional smoothing is applied.',
        'upper_input_voltage_pu': [first, first + (count - 1) * increment],
        'upper_physical_voltage_kv': physical_range,
        'caption_alignment_items': [
            'Existing caption calls the lower example Bus 18; source selects the third agent, Bus 30.',
            'Existing caption states an 11.4-12.6 kV shaded range; original upper SVG uses 11.599-12.399 kV.',
            'Upper source mislabeled a 0.80+j*0.001 p.u. input sweep as linspace(10,14,400) kV. The preview corrects tick labels to the actual 9.60-14.388 kV input range, preserving all curve and shaded-area vertices.'
        ]}


def color_of(path, filled=False):
    color = path['non_stroking_color' if filled else 'stroking_color']
    return np.asarray([color] * 3 if isinstance(color, (float, int)) else color)


def match_component(component_axes, view_size, picture, candidates):
    scale = np.asarray([picture['width'], picture['height']]) / view_size
    start = np.asarray([picture['x0'], picture['top']])
    pool = candidates.copy()
    panels, errors = [], []
    for axis in component_axes:
        origin = axis['origin'] * scale + start
        size = axis['size'] * scale
        frame = (*origin, *(origin + size))
        matches = []
        for svg_path in axis['paths']:
            expected = (svg_path['points'] + axis['origin']) * scale + start
            options = [p for p in pool if p['fill'] == svg_path['filled']
                       and len(p['pts']) == len(expected)
                       and np.max(np.abs(color_of(p, svg_path['filled']) - svg_path['color'])) < .0006]
            assert options
            error = lambda p: float(np.max(np.abs(np.asarray(p['pts']) - expected)))
            best = min(options, key=error)
            residual = error(best)
            assert residual < .025, residual
            errors.append(residual)
            pool.remove(best)
            matches.append(best)
        panels.append({'frame': frame, 'paths': matches})
    assert not pool
    return panels, max(errors)


def load_original(source, components, notebook):
    upper_axes, upper_size = svg_axes(components / 'enhanced_policy_plot.svg')
    lower_axes, lower_size = svg_axes(components / 'x_policy_plot_improved.svg')
    assert len(upper_axes) == 5 and len(lower_axes) == 1
    upper_ranges, lower_range, metadata = saved_metadata(notebook)
    with pdfplumber.open(source) as document:
        assert len(document.pages) == 1
        page = document.pages[0]
        assert (page.width, page.height) == (720, 540)
        paths = page.curves
        assert len(paths) == 14
        assert all(command[0] in ('m', 'l', 'h') for path in paths for command in path['path'])
        upper_picture = next(r for r in page.rects if 719 < r['width'] < 721 and 239 < r['height'] < 241)
        lower_picture = next(r for r in page.rects if 338 < r['width'] < 339 and 241 < r['height'] < 243)
        upper, upper_error = match_component(upper_axes, upper_size, upper_picture,
                                             [p for p in paths if p['top'] < 270])
        lower, lower_error = match_component(lower_axes, lower_size, lower_picture,
                                             [p for p in paths if p['top'] > 270])
        shades = sorted([r for r in page.rects if isinstance(r['non_stroking_color'], tuple)
                         and abs(r['non_stroking_color'][0] - .565) < .001
                         and r['height'] > 700], key=lambda r: r['x0'])
        assert len(shades) == 5
        for index, panel in enumerate(upper):
            panel['range'] = upper_ranges[index]
            panel['x_range'] = metadata['upper_physical_voltage_kv']
            panel['shade'] = (shades[index]['x0'], shades[index]['x1'])
            panel['bus'] = BUS_IDS[index]
        lower[0]['range'] = lower_range
        lower[0]['x_range'] = [10., 14.]
        metadata['component_svg_to_published_pdf_maximum_residual_bp'] = max(upper_error, lower_error)
        return upper, lower[0], metadata


def transform(points, source, box):
    points = np.asarray(points)
    x0, y0, x1, y1 = source
    left, bottom, right, top = box
    return np.column_stack((left + (points[:, 0] - x0) / (x1 - x0) * (right - left),
                            top - (points[:, 1] - y0) / (y1 - y0) * (top - bottom)))


def original_path(c, source_path, frame, box):
    path = c.beginPath()
    for command in source_path['path']:
        if command[0] == 'h':
            path.close()
        else:
            x, y = transform([command[1]], frame, box)[0]
            (path.moveTo if command[0] == 'm' else path.lineTo)(x, y)
    c.drawPath(path, stroke=int(source_path['stroke']), fill=int(source_path['fill']))


def ink(c, size=6.5, bold=False):
    c.setFillColor(HexColor(INK))
    c.setFont('Arial-Bold' if bold else 'Arial', size)


def legend_item(c, x, y, label, color, dash=None, band=False):
    c.saveState()
    if band:
        c.setFillColor(HexColor(color))
        c.setFillAlpha(.14 if color == GREEN else 1)
        c.rect(x, y - 2.4, 16, 4.8, fill=1, stroke=0)
    else:
        c.setStrokeColor(HexColor(color))
        c.setLineWidth(1.15)
        c.setDash(dash or [])
        c.line(x, y, x + 16, y)
    c.restoreState()
    ink(c, size=6.8)
    c.drawString(x + 21, y - 2.3, label)
    return 21 + pdfmetrics.stringWidth(label, 'Arial', 6.8)


def axes(c, panel, box, ticks, xticks):
    left, bottom, right, top = box
    low, high = panel['range']
    y = lambda value: bottom + (value - low) / (high - low) * (top - bottom)
    c.setStrokeColor(HexColor(GRID))
    c.setLineWidth(.4)
    for value in ticks:
        c.line(left, y(value), right, y(value))
    c.setStrokeColor(HexColor(ZERO))
    c.setLineWidth(.55)
    c.setDash([1, 2])
    c.line(left, y(0), right, y(0))
    c.setDash([])
    c.setStrokeColor(HexColor(INK))
    c.setLineWidth(.65)
    c.line(left, bottom, right, bottom)
    c.line(left, bottom, left, top)
    ink(c)
    for value in ticks:
        c.line(left - 2.2, y(value), left, y(value))
        c.drawRightString(left - 4, y(value) - 2.2, f'{value:g}')
    for value in xticks:
        x_low, x_high = panel['x_range']
        x = left + (value - x_low) / (x_high - x_low) * (right - left)
        c.line(x, bottom, x, bottom - 2.2)
        c.drawCentredString(x, bottom - 11, str(value))


def draw_upper(c, panel, index):
    box = UPPER_BOXES[index]
    left, bottom, right, top = box
    ink(c, size=7.8, bold=True)
    c.drawCentredString((left + right) / 2, top + 10, f"Bus {panel['bus']}")
    shade_left, shade_right = transform([(panel['shade'][0], panel['frame'][1]),
                                        (panel['shade'][1], panel['frame'][1])], panel['frame'], box)[:, 0]
    c.setFillColor(HexColor(SAFE))
    c.rect(shade_left, bottom, shade_right - shade_left, top - bottom, fill=1, stroke=0)
    ticks = (-2, 0, 2) if index == 0 else ((-1, 0, 1) if index in (1, 2) else (-.8, 0, .8))
    axes(c, panel, box, ticks, (10, 12, 14))
    c.saveState()
    clip = c.beginPath()
    clip.rect(left, bottom, right - left, top - bottom)
    c.clipPath(clip, stroke=0, fill=0)
    for index_path, path in enumerate(panel['paths']):
        c.setStrokeColor(HexColor(RED if index_path == 0 else GREEN))
        c.setLineWidth(.85 if index_path == 0 else 1.25)
        c.setDash([2.5, 1.8] if index_path == 0 else [])
        c.setLineJoin(1)
        original_path(c, path, panel['frame'], box)
    c.restoreState()


def draw_lower(c, panel):
    left, bottom, right, top = box = LOWER_BOX
    axes(c, panel, box, (-.8, -.4, 0, .4, .8), (10, 11, 12, 13, 14))
    c.saveState()
    clip = c.beginPath()
    clip.rect(left, bottom, right - left, top - bottom)
    c.clipPath(clip, stroke=0, fill=0)
    colors, dashes = (GREEN, GREEN, BLUE, RED), ([], [], [3, 1.8], [1, 1.8])
    for index, path in enumerate(panel['paths']):
        c.setStrokeColor(HexColor(colors[index]))
        c.setFillColor(HexColor(colors[index]))
        c.setFillAlpha(.14 if path['fill'] else 1)
        c.setLineWidth(1.25)
        c.setDash(dashes[index])
        c.setLineJoin(1)
        original_path(c, path, panel['frame'], box)
    c.restoreState()


def render(upper, lower, target, font_dir):
    pdfmetrics.registerFont(TTFont('Arial', str(font_dir / 'arial.ttf')))
    pdfmetrics.registerFont(TTFont('Arial-Bold', str(font_dir / 'arialbd.ttf')))
    c = canvas.Canvas(str(target), pagesize=(WIDTH, HEIGHT), invariant=1, pageCompression=1)
    c.setTitle('Bus-dependent and topology-dependent control policies')
    c.setAuthor('RLC-FT')
    for letter, y, title in [('a', 368, 'Policies across buses'),
                              ('b', 132, 'Policies across topologies (Bus 30)')]:
        ink(c, size=9, bold=True)
        c.drawString(5, y, letter)
        ink(c, size=8.2, bold=True)
        c.drawCentredString(WIDTH / 2, y, title)
    for index, panel in enumerate(upper):
        draw_upper(c, panel, index)
    legend_item(c, 284, 218, 'RLC-FT', GREEN)
    legend_item(c, 284, 202, 'Linear', RED, [2.5, 1.8])
    legend_item(c, 284, 186, 'Dead zone', SAFE, band=True)
    legend_labels = [('Topology 1', GREEN, [], False), ('Topology 2', BLUE, [3, 1.8], False),
                     ('Topology 3', RED, [1, 1.8], False), ('Min-max (10 topologies)', GREEN, [], True)]
    widths = [21 + pdfmetrics.stringWidth(item[0], 'Arial', 6.8) for item in legend_labels]
    x = (WIDTH - sum(widths) - 3 * 10) / 2
    for item, width in zip(legend_labels, widths):
        legend_item(c, x, 119, *item)
        x += width + 10
    draw_lower(c, lower)
    ink(c, size=7.3)
    c.drawCentredString(WIDTH / 2, 150, 'Voltage (kV)')
    c.drawCentredString((LOWER_BOX[0] + LOWER_BOX[2]) / 2, 10, 'Voltage (kV)')
    for y in (261.5, (LOWER_BOX[1] + LOWER_BOX[3]) / 2):
        c.saveState()
        c.translate(8, y)
        c.rotate(90)
        ink(c, size=7.3)
        c.drawCentredString(0, 0, 'Q (MVAr)')
        c.restoreState()
    c.showPage()
    c.save()


def verify(upper, lower, target):
    originals = upper + [lower]
    boxes = list(UPPER_BOXES) + [LOWER_BOX]
    with pdfplumber.open(target) as document:
        page = document.pages[0]
        candidates = [p for p in page.curves if len(p['pts']) > 3]
        assert len(candidates) == 14
        errors, sizes = [], []
        for panel, box in zip(originals, boxes):
            for old in panel['paths']:
                expected = transform(old['pts'], panel['frame'], box)
                options = [p for p in candidates if p['fill'] == old['fill']
                           and len(p['pts']) == len(expected)]
                assert options
                def error(p):
                    observed = np.asarray([(x, HEIGHT - y) for x, y in p['pts']])
                    return float(np.max(np.abs(observed - expected)))
                best = min(options, key=error)
                residual = error(best)
                assert residual < .001
                assert [p[0] for p in old['path']] == [p[0] for p in best['path']]
                errors.append(residual)
                sizes.append(len(best['pts']))
                candidates.remove(best)
        visible = [char for char in page.chars if char['text'].strip()]
        assert min(c['size'] for c in visible if c['upright']) >= 6.49
        assert all(c['x0'] >= 0 and c['x1'] <= WIDTH and c['top'] >= 0 and c['bottom'] <= HEIGHT for c in visible)
        return {'preserved_paths': len(sizes), 'path_vertices': sizes, 'total_vertices': sum(sizes),
                'maximum_affine_error_bp': max(errors), 'same_path_commands': True,
                'original_viewports_and_clipping_retained': True, 'original_shaded_geometry_retained': True,
                'upper_tick_labels_corrected_to_actual_input_kv': True,
                'data_recomputed': False, 'additional_smoothing': False,
                'minimum_essential_font_pt': min(c['size'] for c in visible if c['upright']),
                'all_text_within_page': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--components', type=Path, required=True)
    parser.add_argument('--notebook', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--font-dir', type=Path, default=Path('C:/Windows/Fonts'))
    args = parser.parse_args()
    source_hash = digest(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    originals = args.output / 'originals'
    originals.mkdir(exist_ok=True)
    inputs = [args.source, args.components / 'enhanced_policy_plot.svg', args.components / 'x_policy_plot_improved.svg']
    for file in inputs:
        backup = originals / file.name
        if backup.exists():
            assert digest(backup) == digest(file)
        else:
            shutil.copy2(file, backup)
    upper, lower, metadata = load_original(originals / args.source.name, originals, args.notebook)
    target = args.output / 'fig08_policy_adaptiveness_preview.pdf'
    render(upper, lower, target, args.font_dir)
    checks = verify(upper, lower, target)
    assert source_hash == digest(args.source) == digest(originals / args.source.name)
    manifest = {'status': 'preview_only_not_adopted', 'source_pdf': str(args.source.resolve()),
                'source_pdf_sha256': source_hash, 'source_metadata': metadata,
                'source_components': [{'file': str(file.resolve()), 'sha256': digest(file)} for file in inputs[1:]],
                'code': str(Path(__file__).resolve()), 'code_sha256': digest(__file__),
                'output_pdf': str(target.resolve()), 'output_sha256': digest(target),
                'dimensions_pdf_bp': [WIDTH, HEIGHT], 'verification': checks,
                'upper_axes_ranges': [p['range'] for p in upper], 'lower_axes_range': lower['range'],
                'upper_physical_voltage_range_kv': upper[0]['x_range'], 'lower_voltage_range_kv': lower['x_range'],
                'style': ['Five bus panels in two rows', 'Wide topology panel with external condition key',
                          'Arial at actual SI width', 'Main method palette: green RLC-FT, muted-red Linear',
                          'Preserved original policy shapes and min-max envelope']}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
