"""Build the reference-format SignDuplex institution banner.

Run: python docs/assets/build_brand_banner.py
Requires matplotlib plus either CairoSVG or system librsvg/Cairo.
All institution geometry comes from the adjacent official vector SVG files.
No Chinese font is required: 清华大学 is the actual official calligraphy path.
"""
from pathlib import Path
import copy
import ctypes as C
import ctypes.util
import json
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_svg import _path

ROOT = Path(__file__).resolve().parent
NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', NS)
PURPLE = '#85448f'
RED = '#c7000b'


def element(tag, attrs, parent=None):
    node = ET.Element('{'+NS+'}'+tag, {k: str(v) for k, v in attrs.items()})
    if parent is not None:
        parent.append(node)
    return node


def insert_svg(parent, name, x, y, width, height):
    node = ET.parse(ROOT/name).getroot()
    node.attrib.update(x=str(x), y=str(y), width=str(width), height=str(height))
    parent.append(node)


def vector_text(parent, value, x, baseline, size, color, family='DejaVu Sans', weight='normal', centered=False, target_width=None):
    shape = TextPath((0, 0), value, size=size, prop=FontProperties(family=family, weight=weight))
    bounds = shape.get_extents()
    if target_width is not None:
        shape = TextPath((0, 0), value, size=size*target_width/bounds.width, prop=FontProperties(family=family, weight=weight))
        bounds = shape.get_extents()
    if centered:
        x -= (bounds.x0 + bounds.x1)/2
    data = _path.convert_to_string(shape, None, None, False, None, 6, [b'M', b'L', b'Q', b'C', b'z'], False).decode()
    element('path', {'d': data, 'transform': f'translate({x} {baseline}) scale(1 -1)', 'fill': color}, parent)


def huawei_horizontal():
    """Arrange official flower and lettering paths horizontally without warping."""
    source = ET.parse(ROOT/'huawei-logo.svg').getroot()
    root = element('svg', {'width': 512, 'height': 150, 'viewBox': '0 0 512 150'})
    definitions = source.find('{'+NS+'}defs')
    if definitions is not None:
        definitions = copy.deepcopy(definitions)
        for node in list(definitions):
            if node.tag.endswith('style'):
                definitions.remove(node)
        root.append(definitions)
    symbol = next(node for node in source.iter() if node.get('class') == 'cls-3')
    group = element('g', {'transform': 'translate(2 5) scale(3.14) translate(-4.5 -3.33)'}, root)
    group.append(copy.deepcopy(symbol))
    group = element('g', {'transform': 'translate(215 48) scale(5.05) translate(-4.63 -52.92)'}, root)
    for node in source.iter():
        if node.get('class') == 'cls-2':
            group.append(copy.deepcopy(node))
    return root


def render_png(svg_path, png_path, width, height):
    try:
        import cairosvg
    except ImportError:
        cairo = C.CDLL(ctypes.util.find_library('cairo'))
        rsvg = C.CDLL(ctypes.util.find_library('rsvg-2'))
        gobject = C.CDLL(ctypes.util.find_library('gobject-2.0'))
        cairo.cairo_image_surface_create.argtypes = [C.c_int, C.c_int, C.c_int]
        cairo.cairo_image_surface_create.restype = C.c_void_p
        cairo.cairo_create.argtypes = [C.c_void_p]
        cairo.cairo_create.restype = C.c_void_p
        cairo.cairo_scale.argtypes = [C.c_void_p, C.c_double, C.c_double]
        cairo.cairo_surface_write_to_png.argtypes = [C.c_void_p, C.c_char_p]
        cairo.cairo_surface_write_to_png.restype = C.c_int
        cairo.cairo_destroy.argtypes = [C.c_void_p]
        cairo.cairo_surface_destroy.argtypes = [C.c_void_p]
        rsvg.rsvg_handle_new_from_file.argtypes = [C.c_char_p, C.c_void_p]
        rsvg.rsvg_handle_new_from_file.restype = C.c_void_p
        rsvg.rsvg_handle_render_cairo.argtypes = [C.c_void_p, C.c_void_p]
        rsvg.rsvg_handle_render_cairo.restype = C.c_int
        gobject.g_object_unref.argtypes = [C.c_void_p]
        handle = rsvg.rsvg_handle_new_from_file(str(svg_path).encode(), None)
        assert handle, 'librsvg could not open the SVG'
        surface = cairo.cairo_image_surface_create(0, width, height)
        context = cairo.cairo_create(surface)
        cairo.cairo_scale(context, width/1400, height/420)
        assert rsvg.rsvg_handle_render_cairo(handle, context), 'SVG rendering failed'
        status = cairo.cairo_surface_write_to_png(surface, str(png_path).encode())
        cairo.cairo_destroy(context)
        cairo.cairo_surface_destroy(surface)
        gobject.g_object_unref(handle)
        assert status == 0, f'PNG write failed with Cairo status {status}'
    else:
        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), output_width=width, output_height=height)


def main():
    ET.ElementTree(huawei_horizontal()).write(ROOT/'huawei-horizontal.svg', encoding='utf-8', xml_declaration=True)
    banner = element('svg', {'width': 1400, 'height': 420, 'viewBox': '0 0 1400 420', 'role': 'img', 'aria-labelledby': 'title desc'})
    element('title', {'id': 'title'}, banner).text = 'Tsinghua University × Huawei · SignDuplex'
    element('desc', {'id': 'desc'}, banner).text = 'Official Tsinghua emblem and Chinese calligraphy, with a single-line Tsinghua University caption, beside the horizontal Huawei flower and wordmark. SignDuplex is centered between purple and red rules below.'
    element('rect', {'width': 1400, 'height': 420, 'fill': '#ffffff'}, banner)
    insert_svg(banner, 'tsinghua-emblem.svg', 86, 65, 200, 200)
    insert_svg(banner, 'tsinghua-calligraphy.svg', 314, 90, 325, 114.084)
    vector_text(banner, 'TSINGHUA UNIVERSITY', 478, 244, 25.7, PURPLE, weight='bold', centered=True, target_width=330.721353515625)
    vector_text(banner, '×', 718, 189, 64, '#b5b5b5', centered=True)
    insert_svg(banner, 'huawei-horizontal.svg', 811, 102, 512, 150)
    element('path', {'d': 'M86 346 H514', 'fill': 'none', 'stroke': PURPLE, 'stroke-width': 2}, banner)
    element('path', {'d': 'M886 346 H1323', 'fill': 'none', 'stroke': RED, 'stroke-width': 2}, banner)
    vector_text(banner, 'SignDuplex', 700, 363, 51, '#202020', weight='bold', centered=True, target_width=285.00931640625)
    svg_path = ROOT/'brand-banner.svg'
    ET.ElementTree(banner).write(svg_path, encoding='utf-8', xml_declaration=True)
    render_png(svg_path, ROOT/'brand-banner-3x.png', 4200, 1260)
    print(json.dumps({'svg': str(svg_path), 'png': str(ROOT/'brand-banner-3x.png'), 'viewBox': [0, 0, 1400, 420], 'png_size': [4200, 1260]}))


if __name__ == '__main__':
    main()
