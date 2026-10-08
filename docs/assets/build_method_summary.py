"""Build the compact SignDuplex paper-method SVG and its 3x PNG fallback.

Run: python docs/assets/build_method_summary.py
Optionally pass --font /path/to/DejaVuSans.ttf.
Requires matplotlib, CairoSVG, and DejaVu Sans Regular.
Edit LABELS to change native labels;
the exported SVG outlines the font to keep typography consistent in README.
"""

from pathlib import Path
import argparse
import html
import json
import xml.etree.ElementTree as ET

import cairosvg

from matplotlib.font_manager import FontProperties, findfont, get_font
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath


ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT, SCALE = 1000, 200, 3
PURPLE, RED, INK, MUTED = "#702082", "#c62642", "#24182e", "#74687c"
FONT = "DejaVu Sans"
LABELS = [
    (16, 226, "RGB · Pose · State", "Causal input tokens", 23, 19, False),
    (266, 180, "Qwen3-0.6B", "LoRA", 25, 20, False),
    (470, 212, "Five-event head", "Event prediction", 23, 19, False),
    (706, 278, "Interaction controller", "Receive · Respond · Interrupt", 23, 18, False),
]


def outlined_text(x, y, label, size, color, bold=False, font_file=None):
    path = TextPath((0, 0), label, prop=FontProperties(
        fname=font_file, family=FONT, size=size, weight="normal"))
    bounds = path.get_extents()
    tx = x - (bounds.x0 + bounds.x1) / 2
    commands = []
    for values, code in path.iter_segments(curves=True, simplify=False):
        if code == MplPath.CLOSEPOLY:
            commands.append("Z")
            continue
        letter = {MplPath.MOVETO: "M", MplPath.LINETO: "L",
                  MplPath.CURVE3: "Q", MplPath.CURVE4: "C"}[code]
        commands.append(letter + " ".join(f"{value:.3f}" for value in values))
    label = html.escape(label, quote=True)
    markup = (
        f'<g role="img" aria-label="{label}" data-label="{label}" '
        f'data-font-size="{size}" transform="translate({tx:.3f} {y}) scale(1 -1)">'
        f'<title>{label}</title><path fill="{color}" '
        f'd="{" ".join(commands)}"/></g>'
    )
    return markup, [tx + bounds.x0, y - bounds.y1,
                    tx + bounds.x1, y - bounds.y0]


def build(font_file=None):
    font_file = font_file or findfont(
        FontProperties(family=FONT, weight="normal"), fallback_to_default=False)
    font = get_font(font_file)
    if font.family_name != FONT or font.style_name not in ("Book", "Regular"):
        raise ValueError("The supplied font must be DejaVu Sans Regular.")
    body = ['<rect width="1000" height="200" fill="#fff"/>']
    bounds = []
    for index, (x, width, heading, subtitle, size, sub_size, bold) in enumerate(LABELS):
        controller = index == 3
        fill = "#fff5f6" if controller else "#fff"
        stroke = "#e6a7b2" if controller else "#d4bbdf"
        body.append(f'<rect x="{x}" y="38" width="{width}" height="124" '
                    f'rx="14" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
        markup, box = outlined_text(x + width / 2, 92, heading, size,
                                     RED if controller else INK, bold, font_file)
        body.append(markup)
        bounds.append((heading, box, x, x + width))
        markup, box = outlined_text(x + width / 2, 127, subtitle, sub_size,
                                     RED if controller else MUTED, font_file=font_file)
        body.append(markup)
        bounds.append((subtitle, box, x, x + width))
        if index < 3:
            next_x = LABELS[index + 1][0]
            start, end = x + width + 5, next_x - 5
            color = RED if index == 2 else PURPLE
            body.append(f'<path d="M{start} 100 H{end - 5}" fill="none" '
                        f'stroke="{color}" stroke-width="2.2" stroke-linecap="round"/>')
            body.append(f'<path d="M{end - 6} 95 L{end} 100 L{end - 6} 105" '
                        f'fill="none" stroke="{color}" stroke-width="2.2" '
                        'stroke-linecap="round" stroke-linejoin="round"/>')

    description = (
        "Compact overview of the proposed SignDuplex method. Causal RGB, pose, "
        "and current system-state tokens are modeled by Qwen3-0.6B with LoRA. "
        "A unified five-event head predicts interaction events. The interaction "
        "controller routes receiving, responding, and user interruption. "
        "Based on the final paper method; only frames observed so far are used."
    )
    metadata = html.escape(json.dumps({
        "source": "SignDuplex final paper, Figure 3 and Sections 3.1–3.2",
        "labels": [{"heading": item[2], "subtitle": item[3],
                    "heading_font_size": item[4], "subtitle_font_size": item[5]}
                   for item in LABELS],
        "font": "DejaVu Sans Regular, outlined vector paths",
        "raster_scale": SCALE,
    }, ensure_ascii=False))
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        'viewBox="0 0 1000 200" role="img" aria-labelledby="title description">'
        '<title id="title">SignDuplex: compact proposed-method overview</title>'
        f'<desc id="description">{html.escape(description)}</desc>'
        f'<metadata>{metadata}</metadata>{"".join(body)}</svg>\n'
    )
    root = ET.fromstring(svg)
    tags = {node.tag.rsplit("}", 1)[-1] for node in root.iter()}
    assert not tags.intersection({"text", "image", "script", "foreignObject"})
    assert all("href" not in attr for node in root.iter() for attr in node.attrib)
    assert all(item[4] >= 23 for item in LABELS)
    assert not any(item[6] for item in LABELS)
    for label, box, block_left, block_right in bounds:
        assert block_left + 4 < box[0] < box[2] < block_right - 4, (label, box)
        assert 38 < box[1] < box[3] < 162, (label, box)
    svg_path = ROOT / "method-summary.svg"
    png_path = ROOT / "method-summary.png"
    svg_path.write_text(svg)
    cairosvg.svg2png(bytestring=svg.encode(), output_width=WIDTH * SCALE,
                    output_height=HEIGHT * SCALE, write_to=str(png_path))
    print(json.dumps({
        "status": "passed", "svg": str(svg_path), "png": str(png_path),
        "viewbox": [0, 0, WIDTH, HEIGHT], "png_size": [WIDTH * SCALE, HEIGHT * SCALE],
        "font": font.family_name, "font_style": font.style_name,
        "minimum_main_font_size": min(item[4] for item in LABELS),
        "external_references": 0, "labels_in_bounds": True,
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", help="Path to the DejaVu Sans Regular font file")
    build(parser.parse_args().font)
