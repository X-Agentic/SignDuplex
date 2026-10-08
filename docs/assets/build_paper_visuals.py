"""Regenerate large-type paper figures for the SignDuplex research README.

The display SVGs contain outlined type for reliable rendering. Matching
*.editable.svg sources retain native text. Every PNG is rendered at 3x scale.
Run: python docs/assets/build_paper_visuals.py
Requires matplotlib, Pillow, and CairoSVG. In this workspace CairoSVG is also
available in /tmp/signduplex-visual-tools.
"""

from __future__ import annotations

import html
import json
import shutil
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

try:
    import cairosvg
except ModuleNotFoundError:
    sys.path.insert(0, "/tmp/signduplex-visual-tools")
    import cairosvg

from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath
from PIL import Image


ASSETS = Path(__file__).resolve().parent
REVIEW = Path("/mnt/ljs/output/SignDuplex_Showcase_20261005/figures")
WIDTH = 1040
INK = "#24182e"
MUTED = "#6e6378"
PURPLE = "#702082"
TEAL = "#226778"
RED = "#c62642"
LINE = "#dbd3e2"
BASELINE = "#b9adbf"
FONT = "DejaVu Sans"

ROWS = [
    ("Offline", 8.20, 15.11),
    ("SL segmentation", 11.85, 6.48),
    ("Online CSLR/SLT", 10.90, 6.26),
    ("SignDuplex", 13.72, 5.99),
    ("Ideal boundary reference", 14.38, 5.91),
]


class Figure:
    def __init__(self, name, height, title, description):
        self.name = name
        self.height = height
        self.title = title
        self.description = description
        self.parts = []
        self.labels = []
        self.data = {}

    def shape(self, markup):
        self.parts.append(("shape", markup))

    def rect(self, x, y, width, height, fill="white", stroke="none", radius=18,
             stroke_width=1.5, dash=None):
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.shape(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'rx="{radius}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{stroke_width}"{dash_attr}/>'
        )

    def path(self, d, color=LINE, width=2.5, dash=None):
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.shape(
            f'<path d="{d}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-linecap="round" '
            f'stroke-linejoin="round"{dash_attr}/>'
        )

    def arrow(self, x, y1, y2, color=LINE, width=2.5):
        self.path(f"M{x} {y1} V{y2 - 8}", color, width)
        self.shape(f'<path d="M{x - 6} {y2 - 9} L{x} {y2} '
                   f'L{x + 6} {y2 - 9} Z" fill="{color}"/>')

    def text(self, x, y, value, size=28, color=INK, bold=False, anchor="start"):
        # Record every visible label and its size for an inspectable readability
        # audit, including labels that are paths in the final display SVG.
        prop = FontProperties(family=FONT, size=size,
                              weight="bold" if bold else "normal")
        path = TextPath((0, 0), value, prop=prop)
        bounds = path.get_extents()
        if anchor == "middle":
            tx = x - (bounds.x0 + bounds.x1) / 2
        elif anchor == "end":
            tx = x - bounds.x1
        else:
            tx = x - bounds.x0
        bbox = [tx + bounds.x0, y - bounds.y1,
                tx + bounds.x1, y - bounds.y0]
        self.labels.append({"label": value, "font_size": size,
                            "bounds": [round(v, 3) for v in bbox]})
        self.parts.append(("text", {
            "x": x, "y": y, "value": value, "size": size,
            "color": color, "bold": bold, "anchor": anchor,
            "path": path, "tx": tx,
        }))

    def svg(self, outlined=True):
        body = []
        for kind, item in self.parts:
            if kind == "shape":
                body.append(item)
                continue
            label = html.escape(item["value"], quote=True)
            if outlined:
                commands = []
                for vertices, code in item["path"].iter_segments(
                        curves=True, simplify=False):
                    if code == MplPath.MOVETO:
                        command = "M"
                    elif code == MplPath.LINETO:
                        command = "L"
                    elif code == MplPath.CURVE3:
                        command = "Q"
                    elif code == MplPath.CURVE4:
                        command = "C"
                    elif code == MplPath.CLOSEPOLY:
                        commands.append("Z")
                        continue
                    else:
                        raise ValueError(f"Unsupported path code: {code}")
                    commands.append(command + " ".join(f"{v:.3f}" for v in vertices))
                body.append(
                    f'<g role="img" aria-label="{label}" '
                    f'data-label="{label}" data-font-size="{item["size"]}" '
                    f'transform="translate({item["tx"]:.3f} {item["y"]}) '
                    f'scale(1 -1)"><title>{label}</title>'
                    f'<path fill="{item["color"]}" d="{" ".join(commands)}"/></g>'
                )
            else:
                body.append(
                    f'<text x="{item["x"]}" y="{item["y"]}" '
                    f'fill="{item["color"]}" font-size="{item["size"]}" '
                    f'font-weight="{700 if item["bold"] else 400}" '
                    f'text-anchor="{item["anchor"]}">{label}</text>'
                )
        metadata = html.escape(json.dumps({
            "source": "SignDuplex final paper, 20260924065321_895159_10552.pdf",
            "text_rendering": "outlined" if outlined else "editable native SVG text",
            "figure_data": self.data,
            "visible_labels": [{k: v for k, v in item.items() if k != "bounds"}
                               for item in self.labels],
        }, ensure_ascii=False))
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
            f'height="{self.height}" viewBox="0 0 {WIDTH} {self.height}" '
            f'role="img" aria-labelledby="title description">'
            f'<title id="title">{html.escape(self.title)}</title>'
            f'<desc id="description">{html.escape(self.description)}</desc>'
            f'<metadata>{metadata}</metadata>'
            f'<g font-family="{FONT}, sans-serif">{"".join(body)}</g></svg>\n'
        )

    def save(self):
        outlined = self.svg(True)
        editable = self.svg(False)
        (ASSETS / f"{self.name}.svg").write_text(outlined)
        (ASSETS / f"{self.name}.editable.svg").write_text(editable)
        cairosvg.svg2png(bytestring=outlined.encode(),
                        output_width=WIDTH * 3,
                        output_height=self.height * 3,
                        write_to=str(ASSETS / f"{self.name}.png"))
        REVIEW.mkdir(parents=True, exist_ok=True)
        for extension in ("svg", "editable.svg", "png"):
            shutil.copy2(ASSETS / f"{self.name}.{extension}",
                         REVIEW / f"{self.name}.{extension}")
        root = ET.fromstring(outlined)
        tags = [node.tag.rsplit("}", 1)[-1] for node in root.iter()]
        assert not set(tags).intersection({"text", "script", "foreignObject", "image"})
        assert all("href" not in attr for node in root.iter() for attr in node.attrib)
        assert len(root.findall("{http://www.w3.org/2000/svg}title")) == 1
        assert min(label["font_size"] for label in self.labels) >= 24
        outside = [item for item in self.labels
                   if item["bounds"][0] < 0 or item["bounds"][1] < 0
                   or item["bounds"][2] > WIDTH
                   or item["bounds"][3] > self.height]
        assert not outside, outside
        with Image.open(ASSETS / f"{self.name}.png") as image:
            assert image.size == (WIDTH * 3, self.height * 3)
            image.verify()
        return {
            "file": f"{self.name}.svg",
            "viewbox": [0, 0, WIDTH, self.height],
            "png_size": [WIDTH * 3, self.height * 3],
            "display_type": "outlined paths with accessible title and label metadata",
            "editable_source": f"{self.name}.editable.svg",
            "minimum_type_size": min(item["font_size"] for item in self.labels),
            "external_references": 0,
            "out_of_bounds_labels": outside,
            "labels": self.labels,
            "data": self.data,
        }


def method_overview():
    figure = Figure(
        "method-overview", 1230,
        "SignDuplex proposed method: causal dual RGB–Pose event prediction",
        "Only observed RGB frames are used. The RGB pathway applies a SigLIP "
        "vision encoder and visual resampler. The parallel pose pathway applies "
        "RTMPose, an ST-GCN encoder, a Temporal Adapter, and a Projector. "
        "The current receiving or responding state is embedded. RGB, pose, and "
        "state tokens are concatenated and modeled by Qwen3-0.6B with LoRA. "
        "A unified five-class head predicts REJECT, INCOMPLETE, WAIT, "
        "COMPLETE, or BARGE. Based on final paper Fig. 3 and Section 3.2.",
    )
    figure.rect(0, 0, WIDTH, figure.height, "#fff", radius=0)
    figure.text(48, 65, "PROPOSED METHOD", 24, PURPLE, True)
    figure.text(48, 126, "Dual RGB–Pose event prediction", 42, bold=True)
    figure.text(48, 175, "Causal visual input + current system state", 28, MUTED)

    figure.rect(48, 214, 616, 88, "#f2ecf7", radius=16)
    figure.text(356, 250, "Causal video prefix", 32, PURPLE, True, "middle")
    figure.text(356, 284, "Frames available at the current update", 25, MUTED,
                anchor="middle")
    figure.rect(700, 214, 292, 88, "#fff1ed", radius=16)
    figure.text(846, 250, "System state", 30, RED, True, "middle")
    figure.text(846, 284, "Receiving / Responding", 24, MUTED, anchor="middle")

    figure.path("M356 302 V326 H194 V341", PURPLE, 3)
    figure.path("M356 326 H520 V341", TEAL, 3)
    figure.arrow(194, 341, 358, PURPLE, 3)
    figure.arrow(520, 341, 358, TEAL, 3)
    figure.arrow(846, 302, 358, RED, 3)

    figure.rect(48, 358, 292, 342, "#fbf8fd", "#e6d9ed", 16)
    figure.rect(374, 358, 292, 342, "#f5fafb", "#d5e5e8", 16)
    figure.rect(700, 358, 292, 342, "#fff9f6", "#eeded8", 16)
    figure.text(194, 399, "RGB pathway", 29, PURPLE, True, "middle")
    figure.text(520, 399, "Pose pathway", 29, TEAL, True, "middle")
    figure.text(846, 399, "State pathway", 29, RED, True, "middle")
    for x in (72, 398, 724):
        figure.path(f"M{x} 418 H{x + 244}", LINE, 1.5)

    figure.text(194, 463, "SigLIP", 32, bold=True, anchor="middle")
    figure.text(194, 496, "Vision encoder", 27, anchor="middle")
    figure.arrow(194, 510, 536, PURPLE)
    figure.text(194, 572, "Visual resampler", 28, anchor="middle")
    figure.arrow(194, 590, 644, PURPLE)

    for y, label, size in ((455, "RTMPose", 30),
                            (516, "ST-GCN encoder", 28),
                            (577, "Temporal Adapter", 27),
                            (638, "Projector", 28)):
        figure.text(520, y, label, size, anchor="middle")
    for top in (467, 528, 589):
        figure.arrow(520, top, top + 22, TEAL)

    figure.text(846, 538, "State embedding", 27, anchor="middle")
    figure.arrow(846, 558, 644, RED)
    for x, label, color, background in (
            (48, "RGB tokens", PURPLE, "#eadcf2"),
            (374, "Pose tokens", TEAL, "#e0eef1"),
            (700, "State token", RED, "#f7e4dc")):
        figure.rect(x + 14, 659, 264, 41, background, radius=10)
        figure.text(x + 146, 688, label, 27, color, True, "middle")

    figure.path("M194 700 V733 H520", PURPLE, 3)
    figure.path("M846 700 V733 H520", RED, 3)
    figure.path("M520 700 V733", TEAL, 3)
    figure.arrow(520, 733, 757, PURPLE, 3)
    figure.rect(222, 757, 596, 68, "#f3ecf7", radius=16)
    figure.text(520, 800, "Token concatenation", 32, PURPLE, True, "middle")

    figure.arrow(520, 825, 866, PURPLE, 3)
    figure.rect(222, 866, 596, 92, PURPLE, radius=18)
    figure.text(520, 907, "Qwen3-0.6B (LoRA)", 34, "white", True, "middle")
    figure.text(520, 943, "Joint RGB + Pose + State modeling", 27, "#f2e9f7",
                anchor="middle")

    figure.arrow(520, 958, 1000, PURPLE, 3)
    figure.rect(222, 1000, 596, 70, "#fff", "#ccb6d8", 16, 2)
    figure.text(520, 1046, "Unified five-class head", 32, bold=True,
                anchor="middle")
    figure.arrow(520, 1070, 1110, PURPLE, 3)

    events = [
        (48, 158, "REJECT", "#f0eaf4", PURPLE),
        (220, 218, "INCOMPLETE", "#f0eaf4", PURPLE),
        (452, 138, "WAIT", "#f0eaf4", PURPLE),
        (604, 214, "COMPLETE", "#e8f3ee", "#27664d"),
        (832, 160, "BARGE", "#fce9ee", RED),
    ]
    for x, width, label, background, color in events:
        figure.rect(x, 1122, width, 54, background, radius=12)
        figure.text(x + width / 2, 1158, label, 25, color, True, "middle")
    figure.text(48, 1210, "Paper Fig. 3 · streaming update every 1 s", 24, MUTED)
    figure.data = {
        "source": "Figure 3 / Section 3.2",
        "update_interval_seconds": 1,
        "backbone": "Qwen3-0.6B",
        "adaptation": "LoRA",
        "event_classes": [event[2] for event in events],
        "input_policy": "Only video frames observed so far",
    }
    return figure


def results_chart(kind):
    quality = kind == "quality"
    figure = Figure(
        f"results-{kind}", 770,
        "SignDuplex translation quality on continuous multi-sentence input"
        if quality else "SignDuplex first segment submission time on continuous input",
        "Final paper Table 4, continuous multi-sentence input. All strategies use "
        "the same Uni-Sign translation model. "
        + ("BLEU-4 values: Offline 8.20, SL segmentation 11.85, Online CSLR/SLT "
           "10.90, SignDuplex 13.72, ideal boundary reference 14.38. Higher is better. "
           "SignDuplex improves BLEU-4 by 67.3 percent relative to Offline."
           if quality else
           "Mean first submission times in seconds: Offline 15.11, SL segmentation "
           "6.48, Online CSLR/SLT 6.26, SignDuplex 5.99, ideal boundary reference "
           "5.91. Lower is better. Time runs from first video frame until the first "
           "valid segment is submitted to the translator. SignDuplex reduces "
           "this time by 60.4 percent relative to Offline.")
        + " The reference uses ground-truth sentence boundaries. Bar axes start at zero.",
    )
    accent = PURPLE if quality else RED
    figure.rect(0, 0, WIDTH, figure.height, "#fff", radius=0)
    figure.text(48, 65, "CONTINUOUS MULTI-SENTENCE INPUT", 24, accent, True)
    figure.text(48, 126, "Translation quality" if quality else "First segment submission",
                42 if quality else 38, bold=True)
    figure.text(992, 125, "+67.3%" if quality else "−60.4%", 36, accent, True, "end")
    figure.text(992, 168, "vs. offline", 25, MUTED, anchor="end")
    figure.text(48, 176, "BLEU-4 ↑ · higher is better" if quality
                else "Tfirst (s) ↓ · lower is faster", 28, MUTED)
    figure.text(48, 213, "Same Uni-Sign translator · paper Table 4" if quality
                else "From first frame to valid segment submission", 25, MUTED)

    origin, track_width, axis_max = 380, 540, 16
    top, row_step, bar_height = 249, 81, 40
    axis_y = 663
    for tick in (0, 4, 8, 12, 16):
        x = origin + tick / axis_max * track_width
        figure.path(f"M{x} {top - 12} V{axis_y}", "#eee8f2", 1.5)

    for index, (name, bleu, seconds) in enumerate(ROWS):
        value = bleu if quality else seconds
        y = top + index * row_step
        is_ours = name == "SignDuplex"
        reference = index == 4
        if is_ours:
            figure.rect(32, y - 16, 976, 72,
                        "#f7eefb" if quality else "#fff0f3", radius=12)
        if reference:
            figure.text(48, y + 24, "Ideal boundary", 30, INK)
            figure.text(48, y + 55, "reference", 30, INK)
        else:
            figure.text(48, y + 30, name, 30,
                        accent if is_ours else INK, is_ours)
        width = value / axis_max * track_width
        if reference:
            figure.rect(origin, y, width, bar_height, "#fff", "#a48aaf", 5,
                        2, "7 5")
        else:
            figure.rect(origin, y, width, bar_height,
                        accent if is_ours else BASELINE, radius=5)
        figure.text(origin + width + 15, y + 31, f"{value:.2f}", 34,
                    accent if is_ours else MUTED, is_ours)
    figure.path(f"M{origin} {axis_y} H{origin + track_width}", "#aa9bb4", 2)
    for tick in (0, 4, 8, 12, 16):
        x = origin + tick / axis_max * track_width
        figure.path(f"M{x} {axis_y} V{axis_y + 7}", "#aa9bb4", 2)
        figure.text(x, axis_y + 38, str(tick), 26, MUTED, anchor="middle")
    figure.text(48, 749, "Reference uses ground-truth sentence boundaries.", 25, MUTED)
    figure.data = {
        "source": "Table 4, continuous multi-sentence input",
        "metric": "BLEU-4" if quality else "Mean first valid segment submission time",
        "unit": "BLEU-4" if quality else "seconds",
        "axis_minimum": 0,
        "axis_maximum": axis_max,
        "values": {name: bleu if quality else seconds for name, bleu, seconds in ROWS},
        "reference_protocol": "Ground-truth sentence boundaries; ideal boundary conditions",
        "relative_change_vs_offline_percent": 67.3 if quality else -60.4,
    }
    return figure


def main():
    figures = [method_overview(), results_chart("quality"), results_chart("latency")]
    validation = {"status": "passed", "renderer": "CairoSVG", "raster_scale": 3,
                  "font": FONT, "figures": [figure.save() for figure in figures]}
    for output in (ASSETS, REVIEW):
        (output / "paper-visuals-validation.json").write_text(
            json.dumps(validation, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "passed", "figures": [figure.name for figure in figures],
                      "assets": str(ASSETS), "review": str(REVIEW)}, indent=2))


if __name__ == "__main__":
    main()
