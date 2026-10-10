"""Extract the institution-only vector header without altering the README head.

Run: python docs/assets/build_institutions_banner.py
Source: adjacent brand-banner.svg; no fonts or source PDFs are needed.
Rendering uses CairoSVG if available, otherwise system librsvg/Cairo.
"""
from pathlib import Path
import copy
import ctypes as C
import ctypes.util
import hashlib
import json
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", NS)
WIDTH, HEIGHT = 1300, 260
CROP_X, CROP_Y = 55, 36


def render_png(svg_path, png_path, width=WIDTH * 3, height=HEIGHT * 3):
    try:
        import cairosvg
    except ImportError:
        cairo = C.CDLL(ctypes.util.find_library("cairo"))
        rsvg = C.CDLL(ctypes.util.find_library("rsvg-2"))
        gobject = C.CDLL(ctypes.util.find_library("gobject-2.0"))
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
        if not handle:
            raise RuntimeError("Cannot load the institution SVG")
        surface = cairo.cairo_image_surface_create(0, width, height)
        context = cairo.cairo_create(surface)
        cairo.cairo_scale(context, width / WIDTH, height / HEIGHT)
        rendered = rsvg.rsvg_handle_render_cairo(handle, context)
        status = cairo.cairo_surface_write_to_png(surface, str(png_path).encode())
        cairo.cairo_destroy(context)
        cairo.cairo_surface_destroy(surface)
        gobject.g_object_unref(handle)
        if not rendered or status:
            raise RuntimeError("Institution PNG rendering failed")
    else:
        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), output_width=width, output_height=height)


def main():
    source_path = ROOT / "brand-banner.svg"
    source_bytes = source_path.read_bytes()
    source = ET.fromstring(source_bytes)
    # Existing header order: metadata, background, five institution elements,
    # then the two bottom rules and the SignDuplex project label.
    children = list(source)
    kept = children[3:8]
    assert [node.tag.rsplit("}", 1)[-1] for node in kept] == ["svg", "svg", "path", "path", "svg"]
    assert children[8].get("d") == "M86 346 H514"
    banner = ET.Element("{" + NS + "}svg", {
        "width": str(WIDTH), "height": str(HEIGHT),
        "viewBox": f"0 0 {WIDTH} {HEIGHT}",
        "role": "img", "aria-labelledby": "institutions-title institutions-desc",
    })
    ET.SubElement(banner, "{" + NS + "}title", {"id": "institutions-title"}).text = "Tsinghua University × Huawei"
    ET.SubElement(banner, "{" + NS + "}desc", {"id": "institutions-desc"}).text = (
        "Official purple Tsinghua emblem and Chinese calligraphy with its English name, "
        "a gray multiplication sign, and Huawei's red flower and horizontal black wordmark."
    )
    ET.SubElement(banner, "{" + NS + "}rect", {"width": str(WIDTH), "height": str(HEIGHT), "fill": "#ffffff"})
    group = ET.SubElement(banner, "{" + NS + "}g", {"transform": f"translate(-{CROP_X} -{CROP_Y})"})
    for node in kept:
        group.append(copy.deepcopy(node))
    # Verify that every retained element remains byte-identical after parsing.
    assert [ET.tostring(node) for node in kept] == [ET.tostring(node) for node in group]
    assert not [node for node in banner.iter() if node.tag.rsplit("}", 1)[-1] in {"image", "text"}]
    output = ROOT / "institutions-banner.svg"
    ET.ElementTree(banner).write(output, encoding="utf-8", xml_declaration=True)
    render_png(output, ROOT / "institutions-banner.png")
    provenance = {
        "source": "brand-banner.svg",
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_provenance": "brand-provenance.json",
        "crop_in_source_coordinates": [CROP_X, CROP_Y, WIDTH, HEIGHT],
        "viewBox": [0, 0, WIDTH, HEIGHT],
        "png_dimensions": [WIDTH * 3, HEIGHT * 3],
        "derivation": "Copied the five upper institution vector elements exactly, removed the bottom project name and rules, and translated the crop. No path, color, aspect ratio or font changes.",
        "external_images": 0,
        "visible_text_nodes": 0,
        "original_header_unchanged": source_path.read_bytes() == source_bytes,
        "rebuild": "python docs/assets/build_institutions_banner.py",
    }
    (ROOT / "institutions-provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
    print(json.dumps(provenance))


if __name__ == "__main__":
    main()
