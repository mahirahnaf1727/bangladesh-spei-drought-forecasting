"""
check_figs.py  --  checks the exported files in FIG_OUT.

For every FigXX.pdf: page size in mm (width must be 88, 120 or 180; height <= 225)
and whether every font is embedded (and which fonts). For every FigXX.tif:
pixel size and dpi (must be 600) and compression (must be LZW). For every FigXX.png:
pixel size and dpi (must be 600).

Needs: pypdf and Pillow  (pip install pypdf pillow)
"""

import sys

import p1_fig_config as C

WIDTHS = (88, 120, 180)


def check_pdf(path):
    from pypdf import PdfReader
    r = PdfReader(str(path))
    page = r.pages[0]
    w = float(page.mediabox.width) / 72 * 25.4
    h = float(page.mediabox.height) / 72 * 25.4
    fonts, not_embedded = set(), set()

    def walk(res):
        if res is None:
            return
        res = res.get_object()
        for f in (res.get("/Font") or {}).values():
            f = f.get_object()
            name = str(f.get("/BaseFont", "?")).lstrip("/")
            fonts.add(name.split("+")[-1])
            desc = f.get("/FontDescriptor")
            if desc is None and "/DescendantFonts" in f:
                desc = f["/DescendantFonts"][0].get_object().get("/FontDescriptor")
            desc = desc.get_object() if desc is not None else {}
            if not any(k in desc for k in ("/FontFile", "/FontFile2", "/FontFile3")):
                if f.get("/Subtype") != "/Type3":
                    not_embedded.add(name)
        for xo in (res.get("/XObject") or {}).values():
            walk(xo.get_object().get("/Resources"))

    walk(page.get("/Resources"))
    ok = any(abs(w - x) < 0.6 for x in WIDTHS) and h <= 225.5 and not not_embedded
    return ok, f"{w:.0f} x {h:.0f} mm | fonts: {', '.join(sorted(fonts)) or 'none'}" + \
        (f" | NOT EMBEDDED: {', '.join(sorted(not_embedded))}" if not_embedded else "")


def check_tif(path):
    from PIL import Image
    with Image.open(path) as im:
        dpi = tuple(float(v) for v in im.info.get("dpi", (0, 0)))
        comp = im.info.get("compression", "?")
        ok = round(dpi[0]) == 600 and comp == "tiff_lzw"
        return ok, f"{im.size[0]} x {im.size[1]} px | {dpi[0]:.0f} dpi | {comp}"


def check_png(path):
    from PIL import Image
    with Image.open(path) as im:
        dpi = tuple(float(v) for v in im.info.get("dpi", (0, 0)))
        ok = round(dpi[0]) == 600
        return ok, f"{im.size[0]} x {im.size[1]} px | {dpi[0]:.0f} dpi"


def main():
    try:
        import pypdf  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError:
        sys.exit("Install the checkers first:  pip install pypdf pillow")
    files = (sorted(C.FIG_OUT.glob("Fig*.pdf")) + sorted(C.FIG_OUT.glob("Fig*.tif"))
             + sorted(C.FIG_OUT.glob("Fig*.png")))
    if not files:
        sys.exit(f"No figures in {C.FIG_OUT}; run make_all.py first.")
    lines = ["# Figure file check\n", "| File | OK | Details |", "|---|---|---|"]
    bad = 0
    for f in files:
        ok, msg = {".pdf": check_pdf, ".tif": check_tif, ".png": check_png}[f.suffix](f)
        bad += not ok
        lines.append(f"| {f.name} | {'yes' if ok else 'NO'} | {msg} |")
        print(f"  {'OK ' if ok else 'BAD'} {f.name:<14} {msg}")
    (C.FIG_OUT / "figures_file_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{len(files) - bad} of {len(files)} files OK. Report: {C.FIG_OUT / 'figures_file_check.md'}")


if __name__ == "__main__":
    main()
