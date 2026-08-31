#!/usr/bin/env python3
"""Face-on surface validation: does a segment follow a papyrus sheet, or cut
across the windings?

`vc_grow_seg_from_seed` can return surfaces that slice through many sheets
instead of following one. Nothing errors: the render looks like plausible
fibrous texture, and an ink model run on it returns confident-looking noise.
We lost three probe results to this before catching it (see
docs/vc3d-bug-reports.md #4).

This tool decides the question in seconds from a rendered surface volume, by
measuring the fraction of *dark inter-sheet gaps* in the mid-depth layer.
A true sheet surface is continuous material with a few percent of damage; a
winding cross-section is striped with the empty space between sheets.

    valid sheet          dark-gap fraction  0.013 - 0.017
    winding cross-section                   0.105 - 0.127

roughly an 8x separation, which is why a single threshold works.

Usage
-----
    # a rendered surface volume (zarr from vc_render_tifxyz -> tifs_to_zarr)
    python validate_surface.py surface.zarr

    # a directory of layer TIFFs straight out of vc_render_tifxyz
    python validate_surface.py path/to/layers/

    # batch, quiet, exit non-zero if any segment fails
    python validate_surface.py /data/scroll/surface-volumes/*/ --quiet

    # tune for a scroll with unusual damage, and write a contact sheet
    python validate_surface.py surface.zarr --threshold 0.07 --png check.png

Exit code is 0 if every input passes, 1 otherwise, so it drops into a
pipeline before the expensive inference step:

    python validate_surface.py "$OUT/vol.zarr" --quiet || { echo skip; continue; }

Input formats: OME-Zarr / plain zarr (any of `<name>.zarr`, `<name>.zarr/0`),
or a directory of numbered layer TIFFs. Output: one line per segment on
stdout, plus optional PNG contact sheet. Nothing is written to the input.
"""
import argparse, glob, os, sys

import numpy as np


def load_mid_layer(path):
    """Return the mid-depth layer of a rendered surface volume."""
    if os.path.isdir(path) and not path.rstrip("/").endswith(".zarr"):
        tifs = sorted(glob.glob(os.path.join(path, "*.tif")))
        if not tifs:
            raise ValueError(f"no .tif layers in {path}")
        import tifffile
        return tifffile.imread(tifs[len(tifs) // 2]), len(tifs)
    import zarr
    z = zarr.open(path, mode="r")
    if not hasattr(z, "shape"):          # a group: take the first resolution
        z = z["0"]
    return np.asarray(z[z.shape[0] // 2]), z.shape[0]


def dark_gap_fraction(layer, dark_below=40):
    """Fraction of *rendered* pixels that are inter-sheet void.

    Restricted to the rendered region (value > 0): pixels outside the
    segment's footprint are not evidence either way.
    """
    v = layer[layer > 0]
    if v.size == 0:
        return 1.0, 0
    return float((v < dark_below).mean()), int(v.size)


def main():
    ap = argparse.ArgumentParser(
        description="Check whether rendered segments follow a sheet (face-on) "
                    "or cut across windings.")
    ap.add_argument("paths", nargs="+",
                    help="surface volume .zarr, or a directory of layer TIFFs")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="fail above this dark-gap fraction (default 0.05; "
                         "valid sheets measure ~0.015, cross-sections ~0.115)")
    ap.add_argument("--dark-below", type=int, default=40,
                    help="intensity below which a rendered pixel counts as "
                         "void (default 40, for 8-bit renders)")
    ap.add_argument("--png", metavar="OUT",
                    help="write a contact sheet of the mid layers for eyeballing")
    ap.add_argument("--quiet", action="store_true", help="only report failures")
    args = ap.parse_args()

    tiles, failed = [], 0
    for path in args.paths:
        # a bare "layers"/"0" tells the user nothing in batch mode; name the segment
        parts = [p for p in path.rstrip("/").split(os.sep) if p]
        name = next((p for p in reversed(parts)
                     if p not in ("layers", "0") and not p.startswith(".")), path)
        try:
            layer, depth = load_mid_layer(path)
        except Exception as exc:                      # unreadable is a failure
            print(f"FAIL  {name}: unreadable ({exc})")
            failed += 1
            continue
        gap, n = dark_gap_fraction(layer, args.dark_below)
        ok = gap <= args.threshold
        failed += 0 if ok else 1
        if not ok or not args.quiet:
            verdict = "PASS " if ok else "FAIL "
            note = "" if ok else "  <- looks like a winding cross-section"
            print(f"{verdict} {name}: dark-gap {gap:.3f}  "
                  f"({depth} layers, {n/1e6:.1f} MP rendered){note}")
        if args.png:
            tiles.append((name, layer, gap, ok))

    if args.png and tiles:
        write_contact_sheet(tiles, args.png)
        print(f"wrote {args.png}")

    return 1 if failed else 0


def write_contact_sheet(tiles, out, cell=760):
    import PIL.Image as I
    from PIL import ImageDraw
    I.MAX_IMAGE_PIXELS = None
    cols = min(3, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    sheet = I.new("L", (cols * cell + 8, rows * (cell + 22) + 8), 20)
    draw = ImageDraw.Draw(sheet)
    for i, (name, layer, gap, ok) in enumerate(tiles):
        im = I.fromarray(layer.astype(np.uint8)).resize((cell, cell), I.LANCZOS)
        x, y = (i % cols) * cell + 4, (i // cols) * (cell + 22) + 22
        sheet.paste(im, (x, y))
        draw.text((x + 2, y - 16),
                  f"{'PASS' if ok else 'FAIL'}  {name[:44]}  gap {gap:.3f}", fill=255)
    sheet.save(out)


if __name__ == "__main__":
    sys.exit(main())
