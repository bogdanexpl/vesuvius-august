"""Exp 7 scoring: response-vs-offset curves for the 1447 surface-placement sweep.

Windows are 65-layer slices [k, k+65) of an n=193 slice-step-0.375 render;
window-centre offset from the traced surface = (k+32-96) layers x 3.24 um.
Outputs a table + JSON; letterform sheets are made separately for the
max-response window of each segment (scripts/letterform_sheet.py).
"""
import glob, json, os, re, sys
import numpy as np, tifffile

R = "/data/vesuvius/runs/exp7_surface"
rows = {}
for p in sorted(glob.glob(f"{R}/pred_*.tif")):
    m = re.match(r"pred_(.+?)_(k(\d+)|center)\.tif", os.path.basename(p))
    if not m: continue
    seg, k = m.group(1), (int(m.group(3)) if m.group(3) else None)
    a = tifffile.imread(p)
    v = a[a > 0]  # rendered-region only (0 = outside mask)
    off = None if k is None else (k + 32 - 96) * 3.24
    rows.setdefault(seg, []).append({
        "k": k, "offset_um": off,
        "frac_gt200": float((v > 200).mean()) if v.size else 0.0,
        "frac_gt128": float((v > 128).mean()) if v.size else 0.0,
        "std": float(v.std()) if v.size else 0.0,
        "valid_px": int(v.size)})

for seg, rs in rows.items():
    rs.sort(key=lambda r: (r["k"] is None, r["k"]))
    print(f"\n== {seg}")
    print(f"{'k':>6} {'offset_um':>10} {'frac>200':>9} {'frac>128':>9} {'std':>7}")
    for r in rs:
        kk = "center" if r["k"] is None else str(r["k"])
        oo = "0" if r["offset_um"] is None else f"{r['offset_um']:+.0f}"
        print(f"{kk:>6} {oo:>10} {r['frac_gt200']:9.4f} {r['frac_gt128']:9.4f} {r['std']:7.1f}")
    if len(rs) > 1:
        best = max(rs, key=lambda r: r["frac_gt200"])
        print(f"   peak frac>200 at k={best['k']} ({best['offset_um']:+.0f} um): {best['frac_gt200']:.4f}")

json.dump(rows, open(f"{R}/exp7_scores.json", "w"), indent=1)
print(f"\nwrote {R}/exp7_scores.json")
