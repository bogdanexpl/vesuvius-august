"""Radius-aware seed picking from a local surface-prediction band.

Usage: pick_seeds.py <band_zarr> <z...>  -> prints "x,y,z" seeds, 3 radial
tiers (inner/mid/outer winding) per z slice.
"""
import sys, numpy as np, zarr

a = zarr.open(sys.argv[1], mode="r")
a = a if isinstance(a, zarr.Array) else a["0"]
out = []
for z in [int(v) for v in sys.argv[2:]]:
    s = np.asarray(a[z, ::8, ::8])
    ys, xs = np.nonzero(s > 128)
    if not len(ys): continue
    cy, cx = ys.mean(), xs.mean()
    r = np.hypot(ys - cy, xs - cx)
    for lo, hi in [(0.05, 0.3), (0.4, 0.6), (0.75, 0.95)]:
        rlo, rhi = np.quantile(r, [lo, hi])
        sel = np.nonzero((r >= rlo) & (r <= rhi))[0]
        if not len(sel): continue
        i = sel[len(sel) // 2]
        out.append(f"{xs[i]*8},{ys[i]*8},{z}")
print(" ".join(out))
