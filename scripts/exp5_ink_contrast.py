"""Exp 5: physical ink contrast per scroll — does ink differ in X-ray density?

Herculaneum ink is usually carbon-based (nearly X-ray transparent), but lead and
other metals have been reported in some Herculaneum papyri (Brun et al., PNAS
2016). Metal-bearing ink is directly X-ray dense; carbon-only ink is not, and is
recoverable only from subtle morphology. If our transfer clusters track ink
recipe, the labeled-ink voxels of clustered scrolls should share a contrast
signature.

For each fragment: compare intensity distributions of ink-labeled vs
background voxels (both inside the supervision mask), through the surface
volume's depth. Reports per-layer contrast and the peak separation.
"""
import glob, os
import numpy as np, zarr

ROOT = "/data/vesuvius/ink-dataset/frags"

def arr(p):
    z = zarr.open(p, mode="r")
    return z if isinstance(z, zarr.Array) else z["0"]

print(f"{'fragment':20s} {'best_z':>6s} {'ink_mean':>9s} {'bg_mean':>8s} "
      f"{'delta':>7s} {'cohen_d':>8s} {'ink_n':>9s}")
rows = []
for d in sorted(glob.glob(f"{ROOT}/*")):
    name = os.path.basename(d)
    seg = f"{d}/{name}"
    try:
        vol = arr(f"{seg}/{name}.zarr")
        lab = np.asarray(arr(f"{seg}/{name}_inklabels.zarr")[32]) > 127
        msk = np.asarray(arr(f"{seg}/{name}_supervision_mask.zarr")[32]) > 127
    except Exception as e:
        print(f"{name:20s} SKIP ({e})"); continue

    # subsample spatially for speed; keep class balance honest
    ss = max(1, min(lab.shape) // 1500)
    lab_s, msk_s = lab[::ss, ::ss], msk[::ss, ::ss]
    ink_px = lab_s & msk_s
    bg_px = (~lab_s) & msk_s
    if ink_px.sum() < 500 or bg_px.sum() < 500:
        print(f"{name:20s} SKIP (too few labeled px)"); continue

    best = None
    for z in range(0, 65, 4):
        layer = np.asarray(vol[z, ::ss, ::ss]).astype(np.float32)
        a, b = layer[ink_px], layer[bg_px]
        a, b = a[a > 0], b[b > 0]
        if a.size < 500 or b.size < 500: continue
        pooled = np.sqrt((a.var() + b.var()) / 2) + 1e-6
        dcoh = (a.mean() - b.mean()) / pooled
        if best is None or abs(dcoh) > abs(best[4]):
            best = (z, float(a.mean()), float(b.mean()), float(a.mean() - b.mean()), float(dcoh), int(a.size))
    if best is None:
        print(f"{name:20s} SKIP (no valid layer)"); continue
    z, am, bm, delta, dcoh, n = best
    print(f"{name:20s} {z:6d} {am:9.2f} {bm:8.2f} {delta:+7.2f} {dcoh:+8.3f} {n:9d}")
    rows.append((name, dcoh, delta))

print()
print("Reading: cohen_d > 0 => ink voxels are DENSER (brighter) than background,")
print("consistent with metal-bearing ink. |d| < 0.1 => no bulk density contrast;")
print("ink is carbon-like and only recoverable from morphology/texture.")
