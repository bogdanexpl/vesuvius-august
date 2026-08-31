"""Build a letterform inspection sheet from an ink prediction.

Produces, for one prediction TIFF:
  <out>_overview.png  - whole surface, contrast-stretched (layout / text rows)
  <out>_crops.png     - grid of native-resolution crops from the most ink-dense
                        regions, each ~<mm> across, so letterforms can be judged
                        by eye rather than by summary statistics.

Crop selection is ink-density driven, so it shows the model's BEST case: if
letters exist anywhere, they appear here; if these crops are shapeless, the
prediction has no letterforms.

Usage: letterform_sheet.py <pred.tif> <out_prefix> [--um-per-px 8.64] [--mm 11]
                           [--thr 200] [--grid 3]
"""
import sys, argparse
import numpy as np, tifffile
import PIL.Image as I
I.MAX_IMAGE_PIXELS = None

ap = argparse.ArgumentParser()
ap.add_argument("pred"); ap.add_argument("out")
ap.add_argument("--um-per-px", type=float, default=8.64)
ap.add_argument("--mm", type=float, default=11.0)
ap.add_argument("--thr", type=int, default=200)
ap.add_argument("--grid", type=int, default=3)
a = ap.parse_args()

img = tifffile.imread(a.pred)
if img.ndim == 3: img = img[img.shape[0] // 2]
H, W = img.shape
crop = int(round(a.mm * 1000 / a.um_per_px))          # px for the requested mm
print(f"{a.pred}: {img.shape}, crop {crop}px = {a.mm}mm at {a.um_per_px}um/px")

def stretch(x, lo_p=2, hi_p=99.5):
    v = x[x > 0]
    if v.size < 10: return np.zeros_like(x, np.uint8)
    lo, hi = np.percentile(v, [lo_p, hi_p])
    return np.clip((x.astype(np.float32) - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)

# --- overview
d = max(1, max(H, W) // 1600)
I.fromarray(stretch(img[::d, ::d])).save(f"{a.out}_overview.png")

# --- density map over crop-sized blocks, pick the top-N non-overlapping
ink = (img > a.thr).astype(np.float32)
bs = max(crop // 4, 1)
bh, bw = H // bs, W // bs
if bh < 2 or bw < 2:
    print("image too small for crop grid"); sys.exit(0)
dens = ink[:bh * bs, :bw * bs].reshape(bh, bs, bw, bs).mean(axis=(1, 3))
k = max(1, crop // bs)
# box-sum over kxk blocks = ink density of a candidate crop
cs = np.cumsum(np.cumsum(np.pad(dens, ((1, 0), (1, 0))), 0), 1)
score = cs[k:, k:] - cs[:-k, k:] - cs[k:, :-k] + cs[:-k, :-k]
picks, taken = [], np.zeros_like(score, bool)
n = a.grid * a.grid
for _ in range(n):
    s = np.where(taken, -1, score)
    if s.max() <= 0: break
    i, j = np.unravel_index(np.argmax(s), s.shape)
    picks.append((i * bs, j * bs, float(score[i, j] / (k * k))))
    taken[max(0, i - k):i + k, max(0, j - k):j + k] = True

print(f"selected {len(picks)} crops; ink density in them: " +
      ", ".join(f"{p[2]:.3f}" for p in picks))

pad = 8
sheet = np.zeros((a.grid * (crop + pad), a.grid * (crop + pad)), np.uint8)
for idx, (y, x, _) in enumerate(picks):
    r, c = divmod(idx, a.grid)
    tile = stretch(img[y:y + crop, x:x + crop])
    sheet[r * (crop + pad):r * (crop + pad) + tile.shape[0],
          c * (crop + pad):c * (crop + pad) + tile.shape[1]] = tile
out = I.fromarray(sheet)
if sheet.shape[0] > 2600:
    out = out.resize((2600, int(2600 * sheet.shape[0] / sheet.shape[1]))
                     if sheet.shape[1] >= sheet.shape[0] else
                     (int(2600 * sheet.shape[1] / sheet.shape[0]), 2600), I.LANCZOS)
out.save(f"{a.out}_crops.png")
print(f"wrote {a.out}_overview.png and {a.out}_crops.png")
