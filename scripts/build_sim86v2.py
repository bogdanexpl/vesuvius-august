"""sim86v2: acquisition-matched variant of the sim86 dataset.

sim86 (plain 3.6x downsample of 2.4um Paris4) has 2.3x the high-frequency
energy of the real 1447 8.64um/116keV/1.2m acquisition and a darker histogram.
This applies: 3D gaussian blur (0.5, 1.35, 1.35) -> quantile LUT matching
blurred-sim papyrus voxels to real-1447 render voxels. Zeros preserved.
Labels/mask/geometry reused from sim86 via symlinks.
"""
import numpy as np, zarr, os
from pathlib import Path
from scipy.ndimage import gaussian_filter

SRC = Path('/data/vesuvius/ink-dataset/phercparis4_sim86/w00_20231016151002')
DST = Path('/data/vesuvius/ink-dataset/phercparis4_sim86v2/w00_20231016151002')
NAME = 'w00_20231016151002'
REAL = '/data/vesuvius/PHerc1447/surface-volumes/auto_grown_20250703034159599/auto_grown_20250703034159599.zarr'
SIG = (0.5, 1.35, 1.35)

def arr(p, mode='r'):
    z = zarr.open(str(p), mode=mode)
    return z if isinstance(z, zarr.Array) else z['0']

real = arr(REAL)
rv = np.asarray(real[28:37, ::4, ::4]).ravel(); rv = rv[rv > 0]
src = arr(SRC / f'{NAME}.zarr')
Z, H, W = src.shape

# pass 1: blurred-sim quantiles from a sample stripe
stripe = np.asarray(src[:, ::3, ::3]).astype(np.float32)
stripe_b = gaussian_filter(stripe, SIG)
sv = stripe_b[stripe > 0]
qs = np.linspace(0, 100, 512)
sim_q = np.percentile(sv, qs); real_q = np.percentile(rv, qs)
del stripe, stripe_b, sv
print('LUT ready', flush=True)

DST.mkdir(parents=True, exist_ok=True)
g = zarr.open_group(str(DST / f'{NAME}.zarr'), mode='w')
d = g.create_dataset('0', shape=(Z, H, W), chunks=(Z, 128, 128), dtype='u1')

PAD = 8
slab = 1024
for y0 in range(0, H, slab):
    y1 = min(y0 + slab, H)
    ry0, ry1 = max(0, y0 - PAD), min(H, y1 + PAD)
    block = np.asarray(src[:, ry0:ry1, :]).astype(np.float32)
    zero = block == 0
    b = gaussian_filter(block, SIG)
    out = np.interp(b, sim_q, real_q)
    out[zero] = 0
    d[:, y0:y1, :] = np.clip(out[:, y0 - ry0:y1 - ry0, :], 0, 255).astype(np.uint8)
    print(f'{y0}/{H}', flush=True)

for f in [f'{NAME}_inklabels.zarr', f'{NAME}_supervision_mask.zarr', 'x.tif', 'y.tif', 'z.tif', 'meta.json']:
    t = DST / f
    if not t.exists(): os.symlink(SRC / f, t)

# verify
v2 = arr(DST / f'{NAME}.zarr')
p = np.asarray(v2[9, 4000:5000, 6000:7000]).astype(np.float32)
m = p > 0
dd = np.diff(p, axis=1); dm = m[:, 1:] & m[:, :-1]
print('v2 grad-std', round(float(dd[dm].std()), 1), 'mean', round(float(p[m].mean()), 1),
      'std', round(float(p[m].std()), 1), '(target 5.8 / 100.7 / 32.8)')
print('DONE')
