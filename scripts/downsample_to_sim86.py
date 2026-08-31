"""Simulate 8.6µm acquisition from the 2.4µm Paris-4 ink dataset.

Downsamples volume + inklabels + supervision mask by 8.64/2.4 ≈ 3.6 in x/y and
65→18 in z (trilinear for volume, nearest for labels/mask), producing a koine
dataset for the "is ink readable at 8.6µm at all?" fine-tune experiment.
"""
import json, shutil
from pathlib import Path
import numpy as np, torch, zarr

SRC = Path('/data/vesuvius/ink-dataset/phercparis4/w00_20231016151002')
DST = Path('/data/vesuvius/ink-dataset/phercparis4_sim86/w00_20231016151002')
NAME = 'w00_20231016151002'
F = 8.64 / 2.4          # 3.6 in-plane
ZO = 18                  # 65 layers * (2.4/8.64) ≈ 18

def resample(src_path, dst_path, mode):
    src = zarr.open(str(src_path), mode='r')['0']
    zi, H, W = src.shape
    ho, wo = round(H / F), round(W / F)
    dst = zarr.open(str(dst_path), mode='w', shape=(ZO, ho, wo), chunks=(ZO, 128, 128), dtype='u1')
    slab = 256                       # output rows per step
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    for y0 in range(0, ho, slab):
        y1 = min(y0 + slab, ho)
        sy0 = int(np.floor(y0 * H / ho)); sy1 = int(np.ceil(y1 * H / ho))
        block = np.asarray(src[:, sy0:sy1, :])           # (65, sh, W)
        t = torch.from_numpy(block)[None, None].float().to(dev)
        out = torch.nn.functional.interpolate(
            t, size=(ZO, y1 - y0, wo),
            mode='trilinear' if mode == 'trilinear' else 'nearest-exact',
            **({'align_corners': False} if mode == 'trilinear' else {}))
        dst[:, y0:y1, :] = out[0, 0].round().clamp(0, 255).byte().cpu().numpy()
        if (y0 // slab) % 8 == 0:
            print(f'{dst_path.name}: {y0}/{ho}', flush=True)
    print('done', dst_path.name, (ZO, ho, wo), flush=True)

DST.mkdir(parents=True, exist_ok=True)
resample(SRC / f'{NAME}.zarr', DST / f'{NAME}.zarr', 'trilinear')
resample(SRC / f'{NAME}_inklabels.zarr', DST / f'{NAME}_inklabels.zarr', 'nearest')
resample(SRC / f'{NAME}_supervision_mask.zarr', DST / f'{NAME}_supervision_mask.zarr', 'nearest')

meta = json.load(open(SRC / 'meta.json'))
meta['scale'] = [s * F for s in meta.get('scale', [1, 1])]
meta['sim_note'] = 'downsampled 3.6x xy, 65->18 z, simulating 8.64um from 2.4um'
json.dump(meta, open(DST / 'meta.json', 'w'), indent=1)
print('ALL DONE')
