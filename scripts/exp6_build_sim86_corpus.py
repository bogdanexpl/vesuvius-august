"""Exp 6: degrade the multi-scroll fragment corpus to 8.64um in-plane.

Match to the eligible-scroll renders is done on BOTH axes:
  in-plane: fragments (3.24um, or 2.0um for 343P) -> 8.64um by area-average
  depth:    left at the fragment's native z-spacing (3.24/2.0um), and the
            scroll side is rendered with --slice-step (spacing/8.64) so its 65
            layers span the same physical depth at the same z-sampling.
Result: both domains are xy 8.64um / z ~3.24um, 65 layers, labels at z=32,
so the koine pipeline and patch finder are unchanged.
"""
import os, glob, json, numpy as np, zarr, torch

SRC = "/data/vesuvius/ink-dataset/frags"
DST = "/data/vesuvius/ink-dataset/frags_sim86"
TARGET_UM = 8.64
SRC_UM = {"pherc343p__front": 2.0, "pherc500p2__front": 2.0}   # rest are 3.24
SKIP = {"pherc500p2__front"}   # 27k x 15k -> OOM risk, excluded (documented)

def arr(p):
    z = zarr.open(p, mode="r")
    return z if isinstance(z, zarr.Array) else z["0"]

def downscale_xy(a2d, f):
    t = torch.from_numpy(a2d.astype(np.float32))[None, None]
    h, w = max(1, int(round(a2d.shape[0] / f))), max(1, int(round(a2d.shape[1] / f)))
    return torch.nn.functional.interpolate(t, size=(h, w), mode="area")[0, 0].numpy()

for d in sorted(glob.glob(f"{SRC}/*")):
    name = os.path.basename(d)
    if name in SKIP:
        print(f"{name}: SKIP (memory)"); continue
    out = f"{DST}/{name}/{name}"
    if os.path.exists(f"{out}/{name}_supervision_mask.zarr"):
        print(f"{name}: already built"); continue
    os.makedirs(out, exist_ok=True)
    f = TARGET_UM / SRC_UM.get(name, 3.24)
    vol = arr(f"{SRC}/{name}/{name}/{name}.zarr")
    lab = np.asarray(arr(f"{SRC}/{name}/{name}/{name}_inklabels.zarr")[32])
    msk = np.asarray(arr(f"{SRC}/{name}/{name}/{name}_supervision_mask.zarr")[32])
    Z = vol.shape[0]
    probe = downscale_xy(np.asarray(vol[0]), f)
    H, W = probe.shape
    print(f"{name}: factor {f:.2f}  {vol.shape} -> ({Z},{H},{W})", flush=True)

    g = zarr.open_group(f"{out}/{name}.zarr", mode="w")
    dz = g.create_dataset("0", shape=(Z, H, W), chunks=(Z, 128, 128), dtype="u1")
    for z in range(Z):
        dz[z] = np.clip(downscale_xy(np.asarray(vol[z]), f), 0, 255).astype(np.uint8)
    # labels/mask: nearest so thin strokes survive; placed at the plane the finder samples
    for suffix, plane in [("_inklabels", lab), ("_supervision_mask", msk)]:
        t = torch.from_numpy((plane > 127).astype(np.float32))[None, None]
        small = torch.nn.functional.interpolate(t, size=(H, W), mode="nearest-exact")[0, 0].numpy()
        block = np.zeros((Z, H, W), np.uint8)
        block[Z // 2] = (small > 0.5).astype(np.uint8) * 255
        gg = zarr.open_group(f"{out}/{name}{suffix}.zarr", mode="w")
        gg.create_dataset("0", data=block, chunks=(Z, 128, 128), dtype="u1")
    for n in ("x", "y", "z"):
        src_t = f"{SRC}/{name}/{name}/{n}.tif"
        if os.path.exists(src_t) and not os.path.exists(f"{out}/{n}.tif"):
            os.symlink(src_t, f"{out}/{n}.tif")
    json.dump({"scale": [0.05, 0.05], "type": "seg", "uuid": name, "format": "tifxyz",
               "sim": f"xy downscaled x{f:.2f} to 8.64um"}, open(f"{out}/meta.json", "w"))
    print(f"{name}: BUILT  ink px at plane = {int((small>0.5).sum())}", flush=True)
print("CORPUS DONE")
