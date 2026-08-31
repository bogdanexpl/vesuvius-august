"""Exp 10: pack the in-scroll Scroll 4 (PHerc 1667) labeled crops into koine
flat datasets (same layout as exp4_fetch_frags.py). Source: bruniss
train_ink_eval (100 x (65,512,512) image/label/supervision TIFF triples)."""
import glob, json, os
import numpy as np, tifffile, zarr

SRC = "/data/vesuvius/ink-dataset/scroll4_inscroll/train_ink_eval"
DST = "/data/vesuvius/ink-dataset/s4crops"
built = 0
for p in sorted(glob.glob(f"{SRC}/images/*.tif")):
    n = os.path.basename(p)[:-4]
    name = "s4_" + n.split("_")[0] + "_" + n.split("_")[1]  # s4_000_w023
    seg = f"{DST}/{name}/{name}"
    if os.path.exists(f"{seg}/{name}_supervision_mask.zarr"):
        built += 1; continue
    os.makedirs(seg, exist_ok=True)
    vol = tifffile.imread(p)
    lab = tifffile.imread(f"{SRC}/labels/{os.path.basename(p)}")
    msk = tifffile.imread(f"{SRC}/supervision_masks/{os.path.basename(p)}")
    zarr.open_group(f"{seg}/{name}.zarr", mode="w").create_dataset(
        "0", data=vol, chunks=(65, 128, 128), dtype="u1")
    for suffix, a in [("_inklabels", lab), ("_supervision_mask", msk)]:
        zarr.open_group(f"{seg}/{name}{suffix}.zarr", mode="w").create_dataset(
            "0", data=a, chunks=(65, 128, 128), dtype="u1")
    dummy = np.ones((26, 26), np.float32)
    for ax in "xyz":
        tifffile.imwrite(f"{seg}/{ax}.tif", dummy)
    json.dump({"scale": [0.05, 0.05], "type": "seg", "uuid": name,
               "format": "tifxyz", "scroll_source": "pherc1667_inscroll"},
              open(f"{seg}/meta.json", "w"))
    built += 1
print("built", built, "crop datasets ->", DST)
