"""Exp 4: fetch classic fragments and pack koine flat datasets.

Per fragment: 65-layer surface volume + inklabels.png + mask.png from
dl.ash2txt, packed as zarr group '0' (65,H,W) + label/mask planes at z=32
(= shape[0]//2, where the koine patch finder samples) + dummy x/y/z.tif so
segment discovery accepts the folder.
Dataset root: /data/vesuvius/ink-dataset/frags/<scrollgroup>__<frag>/...
"""
import io, os, sys, json, urllib.request, base64
import concurrent.futures as cf
import numpy as np, tifffile, zarr
import PIL.Image as I
I.MAX_IMAGE_PIXELS = None

AUTH = base64.b64encode(b"registeredusers:only").decode()
B = "https://dl.ash2txt.org/fragments"
FRAGS = {
    "paris2__frag1": "Frag1/PHercParis2Fr47.volpkg/working/54keV_exposed_surface",
    "paris2__frag2": "Frag2/PHercParis2Fr143.volpkg/working/54keV_exposed_surface",
    "paris1__frag3": "Frag3/PHercParis1Fr34.volpkg/working/54keV_exposed_surface",
    "paris1__frag4": "Frag4/PHercParis1Fr39.volpkg/working/54keV_exposed_surface",
    "pherc1667__frag5": "Frag5/PHerc1667Cr1Fr3.volpkg/working/PHerc1667Cr01Fr03_70keV_3.24um/surface_processing",
    "pherc51__frag6": "Frag6/PHerc51Cr4Fr8.volpkg/working/PHerc0051Cr04Fr08_53keV_3.24um/surface_processing",
    "pherc343p__front": "PHerc0343P/paths/2um_front_surface",
    "pherc500p2__front": "PHerc0500P2/paths/2um_front_surface",
}
# per-fragment filename prefixes (Frag4 uses prefixed assets)
PREFIX = {"paris1__frag4": "PHercParis1Fr39_54keV_",
          "pherc343p__front": "343P_", "pherc500p2__front": "500P2_"}
# fragments whose layers are 66 PNGs under layers/ (no prefix on layer files)
PNG_LAYERS = {"pherc343p__front", "pherc500p2__front"}
ROOT = "/data/vesuvius/ink-dataset/frags"

def fetch(url):
    req = urllib.request.Request(url, headers={"Authorization": f"Basic {AUTH}"})
    return urllib.request.urlopen(req, timeout=600).read()

def build(name, base):
    seg = f"{ROOT}/{name}/{name}"
    if os.path.exists(f"{seg}/{name}_supervision_mask.zarr"):
        print(name, "already built"); return
    os.makedirs(seg, exist_ok=True)
    pfx = PREFIX.get(name, "")
    labels = np.array(I.open(io.BytesIO(fetch(f"{B}/{base}/{pfx}inklabels.png"))).convert("L"))
    try:
        mask = np.array(I.open(io.BytesIO(fetch(f"{B}/{base}/{pfx}mask.png"))).convert("L"))
    except Exception:
        mask = np.full_like(labels, 255)
    H, W = labels.shape
    print(name, "labels", labels.shape, "ink frac", round(float((labels > 127).mean()), 4), flush=True)

    vol = None
    if name in PNG_LAYERS:
        def get_layer(i):
            return i, np.array(I.open(io.BytesIO(fetch(f"{B}/{base}/layers/{i:02d}.png"))))
        nlayers = 65  # use central 65 of 66
    else:
        def get_layer(i):
            return i, tifffile.imread(io.BytesIO(fetch(f"{B}/{base}/{pfx}surface_volume/{i:02d}.tif")))
        nlayers = 65
    with cf.ThreadPoolExecutor(8) as ex:
        for i, img in ex.map(get_layer, range(nlayers)):
            if vol is None:
                vol = np.zeros((65,) + img.shape, np.uint8)
            vol[i] = (img >> 8).astype(np.uint8) if img.dtype == np.uint16 else img.astype(np.uint8)
    assert vol.shape[1:] == (H, W), f"{name}: vol {vol.shape} vs labels {labels.shape}"

    g = zarr.open_group(f"{seg}/{name}.zarr", mode="w")
    g.create_dataset("0", data=vol, chunks=(65, 128, 128), dtype="u1")
    for suffix, plane in [("_inklabels", (labels > 127)), ("_supervision_mask", (mask > 127))]:
        arr = np.zeros((65, H, W), np.uint8)
        arr[32] = plane.astype(np.uint8) * 255
        gg = zarr.open_group(f"{seg}/{name}{suffix}.zarr", mode="w")
        gg.create_dataset("0", data=arr, chunks=(65, 128, 128), dtype="u1")
    dummy = np.zeros((max(H // 20, 4), max(W // 20, 4)), np.float32)
    for n, fill in [("x", 1.0), ("y", 1.0), ("z", 1.0)]:
        tifffile.imwrite(f"{seg}/{n}.tif", dummy + fill)
    json.dump({"scale": [0.05, 0.05], "type": "seg", "uuid": name, "format": "tifxyz",
               "scroll_source": name.split("__")[0]}, open(f"{seg}/meta.json", "w"))
    print(name, "BUILT", vol.shape, flush=True)

for name, base in FRAGS.items():
    try:
        build(name, base)
    except Exception as e:
        print(name, "FAILED:", e, flush=True)
print("FETCH DONE")
