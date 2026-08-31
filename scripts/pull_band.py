"""Pull a z-band of an S3 zarr (level 0) into a local sparse copy.

Usage: pull_band.py <s3_prefix_of_zarr> <dst_dir> <z0> <z1>
Missing chunks (masked regions) 404 -> treated as fill.
Writes a VC-compatible volpkg-style meta.json next to the zarr.
"""
import urllib.request, os, sys, json, concurrent.futures as cf
from collections import Counter

B = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
P, DST, Z0, Z1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

meta_raw = json.load(urllib.request.urlopen(f"{B}/{P}/0/.zarray"))
CH = meta_raw["chunks"][0]
shape = meta_raw["shape"]

os.makedirs(DST + "/0", exist_ok=True)
for f in [".zattrs", ".zgroup"]:
    try: urllib.request.urlretrieve(f"{B}/{P}/{f}", f"{DST}/{f}")
    except Exception: pass
urllib.request.urlretrieve(f"{B}/{P}/0/.zarray", f"{DST}/0/.zarray")
json.dump({
    "type": "vol", "format": "zarr", "uuid": os.path.basename(DST),
    "name": os.path.basename(DST), "voxelsize": 8.64,
    "width": shape[2], "height": shape[1], "slices": shape[0],
    "min": 0.0, "max": 255.0,
}, open(f"{DST}/meta.json", "w"), indent=1)

zr = range(Z0 // CH, Z1 // CH + 1)
yx = range(0, (shape[1] - 1) // CH + 1)
keys = [(z, y, x) for z in zr for y in yx for x in yx]
print(len(keys), "chunks planned", flush=True)

def pull(k):
    z, y, x = k
    dst = f"{DST}/0/{z}/{y}/{x}"
    if os.path.exists(dst): return "have"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        urllib.request.urlretrieve(f"{B}/{P}/0/{z}/{y}/{x}", dst + ".part")
        os.replace(dst + ".part", dst); return "ok"
    except urllib.error.HTTPError as e:
        return "missing" if e.code == 404 else "err"
    except Exception:
        return "err"

c = Counter()
with cf.ThreadPoolExecutor(24) as ex:
    for i, r in enumerate(ex.map(pull, keys), 1):
        c[r] += 1
        if i % 4000 == 0: print(i, dict(c), flush=True)
print("DONE", dict(c))
