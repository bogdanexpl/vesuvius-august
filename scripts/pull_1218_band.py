import urllib.request, urllib.parse, os, concurrent.futures as cf
from collections import Counter
B = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
P = "PHerc1218/representations/predictions/surfaces/20250521120456-surface-20260413222639-surface-m7-L0-th0.2.zarr"
DST = "/data/vesuvius/PHerc1218/surface-m7.zarr"
CH = 192
zr = range(11000//CH, 13000//CH + 1)   # 57..67
yr = range(0, 7593//CH + 1)
xr = range(0, 7593//CH + 1)
os.makedirs(DST + "/0", exist_ok=True)
for f in [".zattrs", ".zgroup"]:
    try: urllib.request.urlretrieve(f"{B}/{P}/{f}", f"{DST}/{f}")
    except Exception: pass
urllib.request.urlretrieve(f"{B}/{P}/0/.zarray", f"{DST}/0/.zarray")
keys = [(z,y,x) for z in zr for y in yr for x in xr]
print(len(keys), "chunks planned")
def pull(k):
    z,y,x = k
    dst = f"{DST}/0/{z}/{y}/{x}"
    if os.path.exists(dst): return "have"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        urllib.request.urlretrieve(f"{B}/{P}/0/{z}/{y}/{x}", dst + ".part")
        os.replace(dst + ".part", dst); return "ok"
    except urllib.error.HTTPError as e:
        return "missing" if e.code == 404 else "err"
    except Exception: return "err"
c = Counter()
with cf.ThreadPoolExecutor(24) as ex:
    for i, r in enumerate(ex.map(pull, keys), 1):
        c[r] += 1
        if i % 2000 == 0: print(i, dict(c), flush=True)
print("DONE", dict(c))
