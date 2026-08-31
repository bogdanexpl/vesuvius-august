import sys, glob, numpy as np, tifffile, zarr

src, dst = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(src + "/*.tif"))
assert files, "no tifs in " + src
first = tifffile.imread(files[0])
h, w = first.shape
d = len(files)
print(f"{d} layers of {h}x{w} {first.dtype}")
a = zarr.open(dst, mode="w", shape=(d, h, w), chunks=(d, 128, 128), dtype="u1")
for i, f in enumerate(files):
    a[i] = tifffile.imread(f)
    if i % 16 == 0: print("layer", i, flush=True)
print("DONE", dst)
