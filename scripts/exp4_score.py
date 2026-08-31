"""Exp 4 scoring: per-held-out-fragment AUC / best-threshold Dice + previews."""
import glob, numpy as np, tifffile, zarr
import PIL.Image as I

RUNS = "/data/vesuvius/runs/exp4_loso"
ROOT = "/data/vesuvius/ink-dataset/frags"

def arr(p):
    z = zarr.open(p, mode="r")
    return z if isinstance(z, zarr.Array) else z["0"]

print(f"{'fragment':22s} {'AUC':>6s} {'diceBest':>8s} {'thr':>4s} {'inkFrac':>7s}")
rows = []
for pred_path in sorted(glob.glob(f"{RUNS}/pred_*.tif")):
    name = pred_path.split("pred_")[1].replace(".tif", "")
    name = name.replace("within_", "")
    tag = "WITHIN " if "within_" in pred_path else "cross  "
    pred = tifffile.imread(pred_path).astype(np.float32) / 255.0
    lab = np.asarray(arr(f"{ROOT}/{name}/{name}/{name}_inklabels.zarr")[32]) > 127
    msk = np.asarray(arr(f"{ROOT}/{name}/{name}/{name}_supervision_mask.zarr")[32]) > 127
    H = min(pred.shape[0], lab.shape[0]); W = min(pred.shape[1], lab.shape[1])
    p, l, m = pred[:H, :W], lab[:H, :W], msk[:H, :W]
    pv, lv = p[m], l[m]
    # AUC via rank statistic on a subsample
    idx = np.random.default_rng(0).choice(pv.size, size=min(2_000_000, pv.size), replace=False)
    ps, ls = pv[idx], lv[idx]
    order = np.argsort(ps)
    ranks = np.empty_like(order, dtype=np.float64); ranks[order] = np.arange(1, ps.size + 1)
    npos, nneg = int(ls.sum()), int((~ls).sum())
    auc = (ranks[ls].sum() - npos * (npos + 1) / 2) / max(npos * nneg, 1)
    best = (0.0, 0.5)
    for t in np.arange(0.2, 0.95, 0.05):
        pb = pv > t
        inter = float((pb & lv).sum())
        dice = 2 * inter / max(float(pb.sum() + lv.sum()), 1)
        if dice > best[0]: best = (dice, t)
    print(f"{tag}{name:22s} {auc:6.3f} {best[0]:8.3f} {best[1]:4.2f} {float(lv.mean()):7.3f}")
    rows.append((name, auc, best[0]))
    d = max(1, H // 1200)
    over = np.stack([ (p[::d,::d]*255).astype(np.uint8),
                      (l[::d,::d]*255).astype(np.uint8),
                      np.zeros_like(p[::d,::d], np.uint8) ], -1)
    I.fromarray(over).save(f"{RUNS}/overlay_{tag.strip()}_{name}.png")  # red=pred, green=GT, yellow=both
if rows:
    print("\nmacro AUC", round(float(np.mean([r[1] for r in rows])), 3),
          "macro Dice", round(float(np.mean([r[2] for r in rows])), 3))
