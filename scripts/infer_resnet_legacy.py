"""Zero-shot probe driver: koine-wrapped Resnet3D with GP-era legacy preprocessing.

The post-refactor koine infer.py hard-codes tifxyz_robust normalization; the
2023/24 resnet pipeline trained on uint8 layers clipped to [0,200] then /255.
This driver reproduces that so preprocessing-mismatch can be ruled out.

Requires INK_DETECTION_REPO pointing at a villa `ink-detection` checkout
(for the koine imports).

Usage: .venv/bin/python infer_resnet_legacy.py <zarr> <wrapped_ckpt> <out_tif>
         [--layer-start N] [--layer-end N] [--patch 256] [--stride 192]
"""
import argparse, os, sys
import numpy as np
import tifffile, torch, zarr

sys.path.insert(0, os.environ.get("INK_DETECTION_REPO", "villa/ink-detection"))
from koine_machines.models.resnet3d import ResNet3DSegmentationModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_zarr"); ap.add_argument("checkpoint"); ap.add_argument("output_tiff")
    ap.add_argument("--layer-start", type=int, default=23)
    ap.add_argument("--layer-end", type=int, default=41)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--stride", type=int, default=192)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    root = zarr.open(args.input_zarr, mode="r")
    arr = root["0"] if not isinstance(root, zarr.Array) else root
    vol = arr[args.layer_start:args.layer_end]  # (z, H, W) uint8
    z, H, W = vol.shape
    print("volume", vol.shape)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ResNet3DSegmentationModel(depth=50, pretrained=False)
    model.load_state_dict(payload["model"], strict=False)
    model.cuda().eval()

    # legacy normalization: clip to 200, scale to [0,1]
    volf = np.clip(vol, 0, 200).astype(np.float32) / 255.0

    P, S = args.patch, args.stride
    ys = list(range(0, max(H - P, 0) + 1, S)) or [0]
    xs = list(range(0, max(W - P, 0) + 1, S)) or [0]
    if ys[-1] != H - P: ys.append(H - P)
    if xs[-1] != W - P: xs.append(W - P)

    acc = np.zeros((H, W), dtype=np.float32)
    wgt = np.zeros((H, W), dtype=np.float32)
    coords = [(y, x) for y in ys for x in xs]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(coords), args.batch):
            chunk = coords[i:i + args.batch]
            batch = np.stack([volf[:, y:y+P, x:x+P] for y, x in chunk])
            t = torch.from_numpy(batch).cuda()  # (B, z, P, P) -> model adds channel dim
            out = model(t)["ink"]               # (B, 1, p', p')
            out = torch.sigmoid(out.float())
            out = torch.nn.functional.interpolate(out, size=(P, P), mode="bilinear")
            out = out[:, 0].cpu().numpy()
            for (y, x), o in zip(chunk, out):
                acc[y:y+P, x:x+P] += o
                wgt[y:y+P, x:x+P] += 1.0
            if (i // args.batch) % 50 == 0:
                print(f"{i}/{len(coords)} patches", flush=True)
    pred = acc / np.maximum(wgt, 1e-6)
    tifffile.imwrite(args.output_tiff, (pred * 255).astype(np.uint8))
    print("wrote", args.output_tiff)


if __name__ == "__main__":
    main()
