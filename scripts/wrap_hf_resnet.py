#!/usr/bin/env python3
"""Wrap the public HF `scrollprize/resnet50_7.9um_scroll1_frags` state dict
into the koine checkpoint envelope used by `infer_resnet_legacy.py` and the
`checkpoint:` config key (rung 3+, `koine_wrapped.pth`).

The HF state dict is shape-identical to koine's `ResNet3DSegmentationModel`
(including the decoder); only the unused 1139-class `fc` head differs, and
the loaders here use `strict=False`, so it is dropped.

Usage: python wrap_hf_resnet.py <hf_state_dict.pth> <out_koine_ckpt.pth>
"""
import sys

import torch

src, dst = sys.argv[1], sys.argv[2]
sd = torch.load(src, map_location="cpu", weights_only=False)
for key in ("state_dict", "model"):
    if isinstance(sd, dict) and key in sd and not hasattr(sd[key], "shape"):
        sd = sd[key]
sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
torch.save({"model": sd}, dst)
print(f"wrapped {len(sd)} tensors -> {dst}")
