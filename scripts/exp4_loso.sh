#!/usr/bin/env bash
# Exp 4: leave-one-scroll-out cross-chemistry benchmark over the fragment corpus.
# For each scroll group: train flat 64-ch vesuvius_unet on all OTHER groups
# (5k iters), then predict every fragment of the held-out group and score.
set -uo pipefail
IV="${INK_VENV_PY:?set INK_VENV_PY to the villa ink-detection venv python (see README prerequisites)}"
ROOT=/data/vesuvius/ink-dataset/frags
RUNS=/data/vesuvius/runs/exp4_loso
SCROLLS=$(ls "$ROOT" | sed 's/__.*//' | sort -u)
echo "groups: $SCROLLS"
mkdir -p "$RUNS"

for HOLD in $SCROLLS; do
  OUT="$RUNS/hold_$HOLD"
  [ -f "$OUT/ckpt_005000.pth" ] && { echo "SKIP $HOLD (trained)"; } || {
    echo "=== FOLD hold=$HOLD"
    mkdir -p "$OUT/trainset"
    rm -f "$OUT/trainset"/* 2>/dev/null
    for d in "$ROOT"/*; do
      g=$(basename "$d" | sed 's/__.*//')
      [ "$g" != "$HOLD" ] && ln -sf "$d/$(basename $d)" "$OUT/trainset/$(basename $d)"
    done
    cat > "$OUT/config.json" <<EOF
{
  "out_dir": "$OUT",
  "seed": 42,
  "mode": "flat",
  "model_type": "vesuvius_unet",
  "in_channels": 1,
  "model_config": { "autoconfigure": true, "z_projection_mode": "max" },
  "targets": { "ink": { "out_channels": 1, "activation": "none", "z_projection_mode": "max" } },
  "patch_size": [64, 256, 256],
  "patch_overlap": 0.5,
  "patch_min_labeled_coverage": 0.05,
  "batch_size": 2,
  "num_iterations": 5000,
  "learning_rate": 0.01,
  "mixed_precision": "fp16",
  "dataloader_workers": 8,
  "val_every": 1000,
  "save_every": 5000,
  "datasets": [ { "segments_path": "$OUT/trainset", "volume_scale": "0" } ]
}
EOF
    cd $HOME/code/vesuvius/villa/ink-detection && $IV -m koine_machines.training.train "$OUT/config.json" \
      > "$OUT/train.log" 2>&1 || { echo "TRAIN FAIL $HOLD"; tr '\r' '\n' < "$OUT/train.log" | tail -4; continue; }
    echo "TRAINED $HOLD"
  }
  for d in "$ROOT"/${HOLD}__*; do
    f=$(basename "$d")
    [ -f "$RUNS/pred_${f}.tif" ] && continue
    cd $HOME/code/vesuvius/villa/ink-detection && $IV -m koine_machines.inference.infer \
      "$d/$f/$f.zarr" "$OUT/ckpt_005000.pth" "$RUNS/pred_${f}.tif" \
      > /dev/null 2>&1 || { echo "INFER FAIL $f"; continue; }
    echo "PREDICTED $f"
  done
done
echo "LOSO COMPLETE"
