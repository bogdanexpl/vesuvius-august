#!/usr/bin/env bash
# Batch ink-probe published PHerc1447 auto_grown segments with the sim86ft model.
# Per segment: S3-stream render 65 layers -> pack zarr -> infer (layers 23-41) -> overview PNG.
# Layer TIFFs deleted after packing (disk). Skips segments with existing predictions.
set -uo pipefail
V="https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1447/volumes/20250521151220-8.640um-1.2m-116keV-masked.zarr"
CKPT=/data/vesuvius/runs/ink_sim86_resnet_ft/ckpt_005000.pth
SV=/data/vesuvius/PHerc1447/surface-volumes
OUT=/data/vesuvius/runs/1447_probe/batch
IV=$HOME/code/vesuvius/villa/ink-detection/.venv/bin/python
PACK=$HOME/code/vesuvius/scripts/tifs_to_zarr.py
mkdir -p "$OUT" "$SV" /data/vesuvius/PHerc1447/vc-cache/dummy

for SEGDIR in /data/vesuvius/PHerc1447/segments/raw/auto_grown_*; do
  SEG=$(basename "$SEGDIR")
  PRED="$OUT/${SEG}.tif"
  [ -f "$PRED" ] && { echo "SKIP $SEG (done)"; continue; }
  AVAIL=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
  [ "$AVAIL" -lt 6 ] && { echo "STOP: only ${AVAIL}G free"; break; }
  echo "=== $SEG"
  if [ ! -d "$SV/$SEG/$SEG.zarr" ]; then
    docker run --rm --name lab-vc-batch --entrypoint bash -v /data/vesuvius:/data \
      ghcr.io/scrollprize/villa/volume-cartographer:edge -c \
      "vc_render_tifxyz -v /data/PHerc1447/vc-cache/dummy --remote-url '$V' \
       -s /data/PHerc1447/segments/raw/$SEG --scale 1 -g 0 -n 65 --slice-step 1 \
       --cache-gb 20 --tif-output /data/PHerc1447/surface-volumes/$SEG/layers" \
      || { echo "RENDER FAIL $SEG"; continue; }
    sudo chown -R ubuntu:ubuntu "$SV/$SEG"
    $IV "$PACK" "$SV/$SEG/layers" "$SV/$SEG/$SEG.zarr" || { echo "PACK FAIL $SEG"; continue; }
    rm -rf "$SV/$SEG/layers"
  fi
  cd "$HOME/code/vesuvius/villa/ink-detection" && $IV -m koine_machines.inference.infer \
    "$SV/$SEG/$SEG.zarr" "$CKPT" "$PRED" --layer-start 23 --layer-end 41 \
    || { echo "INFER FAIL $SEG"; continue; }
  echo "DONE $SEG"
done
echo "BATCH COMPLETE"
