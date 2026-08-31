#!/usr/bin/env bash
set -uo pipefail
V="https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1218/volumes/20250521120456-8.640um-1.2m-116keV-masked.zarr"
CKPT=/data/vesuvius/runs/ink_sim86v2_resnet_ft/ckpt_005000.pth
IV=$HOME/code/vesuvius/villa/ink-detection/.venv/bin/python
PACK=$HOME/code/vesuvius/scripts/tifs_to_zarr.py
for S in 5112,3216,11200 2216,4232,11200 4184,5280,11200 5776,3264,11800 1744,4184,11800 4592,5200,11800 2784,3344,12400 5848,4296,12400 5904,5312,12400; do
  AVAIL=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
  [ "$AVAIL" -lt 6 ] && { echo "STOP disk ${AVAIL}G"; break; }
  X=${S%%,*}; R=${S#*,}; Y=${R%%,*}; Z=${R#*,}
  echo "=== seed $X $Y $Z"
  before=$(ls /data/vesuvius/PHerc1218/paths | wc -l)
  docker run --rm --gpus all --entrypoint bash -v /data/vesuvius:/data \
    ghcr.io/scrollprize/villa/volume-cartographer:edge -c \
    "vc_grow_seg_from_seed -v /data/PHerc1218/surface-m7.zarr -t /data/PHerc1218/paths -p /data/PHerc1218/seed_params.json -s $X $Y $Z" >/dev/null 2>&1
  SEG=$(ls -t /data/vesuvius/PHerc1218/paths | head -1)
  after=$(ls /data/vesuvius/PHerc1218/paths | wc -l)
  [ "$after" -le "$before" ] && { echo "GROW FAIL/too-small seed $S"; continue; }
  echo "grown $SEG"
  docker run --rm --entrypoint bash -v /data/vesuvius:/data \
    ghcr.io/scrollprize/villa/volume-cartographer:edge -c \
    "vc_render_tifxyz -v /data/PHerc1218/vc-cache/dummy --remote-url '$V' -s /data/PHerc1218/paths/$SEG --scale 1 -g 0 -n 65 --slice-step 1 --cache-gb 20 --tif-output /data/PHerc1218/surface-volumes/$SEG/layers" >/dev/null 2>&1 \
    || { echo "RENDER FAIL $SEG"; continue; }
  sudo chown -R ubuntu:ubuntu /data/vesuvius/PHerc1218/surface-volumes/$SEG
  $IV "$PACK" /data/vesuvius/PHerc1218/surface-volumes/$SEG/layers /data/vesuvius/PHerc1218/surface-volumes/$SEG/$SEG.zarr >/dev/null || { echo "PACK FAIL $SEG"; continue; }
  rm -rf /data/vesuvius/PHerc1218/surface-volumes/$SEG/layers
  cd $HOME/code/vesuvius/villa/ink-detection && $IV -m koine_machines.inference.infer \
    /data/vesuvius/PHerc1218/surface-volumes/$SEG/$SEG.zarr "$CKPT" \
    /data/vesuvius/runs/1218_probe/$SEG.tif --layer-start 23 --layer-end 41 >/dev/null 2>&1 \
    || { echo "INFER FAIL $SEG"; continue; }
  echo "PROBED $SEG"
done
echo "SWEEP COMPLETE"
