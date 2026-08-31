#!/usr/bin/env bash
# Generalized grow+render+probe sweep for one scroll band.
# Usage: grow_probe_sweep.sh <SCROLL> <VOLUME_ZARR_S3_NAME> <BAND_DIR_NAME> <seed> [seed...]
# Seeds: "x,y,z". Cleans per-segment surface volumes after probing (keeps predictions).
set -uo pipefail
SCROLL=$1; VOLNAME=$2; BAND=$3; shift 3
V="https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/$SCROLL/volumes/$VOLNAME"
CKPT=/data/vesuvius/runs/ink_sim86v2_resnet_ft/ckpt_005000.pth
IV="${INK_VENV_PY:?set INK_VENV_PY to the villa ink-detection venv python (see README prerequisites)}"
PACK="$(dirname "$0")/tifs_to_zarr.py"
ROOT=/data/vesuvius/$SCROLL
mkdir -p "$ROOT/paths" "$ROOT/vc-cache/dummy" "/data/vesuvius/runs/${SCROLL}_probe"
[ -f "$ROOT/seed_params.json" ] || cat > "$ROOT/seed_params.json" <<EOF
{"mode":"seed","generations":200,"min_area_cm":0.3,"thread_limit":1,"cache_size":4e9,"use_cuda":true,"cache_root":"/data/$SCROLL/cache"}
EOF
mkdir -p "$ROOT/cache"
for S in "$@"; do
  AVAIL=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
  [ "$AVAIL" -lt 6 ] && { echo "STOP disk ${AVAIL}G"; break; }
  X=${S%%,*}; R=${S#*,}; Y=${R%%,*}; Z=${R#*,}
  echo "=== $SCROLL/$BAND seed $X $Y $Z"
  before=$(ls "$ROOT/paths" | wc -l)
  docker run --rm --gpus all --entrypoint bash -v /data/vesuvius:/data \
    ghcr.io/scrollprize/villa/volume-cartographer:edge -c \
    "vc_grow_seg_from_seed -v /data/$SCROLL/$BAND -t /data/$SCROLL/paths -p /data/$SCROLL/seed_params.json -s $X $Y $Z" >/dev/null 2>&1
  after=$(ls "$ROOT/paths" | wc -l)
  [ "$after" -le "$before" ] && { echo "GROW FAIL/small seed $S"; continue; }
  SEG=$(ls -t "$ROOT/paths" | head -1)
  echo "grown $SEG"
  docker run --rm --entrypoint bash -v /data/vesuvius:/data \
    ghcr.io/scrollprize/villa/volume-cartographer:edge -c \
    "vc_render_tifxyz -v /data/$SCROLL/vc-cache/dummy --remote-url '$V' -s /data/$SCROLL/paths/$SEG --scale 1 -g 0 -n 65 --slice-step 1 --cache-gb 20 --tif-output /data/$SCROLL/surface-volumes/$SEG/layers" >/dev/null 2>&1 \
    || { echo "RENDER FAIL $SEG"; continue; }
  sudo chown -R ubuntu:ubuntu "$ROOT/surface-volumes/$SEG"
  $IV "$PACK" "$ROOT/surface-volumes/$SEG/layers" "$ROOT/surface-volumes/$SEG/$SEG.zarr" >/dev/null \
    || { echo "PACK FAIL $SEG"; continue; }
  rm -rf "$ROOT/surface-volumes/$SEG/layers"
  cd "$HOME/code/vesuvius/villa/ink-detection" && $IV -m koine_machines.inference.infer \
    "$ROOT/surface-volumes/$SEG/$SEG.zarr" "$CKPT" \
    "/data/vesuvius/runs/${SCROLL}_probe/$SEG.tif" --layer-start 23 --layer-end 41 >/dev/null 2>&1 \
    || { echo "INFER FAIL $SEG"; continue; }
  rm -rf "$ROOT/surface-volumes/$SEG/$SEG.zarr"
  echo "PROBED $SEG"
done
echo "BAND DONE $SCROLL/$BAND"
