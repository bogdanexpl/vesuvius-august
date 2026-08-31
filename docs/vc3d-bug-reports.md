# VC3D bug reports (drafts for ScrollPrize/villa issues)

> 4 issues: render zarr-output segfault, local-zarr chunk lookup, koine cache race,
> grower cross-cutting surfaces on coarse binary predictions.

Found 2026-08-11/12 while rendering PHerc1447 published segments on the `:edge`
container image (`ghcr.io/scrollprize/villa/volume-cartographer:edge`).
Host: Ubuntu 22.04, Docker, 24 cores / 62 GB RAM.

---

## 1. `vc_render_tifxyz --zarr-output` segfaults deterministically

**Summary.** Any render using `--zarr-output` crashes (SIGSEGV, exit 139 — one
configuration aborted with exit 134). The identical command with `--tif-output`
completes fine. Reproduced with n=1 and n=65 slices, with both a local `-v`
volume and `--remote-url` streaming, and with `--resume` (crashes at the same
tile row it previously reached).

**Repro.**

```bash
docker run --rm --entrypoint bash -v /data/vesuvius:/data \
  ghcr.io/scrollprize/villa/volume-cartographer:edge -c \
  "vc_render_tifxyz \
     -v /data/PHerc1447/vc-cache/dummy \
     --remote-url 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1447/volumes/20250521151220-8.640um-1.2m-116keV-masked.zarr' \
     -s /data/PHerc1447/segments/raw/auto_grown_20250703034159599 \
     --scale 1 -g 0 -n 1 \
     --zarr-output /tmp/out.zarr"
# → Segmentation fault (core dumped), exit 139
# Same command with --tif-output /tmp/out → completes, correct images
```

Observed matrix (segment `auto_grown_20250703034159599`, PHerc1447 8.64 µm volume):

| output | n | volume | result |
|---|---|---|---|
| tif  | 1  | remote | OK |
| tif  | 3  | local  | OK |
| tif  | 65 | remote | OK (500 MB, correct) |
| zarr | 1  | local  | SIGSEGV |
| zarr | 1  | remote | SIGSEGV |
| zarr | 3  | local  | SIGSEGV |
| zarr | 9  | local  | SIGABRT (134) |
| zarr | 65 | remote | SIGSEGV at tile-row 6/29; `--resume` re-crashes just past the resume point |

Partially written zarr chunks are left behind (valid uint8 data for completed
tile rows), so the writer path works until it dies — likely an OOB/synchronization
issue in the zarr chunk writer rather than data-dependent input.

---

## 2. Local zarr volume reads return fill-value; chunk lookups use wrong indices

**Summary.** Rendering from a *local* OME-Zarr copy (`-v /path/vol.zarr`, no
`--remote-url`) silently produces all-zero images: the reader never opens any
chunk file. `strace` shows chunk existence checks against index triples that
match neither (z,y,x) nor (x,y,z) of the volume. The same segment rendered from
the same volume via `--remote-url` streaming reads the correct chunks.

**Setup.** Local sparse copy of
`PHerc1447/volumes/20250521151220-8.640um-1.2m-116keV-masked.zarr` (level 0
only, zarr v2, uint8, 128³ chunks, `dimension_separator: "/"`, shape
(24297, 8343, 8343)); segment bbox chunks present, verified readable in Python
(zarr-python reads correct data).

**Evidence.** Segment bbox spans chunk indices z 172–193, y 20–36, x 34–45.
Rendering it locally:

```
newfstatat(... "/vol.zarr/0/34.41.21") = ENOENT
```

— all probed triples fall in ranges dim1 28–56, dim2 21–43, dim3 15–34 (~4–5×
too small in the z dimension to be the correct chunk grid). No chunk file is
ever opened; output = fill value everywhere. Renaming chunks to `.`-separated
form and flipping `dimension_separator` changes nothing (both layouts present →
still zero). The `--remote-url` path renders the identical segment correctly,
so the segment geometry and volume data are fine.

Suspect: axis order / scale mapping in the local `FileSystemChunkSource` (or
`utils::ZarrArray`) chunk-key computation, possibly missing the OME multiscale
coordinate transform that the HTTP path applies.

---

*Both bugs bite the exact "alternative unwrapping pipelines integrated with
VC3D" workflow the team encourages: streaming render → zarr surface volume →
ink inference. Workaround used: `--remote-url` + `--tif-output`, then repack
TIFFs to zarr out-of-band.*

---

## 3. koine `disk_cache._evict()` crashes on concurrent worker eviction

**Summary.** `koine_machines/common/disk_cache.py` `_evict()` calls
`entry.stat()` on a scandir snapshot; with multiple dataloader workers each
holding their own cache store over the same `--cache-dir`, a sibling worker
can evict a file between scandir and stat → `FileNotFoundError` kills the
whole `infer_full3d_tifxyz` run (observed ~2 h into a 7 h inference).

**Fix (one-liner class):** wrap the stat in try/except FileNotFoundError and
skip vanished entries (unlink already uses `missing_ok=True`). Patch applied
locally and verified; happy to PR.

## 4. `vc_grow_seg_from_seed` produces cross-cutting (non-sheet) surfaces on a coarse binary prediction

**Observed** on PHerc 1203's 2.403 µm scan, whose only published surface
prediction is `…-surface-m7-L2-th0.2.zarr` — binary (all nonzero voxels are
exactly 255) at quarter resolution (L2, 9.6 µm effective), on a heavily
compressed scroll where that mask visually fuses adjacent sheets.

**Symptom.** Growth completes normally (plausible area, generations, meta),
but the resulting tifxyz surfaces cut *across* the windings instead of
following a sheet. Renders of them look like credible fibrous texture at
every depth — actually winding cross-sections — and any ink model probing
them returns structured noise. Nothing in the tool errors or warns.

**Evidence / repro measurements** (segments grown 2026-08-27, seeds picked
on the prediction itself, full prediction pulled so support is complete):
- Sampling the volume at mesh vertices: no scale (×1/×2/×4) or axis
  permutation of the tifxyz coordinates exceeds ~60 % on-material — a true
  sheet surface sits at ~100 %.
- Sampling the grower's **own input prediction** at mesh vertices (its native
  frame): only 6–10 % of vertices land on nonzero prediction — the traced
  surface does not follow the signal it was grown from.
- The same grower + identical parameters produced valid sheet surfaces on
  PHerc 1447/1218/0800/0268 using native-resolution (L0) binary predictions,
  so binariness alone is not the trigger; the L2 + sheet-fusion combination
  is the distinguishing factor.

**Suggested guards.** (a) After growth, verify mesh-vertex on-prediction
fraction and refuse/warn below a threshold; (b) document that L2-scale
predictions on compressed scrolls are not valid growth targets. A cheap
downstream detector for already-grown surfaces: render the mid-layer and
check the in-segment dark-gap fraction (winding cross-sections show ~10–13 %
dark inter-sheet gaps vs ~1.5 % for true sheets in our data).
