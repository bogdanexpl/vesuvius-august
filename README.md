# Ink detection at production scan resolution: a benchmark, a scaling law, and a validated diagnosis

*Vesuvius Challenge progress submission, August 2026. MIT licensed. All
experiments on one RTX A6000, all data from the public open-data bucket.*

**What this contributes, in order of usefulness to someone reading scrolls:**

1. **Held-out cross-scroll ink detection is log-linear in training budget:
   the fold we scaled rose from AUC 0.61 to 0.868 on the public fragment
   data** (corroborated at 0.857 by its second held-out fragment; the
   six-fold LOSO macro at community-practice budgets is 0.62 — our own
   measurement, as no prior public number exists) — no new labels, no new
   architecture, just training past the point everyone (ourselves included)
   had stopped. The gain is
   log-linear at +0.101 AUC per doubling of per-fragment exposure and was
   still climbing when we ran out of schedule. Every published cross-scroll
   number, ours included, is a lower bound set by budget rather than by ink
   chemistry. See [the scaling law](#figures) and
   [training-runs.md](docs/training-runs.md).
2. **The first public cross-scroll ink benchmark** — leave-one-scroll-out
   folds and a 4x4 transfer matrix over 8 IR-labeled fragments from 6
   physical scrolls, with a reproducible corpus builder, so this axis can be
   measured rather than asserted.
3. **A surface-validation tool that catches a silent, results-destroying
   failure** — `scripts/validate_surface.py`. Segments that cut across
   windings render as plausible fibrous texture and make ink models emit
   confident noise. This separates them from real sheet surfaces by ~8x on a
   single number, in seconds, before you spend GPU time. It cost us three
   probe results to learn; it costs you one command.
4. **A controlled diagnosis of the 8.64 µm eligible batch** — every scroll of
   it silent under a budget-adequate model that carries passing positive
   controls, with surface error, segmentation generation, training budget and
   image-space acquisition statistics each excluded by experiment. This is
   the "no signal versus signal present but unrecovered" diagnostic the
   2026 open-problems post asked for, resolved to a batch-level cause.
5. **Four upstream bug reports**, one with a working patch and one with a
   validated detector tool — including two silent failures that produce
   plausible-looking garbage rather than errors.
   [docs/vc3d-bug-reports.md](docs/vc3d-bug-reports.md).

## Quickstart

Everything runs headless against the public bucket; no GUI, no cluster.

```bash
git clone https://github.com/bogdanexpl/vesuvius-august && cd vesuvius-august
pip install numpy zarr tifffile pillow          # scoring + validation tools
```

**Check whether a segment is usable before spending GPU time on it** — the
one command most likely to save you a day:

```bash
# a rendered surface volume, or a directory of layer TIFFs from vc_render_tifxyz
python scripts/validate_surface.py /data/scroll/surface-volumes/*/ --png check.png
# PASS  auto_grown_20251005230830031: dark-gap 0.013  (65 layers, 5.8 MP rendered)
# FAIL  auto_grown_20260827171030809: dark-gap 0.127  <- looks like a winding cross-section
```

It exits non-zero if anything failed, so it drops straight into a pipeline:

```bash
python scripts/validate_surface.py "$SEG/vol.zarr" --quiet || { echo "skipping $SEG"; continue; }
```

> **Prerequisites for the training/inference stages:** the ScrollPrize
> [villa](https://github.com/ScrollPrize/villa) `ink-detection` component
> (koine), checked out with its own venv — we used commit `4b8694ab`
> (2026-07-13). Point `INK_VENV_PY` at that venv's `python` (used by
> `exp4_loso.sh` and the probe drivers) and `INK_DETECTION_REPO` at the
> checkout (used by `infer_resnet_legacy.py`). `downsample_to_sim86.py`
> additionally needs `torch` and the Scroll 1 ink-tutorial dataset. The
> data-root constants at the top of the Python scripts assume
> `/data/vesuvius` — edit them for your layout.

**Rebuild the benchmark corpus** (public fragments, ~34 GB):

```bash
python scripts/exp4_fetch_frags.py          # fetches + packs 8 fragments, 6 scrolls
bash   scripts/exp4_loso.sh                 # leave-one-scroll-out folds
python scripts/exp4_score.py                # mask-restricted AUC and Dice per fold
```

**Reproduce the resolution result** (the 20-minute fine-tune that reads
simulated 8.6 µm text):

```bash
python scripts/downsample_to_sim86.py       # degrade labeled 2.4 µm data
# train with configs/ink_sim86_resnet_ft.json, then:
python scripts/letterform_sheet.py pred.tif out --um-per-px 8.64
```

**Probe a scroll segment end to end** — grow, render, pack, infer:

```bash
python scripts/pull_band.py <zarr-prefix> <dst> <z0> <z1>   # sparse band pull
python scripts/pick_seeds.py <band.zarr> <z...>             # radial-tier seeds
bash   scripts/grow_probe_sweep.sh <scroll> <volume> <band> <seeds...>
```

## Technical integration

- **Inputs:** OME-Zarr / Zarr volumes (streamed by URL or local), `tifxyz`
  quadmesh segmentations, and the layer-TIFF stacks `vc_render_tifxyz`
  produces. `validate_surface.py` accepts either a packed zarr or a raw
  layer directory.
- **Outputs:** uint8 prediction TIFFs in the input's frame, JSON score files,
  and PNG contact sheets. Nothing writes back into its input.
- **Modularity:** each stage is a separate script communicating through files
  in these formats, so any stage can be replaced — our renders work with
  other people's models and vice versa. The validation gate in particular is
  a pure function of a rendered volume and depends on nothing else here.

## Why

The 2026 open-problems post asks for "diagnostics distinguishing *no signal*
from *signal present but unrecovered*" at production scan resolutions. The 13
GP-eligible scrolls were scanned at 8.6/9.4 µm; every public ink model was
trained at 2.4–7.9 µm, and none has been systematically probed against an
8.6 µm scroll. We ran that probe as a controlled ladder on PHerc. 1447
(8.64 µm batch, 14 published `auto_grown` segments) and isolated **which part
of the domain gap actually blocks ink recovery**.

## The ladder

Each rung changes exactly one variable. All renders are 65-layer surface
volumes produced with `vc_render_tifxyz` streamed from S3 (see *Tooling notes*
for two upstream bugs found on the way).

| rung | teacher | input | result |
|---|---|---|---|
| 1 | 2.4 µm PHerc.Paris4 tutorial UNet | native 8.6 µm render | patch-noise, no letters |
| 2 | same | physically matched render (slice-step 0.278 → ±78 µm depth; 3.6× bicubic in-plane) | checkerboard noise — **naive rescaling does not transfer** |
| 3 | `resnet50_7.9um_scroll1_frags` (HF), zero-shot | native render, central 18 layers | surface-coherent fiber response, no letters |
| 3b | same, GP-era preprocessing (clip-200/255) | same | dead-flat output — negative is not a normalization artifact |
| 4 | **7.9 µm resnet fine-tuned 20 min on simulated-8.6 µm Paris4** (2.4 µm data ↓3.6×, 65→18 z) | simulated-8.6 µm Paris4 | **rows of legible Greek; 78 % of recovered ink lies outside the supervision mask** — genuine within-segment generalization |
| 5 | rung-4 model, zero-shot | native 1447 render | letter-scale amorphous blobs, no letterforms |
| 6a | rung-4 model | z-window sweep (5–23 … 41–59) | identical response at every depth — not a surface-offset error |
| 6b | rung-4 model | **all 14 published 1447 segments** | uniform honeycomb false-positive texture (1.6–5.7 % strong activations), zero letterforms anywhere |
| 7 | **acquisition-matched fine-tune**: measured real-vs-sim stats (real render has 2.3× less high-frequency energy, grad-std 5.8 vs 13.6, and a brighter/narrower histogram), rebuilt the sim dataset with 3D gaussian PSF blur + quantile LUT to real statistics, re-fine-tuned | sim: still reads Greek (sanity holds); real 1447: still amorphous blobs | **matching second-order statistics does not close the gap** |
| 8 | rung-7 model, zero-shot | **real 7.9 µm scan** of the same Scroll 1 segment (original volpkg layers) | **reads the same Greek rows from the real 7.9 µm acquisition** — noisier than sim but clearly legible. The sim-trained model transfers to real scans. |
| 9 | rung-7 model, zero-shot | **46 self-grown segments across three 8.64 µm scrolls** (our own headless `vc_grow_seg_from_seed` segments on 1218/0800/0268, five z-bands, inner/mid/outer windings; the 14 published 1447 segments were covered in rung 6) | uniformly quiet near-zero predictions (0.3–1.1 % strong activations, edge artifacts only), zero letterforms anywhere |
| 10 | **the decisive control** — proven 2.4 µm Scroll-1 model, zero-shot | **PHerc 1203 at its native 2.4 µm scan** (same protocol class as the training scan; 4 self-grown segments) | tile-level noise, no letterforms; row-pitch statistics inconsistent across segments (blob noise, not lines). **Cross-scroll transfer fails even with resolution and protocol held equal.** |

Every result below has an image behind it; see **Figures**.

## Figures

In the order the argument runs, all in [`figures/`](figures/).

**The positives — what a successful read looks like in this pipeline**

**fig9 — rung 4, generalization.** The most important image here. White is ink the model recovered from simulated-8.64 µm data; red marks the only regions it was ever trained on. Most of the legible Greek sits outside the red — 78 % of it — so the model learned to read at this resolution rather than memorising its supervision.

![fig9 — rung 4, generalization](figures/fig9_rung4_generalization.png)

**fig10 — rung 4, the read itself.** The same prediction without the mask overlay: rows of Greek at 8.6 µm-equivalent sampling, after a 20-minute fine-tune.

![fig10 — rung 4, the read itself](figures/fig10_rung4_sim86_read.png)

**fig11 — rung 8, sim to real.** The same model, zero-shot on a *real* 7.9 µm scan of that scroll. Noisier, same rows legible — the recipe survives the jump from simulated to real acquisitions.

![fig11 — rung 8, sim to real](figures/fig11_rung8_real79um_read.png)

**fig1 — the calibration bar.** Density-picked crops of a true positive: closed loops, consistent stroke width, letters on a baseline. Every negative in this submission is judged against this.

![fig1 — the calibration bar](figures/fig1_positive_control.png)

**fig12 — cross-scroll transfer.** A benchmark fold: green is ink recovered on a held-out scroll's fragment by a model that never saw that scroll, red is the fragment surface. This is what the AUC numbers look like as an image.

![fig12 — cross-scroll transfer](figures/fig12_benchmark_crossscroll_overlay.png)

**fig15 — the scaling law.** Held-out cross-scroll AUC against training exposure per fragment: log-linear at +0.101 per doubling, still climbing at the largest budget run, and passing the within-scroll model that had less training.

![fig15 — the scaling law](figures/fig15_budget_scaling_curve.png)

**fig17 — the benchmark as images.** Every held-out scroll's prediction beside its IR ground truth at the community-practice budget (AUC 0.48–0.67), and the same Paris 1 fold after scaling (0.868): the spectrum the numbers describe.

![fig17 — the benchmark as images](figures/fig17_loso_all_targets.png)

**fig2 — the model-validation control.** The exact checkpoint used on the eligible scrolls, reading a simulated-8.64 µm PHerc 1667 fragment. Without this the negatives would be unfalsifiable.

![fig2 — the model-validation control](figures/fig2_exp6_model_validation.png)


**The negatives — the eligible scrolls under that same validated model**

**fig3 — PHerc 1447.** Densest crops from the budget-adequate probe: speckle, no closed forms, no baseline.

![fig3 — PHerc 1447](figures/fig3_1447_budget_adequate_negative.png)

**fig13 — every published 1447 segment.** Contact sheet across the segments: uniform texture, no letterforms anywhere.

![fig13 — every published 1447 segment](figures/fig13_rung9_1447_all_segments.png)

**fig14 — the offset sweep's best window.** The highest-response depth of nine tested from −207 to +207 µm. If the surface were merely offset from the ink layer, letters would appear here.

![fig14 — the offset sweep's best window](figures/fig14_exp7_offset_peak.png)

**fig4 — the rest of the batch.** The loudest of six segments across PHerc 1218, 0800 and 0268 — still speckle.

![fig4 — the rest of the batch](figures/fig4_exp8_best_batch_segment.png)


**The surface-validity work**

**fig5 — the silent failure mode.** Five render depths of a self-grown surface plus its prediction. Every depth shows winding cross-sections, not a sheet: the grower produced geometry that renders plausibly and means nothing.

![fig5 — the silent failure mode](figures/fig5_crosssection_failure_mode.png)

**fig6 — what valid surfaces look like.** Face-on crops of scan-team-published segments: continuous woven fibre. The dark-gap fraction separates these from cross-sections by about 8×.

![fig6 — what valid surfaces look like](figures/fig6_faceon_valid_surfaces.png)


**The late-August results**

**fig16 — selection heuristic refuted.** Choosing training data by ink family does not improve transfer: the same-family pair loses on one target and wins on the other, because one set wins both.

![fig16 — selection heuristic refuted](figures/fig16_family_selection_refuted.png)

**fig7 — Scroll 5 validation.** Our read of a real, read, held-out scroll (top) beside the community's prediction of the same segment (bottom).

![fig7 — Scroll 5 validation](figures/fig7_scroll5_validation.png)

**fig8 — the same region, zoomed.** 3 × 2 cm from both predictions for direct comparison.

![fig8 — the same region, zoomed](figures/fig8_scroll5_zoom_pair.png)



## What this establishes

0. **The recipe transfers to real acquisitions.** A model fine-tuned only on
   *simulated* 8.6 µm data reads the *real* 7.9 µm scan of the same scroll
   zero-shot (rung 8) — so the 1447 failure below is not a sim-to-real
   artifact of our pipeline.
1. **Resolution is not the blocker.** After a 20-minute fine-tune on
   resolution-degraded data, ink at 8.6 µm-equivalent sampling is readable —
   including text far outside the supervised region. The information survives
   the pixel pitch.
2. **Input-side rescaling cannot bridge the gap** (rung 2), and the failure of
   the 7.9 µm model is **not a preprocessing artifact** (rung 3b).
3. **The master variable is scroll identity, not resolution or protocol.** Rung 10
   is the clean control: PHerc 1203 has a native 2.4 µm scan under the same
   protocol the reference model was trained on — and still nothing transfers.
   Combined with rung 8 (the same model family reads Scroll 1 at 2.4 µm,
   7.9 µm and simulated 8.6 µm), the evidence says cross-scroll domain shift
   (ink chemistry / preservation state) dominates everything else — the
   open-problems post's cross-scroll-generalization bottleneck, now with a
   controlled experiment behind it.
4. **The 8.64 µm batch silence (rung 9) is therefore primarily a chemistry/
   domain problem.** A model that provably reads real 7.9 µm Scroll 1 (rung 8)
   and simulated 8.6 µm (rung 4) finds nothing on 60 segments across PHerc 1447, 1218, 800 and 268
   (all 14 published 1447 segments + 46 self-grown) — multiple z-bands,
   inner to outer windings — and the rung-10 control shows the same silence at
   native 2.4 µm on 1203. Acquisition physics may still contribute at 8.64 µm,
   but it is not the primary blocker; a test scan of a known-ink object under
   the 8.64 µm protocol would settle its share (a question for the scan team).
5. **Per-segment ink presence on these scrolls is currently unjudgeable.** Any claim of
   "this segment is blank" would be unsupported until the acquisition gap is
   modeled.

## Revision (2026-08-25): the budget confound, the benchmark, and the adequate probe

Work after the ladder materially refines conclusions 3–4 above.

1. **To our knowledge the first public cross-scroll ink benchmark (LOSO).**
   8 IR-labeled fragments from 6 physical scrolls (public,
   `scripts/exp4_fetch_frags.py`), trained leave-one-scroll-out, plus a full
   4×4 pairwise matrix on the first 4-scroll / 6-fragment corpus. Structure:
   Paris1↔PHerc51 form a tight transfer cluster at ceiling; PHerc 1667 is
   isolated — 0.51 in the v1 LOSO, 0.58 in the six-fold, weakest matrix node
   in both directions (0.54–0.63). Every fold's prediction is shown against
   its IR ground truth in `figures/fig17_loso_all_targets.png`.
2. **Every cross-scroll number is a training-budget lower bound.** Crossing
   corpus size × budget shows held-out AUC is monotone in *iterations per
   training fragment* (r = +0.98): .548 → .868 from 714 → 7500 it/frag,
   log-linear **+0.101 AUC per doubling**, unsaturated. At matched budget the
   cross-scroll penalty is only **−0.02…−0.04 AUC** (~94–97 % of
   within-scroll), and a cross-scroll model at 30k *beats* the within-scroll
   ceiling at 15k. The community's "models don't generalize across scrolls" —
   and our own rung-10 phrasing — is largely (not entirely) an
   under-training artifact. The ladder's negatives all used models we now
   know were under-trained; conclusion 3 stands only for that regime.
3. **The budget-adequate 1447 probe is still negative — and has a passing
   control.** A 5-scroll model trained at the scaling-law operating point on
   8.64 µm-degraded fragments, probed on depth-matched 1447 renders
   (xy 8.64 µm / z 3.24 µm both domains): speckle, no letterforms (0.5–2.2 %
   strong activations). The *same checkpoint* reads clear Greek rows on a
   sim-8.64 µm 1667 fragment — model, resolution treatment and depth match
   all verified. Remaining explanations, ranked: virtual-unwrapping surface
   error (fragments are exposed surfaces; every 1447 surface descends from m7
   predictions), 1447 chemistry outside the 5-scroll corpus span, blank
   regions. Two further pre-registered experiments then eliminated the
   surface explanations and generalized the result: an offset sweep ±207 µm
   found no letterforms at any depth (peaks discordant across segments =
   noise, not placement bias), a ten-times-larger Nov-2025-generation
   segment read *quieter* still (0.22 %), and the first budget-adequate probe
   of PHerc 1218/0800/0268 (six ~47 cm² segments) returned 0.08–0.34 % —
   below every Exp 6–7 1447 value (0.5–2.5 %), bracketing that quiet Nov
   segment. **Every scroll of the 8.64 µm batch is silent
   under a validated model with a passing positive control, and response
   decreases as surface quality improves.** The published `layers_ink`
   predictions shipped with the segments also show no letterforms, so no
   existing model reads them either. The cause is batch-level: the shared
   116 keV / 1.2 m protocol and/or chemistry common to these four scrolls
   but absent from the classic corpus. This is the strongest form of the "what
   happens if the models don't generalize" diagnostic the open-problems
   post poses — and it makes the
   known-ink phantom scan under the 8.64 µm protocol the single most
   valuable experiment now available to the community (a question for the
   scanning team, not for model builders).
4. **Ink density families (exploratory, cause unproven).** Cohen's d of
   ink-vs-background intensity splits the corpus into a carbon-like family
   (Paris + PHerc 51 — no positive contrast, per-fragment d −0.18…+0.06) and
   a denser-ink family (1667/343P/500P2, d ≥ +0.19) that tracks the transfer clusters — but beam energy is
   confounded per fragment, so we report a density-contrast difference of
   unproven cause, gated on a PHerc 51 multi-energy sweep.

## Late addition (2026-08-27): the face-on render control, and a grower failure mode

Probing PHerc 1203's native 2.4 µm scan (the only eligible scroll with a
classic-protocol scan) surfaced a failure mode with community-wide relevance:
`vc_grow_seg_from_seed`, fed the only available surface prediction for that
scan (a *binary, quarter-resolution* mask on a heavily compressed scroll),
produces surfaces that cut **across** the windings — renders look like
plausible fibrous texture but are winding cross-sections, and a model probing
them returns confident-looking noise. Three of our own probe results were
voided this way before any conclusion was drawn. Two takeaways we recommend
to anyone rendering self-grown segments:

- **Validate face-on appearance before inference.** A cheap cropped render's
  mid-layer must look like a single sheet's surface. Quantitatively: the
  in-segment dark-gap fraction separates valid sheets from cross-sections by
  ~8× (0.013–0.017 vs 0.105–0.127 in our data).
- The failure is testable upstream: grown surfaces should sit on their own
  input prediction (ours hit it at only 6–10 % of mesh vertices). Filed with
  our other reports.

The scan team's published Oct-2025 1203 segments pass both checks — figures
`fig5`/`fig6` show the failure and the pass side by side. Figures for all
headline results are in `figures/`.

## Recommended next steps (ours and anyone's)

- **Projection-space acquisition simulation**: rung 7 shows image-space
  statistics matching (PSF energy + histogram) is insufficient — the next
  fidelity level is simulating the 116 keV / 1.2 m propagation in projection
  space (phase-contrast fringes, correlated reconstruction noise) before
  re-running the transfer test. If transfer then works, pseudo-labeling on
  1447 unlocks.
- **Multi-scroll co-training** (the organizers' named ask) — done above
  (revision §3); the open follow-ups are surface-quality verification on the
  eligible scrolls and widening the labeled-chemistry span beyond 6 scrolls.
- **Train longer.** The single cheapest community lever: the scaling curve is
  unsaturated at 7500 it/frag; published cross-scroll baselines understate
  what existing corpora already support.
- The rung-4 recipe (HF resnet → koine wrapper → 20-min fine-tune) is cheap
  enough to be a standing regression probe for any new scroll/resolution.

## Tooling notes (upstream bug reports filed separately)

1. `vc_render_tifxyz --zarr-output` segfaults deterministically (any n, local
   or remote volume); `--tif-output` works. Full matrix + repro in
   `docs/vc3d-bug-reports.md`.
2. Local (non-streaming) zarr volume reads silently return fill-value: the
   chunk-key computation probes wrong indices (strace evidence in the same doc).
   The `--remote-url` path is correct.
3. koine-machines gotchas documented for reproducers: patch-cache invalidation
   after dataset changes, `decoder_upscale` needed for resnet3d in flat-mode
   training (one-line patch included), HF `resnet50_7.9um_scroll1_frags`
   state-dict is drop-in compatible with `ResNet3DSegmentationModel` —
   wrapping is just saving it as `{"model": state_dict}` with the unused
   1139-class `fc` head dropped (the loaders use `strict=False`).

## Reproducibility

- `scripts/batch_probe_1447.sh` — render → pack → infer for every published segment
- `scripts/downsample_to_sim86.py` — simulated-8.6 µm dataset builder
- `scripts/infer_resnet_legacy.py` — GP-era-normalization inference driver
- `configs/ink_sim86_resnet_ft.json` — the 20-minute fine-tune
- **Training runs — one table per model** (question served, data, architecture,
  schedule, wall time, checkpoint used and why it was chosen):
  [docs/training-runs.md](docs/training-runs.md). Checkpoint selection was
  either *final* or, in the one case where a save-interval mistake left no
  final checkpoint, *last surviving* — never chosen by scoring candidates
  against a target.
- Full experiment log with dates and dead ends: `DESIGN.md` in the working repository (not mirrored here; this repository carries the submission and its reproduction path only)

## Four late-August results

Both belong to the next cycle — nothing above has been revised — but they
change how the batch negative should be read.

**The pipeline has since been validated end-to-end on a real, read scroll.**
Every control above is a fragment or a simulation; none is a text-bearing
scroll held out from training. We closed that gap by probing Scroll 5
(PHerc 172, read by the community in 2025) with the same models and pipeline.
Both model lineages produce row-organized, character-scale response — 50 to
100 times the activation level of any eligible-scroll probe — co-located with
the community's own prediction of the same segment: correlation rises from
0.12 at 0.25 mm blocks to 0.40 at 4 mm, while every flipped alignment sits at
zero. Sanity controls pass at 5.3–5.8 %. Letters are smeared rather than
legible, which we attribute to the published layers being JPEG-compressed and
to a z-spacing mismatch, not to detection failure (`figures/fig7`, `fig8`).
The consequence: the silence reported here on PHerc 1447, 1218, 0800, 0268
and 1203 comes from an instrument that demonstrably reads real in-scroll ink.
It is evidence about those scans and inks, not about our methods.

**The ink-density families are not a beam-energy artifact.** That finding was
gated on a multi-energy control, which has now run. PHerc 51 Frag 6 was
scanned at 53, 70 and 88 keV; measured on the same surface with the same
labels, ink-versus-background contrast is flat within noise across all three
(Cohen's d = +0.031, −0.029, −0.029) and carbon-like at every energy. The
decisive comparison: PHerc 1667's dense-family signature was measured at
70 keV, and PHerc 51 at 70 keV shows nothing — beam energy does not
manufacture the dense signature. The families reflect the objects. Caveats:
one fragment carries the energy control, and the dense-family fragments have
no multi-energy scans of their own.

**Choosing training data by ink family does not work.** The natural
application of the ink-family finding is to pick training scrolls from the
target's own family. We tested it directly — two held-out targets, one per
family, each probed by a same-family and a cross-family training pair at
matched budget (two fragments, 15 000 iterations per condition) — and family
membership does not predict transfer. On the dense-ink target the same-family
pair *lost* by 0.154 AUC; on the carbon target it won by 0.300, because one
training set won both targets. Data volume does not explain it: the set with
the most labeled ink (20.2 M pixels) lost, and the winner carries less than
half of that. Label *density* separates them — labeled ink as a fraction of
the supervised region, 0.12–0.18 for the winner against 0.06–0.09 for the
losers — which follows from the patch sampler requiring 5 % labeled coverage.
Density and family are perfectly aligned in this corpus, so the two cannot be
fully separated here. The recommendation that survives is duller than the one
we set out to test: rank candidate training fragments by label density, not
by ink-contrast family. An earlier version of the study report called the
clusters "exploitable" for training-scroll selection; that claim is retracted
there.

![The selection heuristic, tested and refuted. Same-family training loses on the dense-ink target and wins on the carbon target, because one training set wins both.](figures/fig16_family_selection_refuted.png)

**Image-space acquisition differences cannot explain the batch silence, and
the statement now carries a margin.** Readable Scroll 5 data at 9.36 µm
already sits at the eligible batch's high-frequency content (gradient energy
5.15 against their 5.8) and reads perfectly well. Blurred further, reading
survives to 2.96 — about half the eligible level, where co-location with the
community's read actually peaks — and collapses only at 1.80, roughly a
third of it. Image sharpness would have to halve again from the eligible
batch's own value before this pipeline goes silent. What remains, and what
image-space simulation cannot reach, is ink chemistry outside the classic
corpus and propagation physics at 1.2 m.
