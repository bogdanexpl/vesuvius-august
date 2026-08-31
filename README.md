# Ink at 8.6 µm: transfer diagnostics, a cross-scroll benchmark, and the budget confound

*Vesuvius Challenge progress submission, August 2026. Experiments index: [docs/experiments-index.md](docs/experiments-index.md). Full study: [docs/study-2026-08-cross-scroll-ink-transfer.md](docs/study-2026-08-cross-scroll-ink-transfer.md). Upstream bug reports: [docs/vc3d-bug-reports.md](docs/vc3d-bug-reports.md). Figures: [figures/](figures/).*

# Ink at 8.6 µm: a transfer-diagnostics ladder on PHerc. 1447

*Progress-prize submission draft — August 2026 (revised 2026-08-25: adds the
LOSO benchmark, the training-budget scaling result that re-scopes the ladder's
conclusion, and the budget-adequate 1447 probe). Code: this repo (`scripts/`,
`configs/`). All experiments on one RTX A6000 (48 GB), data streamed from the
open-data S3 bucket. Full PhD-style report:
`docs/study-2026-08-cross-scroll-ink-transfer.md`.*

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
| 9 | rung-7 model, zero-shot | **46 self-grown segments across three 8.64 µm scrolls** (PHerc 1447 published + our own headless `vc_grow_seg_from_seed` segments on 1447/1218/0800/0268; multiple z-bands, inner/mid/outer windings) | uniformly quiet near-zero predictions (0.3–1.1 % strong activations, edge artifacts only), zero letterforms anywhere |
| 10 | **the decisive control** — proven 2.4 µm Scroll-1 model, zero-shot | **PHerc 1203 at its native 2.4 µm scan** (same protocol class as the training scan; 4 self-grown segments) | tile-level noise, no letterforms; row-pitch statistics inconsistent across segments (blob noise, not lines). **Cross-scroll transfer fails even with resolution and protocol held equal.** |

Key images (in `runs/` of the data volume):
`ink_sim86_resnet_ft/predictions/overview_div6.png` (rung-4 Greek),
`.../overlay_mask_div6.png` (red = the only trained regions),
`1447_probe/batch/contact_sheet.png` (rung-6b, all segments),
`1447_probe/offset_sweep/sweep_sheet.png` (rung-6a).

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
   open-problems post's §7, now with a controlled experiment behind it.
4. **The 8.64 µm batch silence (rung 9) is therefore primarily a chemistry/
   domain problem.** A model that provably reads real 7.9 µm Scroll 1 (rung 8)
   and simulated 8.6 µm (rung 4) finds nothing on 46 segments spanning four
   sampling campaigns across PHerc 1447, 1218, 800 and 268 — multiple z-bands,
   inner to outer windings — and the rung-10 control shows the same silence at
   native 2.4 µm on 1203. Acquisition physics may still contribute at 8.64 µm,
   but it is not the primary blocker; a test scan of a known-ink object under
   the 8.64 µm protocol would settle its share (a question for the scan team).
5. **Per-segment ink presence on these scrolls is currently unjudgeable.** Any claim of
   "this segment is blank" would be unsupported until the acquisition gap is
   modeled.

## Revision (2026-08-25): the budget confound, the benchmark, and the adequate probe

Work after the ladder materially refines conclusions 3–4 above.

1. **First public cross-scroll ink benchmark (LOSO).** 8 IR-labeled fragments
   from 6 physical scrolls (public, `scripts/exp4_fetch_frags.py`), trained
   leave-one-scroll-out + full 4×4 pairwise matrix. Structure: Paris1↔PHerc51
   form a tight transfer cluster at ceiling; PHerc 1667 is isolated near
   chance in both directions.
2. **Every cross-scroll number is a training-budget lower bound.** Crossing
   corpus size × budget shows held-out AUC is monotone in *iterations per
   training fragment* (r = +0.98): .548 → .868 from 714 → 7500 it/frag,
   log-linear **+0.101 AUC per doubling**, unsaturated. At matched budget the
   cross-scroll penalty is only **−0.02…−0.04 AUC** (~95–97 % of
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
   segment read *quieter* still (0.2 %), and the first budget-adequate probe
   of PHerc 1218/0800/0268 (six ~47 cm² segments) returned 0.08–0.34 % —
   below every 1447 value. **Every scroll of the 8.64 µm batch is silent
   under a validated model with a passing positive control, and response
   decreases as surface quality improves.** The published `layers_ink`
   predictions shipped with the segments also show no letterforms, so no
   existing model reads them either. The cause is batch-level: the shared
   116 keV / 1.2 m protocol and/or chemistry common to these four scrolls
   but absent from the classic corpus. This is the strongest form of the
   §6 diagnostic the open-problems post asked for — and it makes the
   known-ink phantom scan under the 8.64 µm protocol the single most
   valuable experiment now available to the community (a question for the
   scanning team, not for model builders).
4. **Ink density families (exploratory, cause unproven).** Cohen's d of
   ink-vs-background intensity splits the corpus into a carbon-like family
   (Paris + PHerc 51, |d| ≤ 0.1) and a denser-ink family (1667/343P/500P2,
   d ≥ +0.19) that tracks the transfer clusters — but beam energy is
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
headline results are in `docs/figures/`.

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
   state-dict is drop-in compatible with `ResNet3DSegmentationModel`.

## Reproducibility

- `scripts/batch_probe_1447.sh` — render → pack → infer for every published segment
- `scripts/downsample_to_sim86.py` — simulated-8.6 µm dataset builder
- `scripts/infer_resnet_legacy.py` — GP-era-normalization inference driver
- `configs/ink_sim86_resnet_ft.json` — the 20-minute fine-tune
- Full experiment log with dates and dead ends: `DESIGN.md` in the working repository (not mirrored here; this repository carries the submission and its reproduction path only)

## Addendum (2026-08-31): two results that landed after this write-up was frozen

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
