# Cross-Scroll Ink-Detection Transfer in Herculaneum Papyri: a Controlled Ladder Study

*Working report, v2 — August 2026 (v1 2026-08-20; v2 2026-08-25 adds §§9–12:
the training-budget confound that re-scopes the ladder's conclusion, the ink
density families, the budget-adequate PHerc 1447 probe, and the pre-registered
surface-placement experiment). Single-investigator study on one RTX A6000
(48 GB), all data streamed from the Vesuvius Challenge open-data S3 bucket.
Experiment log with timestamps and dead ends: `DESIGN.md` (working repository, not mirrored here); condensed
prize-submission version: `docs/aug2026-progress-submission.md`.*

---

## Abstract

The 13 Grand-Prize-eligible Herculaneum scrolls were scanned in 2025 at
8.64/9.36 µm, while every publicly released ink-detection model was trained at
2.4–7.9 µm on Scrolls 1–5 or fragments. We ran a ten-rung controlled transfer
study to isolate which factor blocks ink recovery on the eligible scrolls:
resolution, acquisition physics, pipeline artifacts, or scroll identity
(ink chemistry / preservation). We find that (i) resolution is **not** the
blocker — a 20-minute fine-tune on resolution-degraded labeled data reads
simulated 8.6 µm text, generalizing well outside its supervision mask;
(ii) the same cheaply fine-tuned model transfers to a **real** 7.9 µm
acquisition of the same scroll, zero-shot; (iii) despite this, 50 segments
sampled across four 8.64 µm-batch scrolls yield no letterforms; and (iv) in
the decisive control, PHerc 1203 — which uniquely has a native 2.4 µm scan
under the same protocol as the reference training data — is equally silent.
We conclude that **cross-scroll domain shift dominates all other factors**,
turning the community's open problem §7 ("models fail across scrolls") from
an observation into a controlled result. We release the full headless
pipeline (segment growing, streamed rendering, probing at ~10 min/segment)
plus two reproducible upstream VC3D bug reports.

## 1. Motivation

- The 2027 Grand Prize and all ten First Letters prizes are payable only on
  scrolls scanned at 8.64/9.36 µm. Whether ink is even detectable at those
  resolutions/protocols was explicitly listed as unproven by the organizers
  (2026 Open Problems, §6), who requested "diagnostics distinguishing *no
  signal* from *signal present but unrecovered*".
- Every prior read (Scrolls 1–5) used 2.4–7.9 µm data. No systematic probe of
  the eligible batch existed publicly as of 2026-08.
- Strategy context (this project): a rigorous answer is progress-prize
  material regardless of sign, and directs our First Letters investment.

## 2. Materials

**Data (all public).**
- PHerc 1447 / 1218 / 0800 / 0268: 8.64 µm masked OME-Zarr volumes + m7
  surface-prediction zarrs (S3). 1447 additionally ships 14 published
  `auto_grown` segments.
- PHerc 1203: both a 9.36 µm scan and a native **2.4 µm** scan with surface
  predictions for each (the only eligible scroll with a same-protocol
  hi-res control).
- Scroll 1 (= PHerc.Paris.4 — naming trap): labeled 2.4 µm ink-tutorial
  dataset (w00 segment, labels on one z-plane), and the original **7.9 µm**
  volpkg layers from dl.ash2txt (rung 8).

**Models.**
- `ink_tutorial` ckpt: 2.5D `vesuvius_unet` (64-ch flat mode) trained by us
  (Phase 0) on the 2.4 µm Scroll 1 tutorial data; verified to read w00 Greek.
- `scrollprize/resnet50_7.9um_scroll1_frags` (HF): Resnet3D-50, 18-layer
  window, trained on 7.9 µm Scroll 1 + fragments. Its safetensors are
  shape-identical to koine's `ResNet3DSegmentationModel` (unused fc head
  aside) — we wrapped it into the koine checkpoint format.
- Our fine-tunes of the above (rungs 4/7) on simulated-8.6 µm data.

**Infrastructure.** Headless, one A6000: band-wise sparse pulls of surface
predictions; `vc_grow_seg_from_seed` (Docker `:edge`) for segment growing
(~4 min / 200 generations); `vc_render_tifxyz --remote-url … --tif-output`
for S3-streamed 65-layer surface volumes; TIFF→zarr repack; koine
`infer` for probing. Scripts: `scripts/pull_band.py`, `pick_seeds.py`
(radial-tier seeds), `grow_probe_sweep.sh`, `tifs_to_zarr.py`,
`downsample_to_sim86.py`, `build_sim86v2.py`, `infer_resnet_legacy.py`.

## 3. Method: the ladder

Design rule: each rung changes **one variable** relative to an anchored
positive control, so any sign flip is attributable. Positive anchors are
re-verified at every model change ("does it still read Scroll 1 text?").

| rung | variable isolated | setup | result |
|---|---|---|---|
| 1 | resolution gap, naive | 2.4 µm UNet on native 8.6 µm 1447 render | patch noise — negative |
| 2 | sampling match | z-step 0.278 (±78 µm depth) + 3.6× bicubic upscale | checkerboard noise — input rescaling insufficient |
| 3 | closer teacher | 7.9 µm resnet, zero-shot, native render | fiber-coherent, no letters |
| 3b | preprocessing | same, GP-era clip-200/255 normalization | dead flat — negative not a normalization artifact |
| 4 | **is 8.6 µm learnable at all?** | fine-tune 7.9 µm resnet 20 min on simulated-8.6 µm Scroll 1 (3.6×↓, 65→18 z) | **legible Greek rows; 78 % of recovered ink outside the supervision mask** |
| 5 | sim→real 1447 | rung-4 model on real 1447 | letter-scale blobs, no letters |
| 6a | surface offset | z-window sweep 5–23 … 41–59 | identical at all depths — offset ruled out |
| 6b | segment coverage | all 14 published 1447 segments | uniform honeycomb false positives, no letters |
| 7 | acquisition statistics | measured real-vs-sim gap (2.3× high-freq energy, histogram shift); rebuilt sim with PSF blur + quantile LUT; re-fine-tuned | sim sanity holds; real 1447 unchanged — 2nd-order matching insufficient |
| 8 | **sim→real, same scroll** | rung-7 model on the real 7.9 µm scan of the same Scroll 1 segment | **reads the same Greek rows** — recipe transfers to real acquisitions |
| 9 | scroll coverage | 46 self-grown segments over 1447/1218/0800/0268, 6 z-bands, inner→outer windings | uniformly quiet near-zero (0.3–1.1 % strong activations), no letters |
| 10 | **scroll identity, controlled** | proven 2.4 µm model on PHerc 1203's native 2.4 µm scan (protocol matched to training) | noise; row-pitch statistics inconsistent — **cross-scroll transfer fails with resolution AND protocol held equal** |

## 4. Results

1. **Resolution is bridgeable.** Rung 4: after a 20-minute fine-tune on
   resolution-degraded labeled data, text at 8.6 µm-equivalent sampling is
   clearly legible, and 78 % of recovered ink lies outside the training
   supervision — generalization, not memorization.
2. **The recipe survives sim-to-real.** Rung 8: the same fine-tuned model
   reads the *real* 7.9 µm acquisition of the same scroll zero-shot.
3. **The eligible batch is silent anyway.** Rung 9: 50 total segments across
   four 8.64 µm scrolls produce no letterforms; responses are *quieter* on
   clean sheets (1218/0800/0268) than on 1447 (fold-textured), indicating the
   model discriminates surface quality rather than hallucinating.
4. **Scroll identity is the master variable.** Rung 10: 1203's native 2.4 µm
   scan — same protocol family as the reference training data — is equally
   unreadable by a model proven on Scroll 1 at that exact resolution.
5. Combined: Scroll 1 reads at 2.4 µm, 7.9 µm, and simulated 8.6 µm; other
   scrolls read at none of those. **Ink chemistry / preservation state
   (cross-scroll domain shift) dominates resolution and acquisition
   physics.**

## 5. Threats to validity

- *Sampling:* 50+4 segments across 5 scrolls, but bands are sparse; blank
  regions (margins, inner windings) could contribute to individual negatives
  — not plausibly to all of them.
- *Model family:* both reference models descend from Scroll 1 training; a
  chemistry-robust architecture could exist. That is precisely the surviving
  research direction, not a refutation of the diagnosis.
- *Surface accuracy:* self-grown segments use the m7 predictions; rung 6a
  rules out global offset, but locally poor surfaces reduce sensitivity.
- *Simulated 8.6 µm ≠ real 8.6 µm protocol* (116 keV/1.2 m vs 78 keV/22 cm
  reconstruction): rung 7 matched second-order statistics only. A known-ink
  phantom scanned under the eligible protocol would bound the residual
  acquisition share (recommended to the scan team).

## 6. Conclusions

The Vesuvius Challenge's binding constraint for the 2027 prizes is not scan
resolution but **cross-scroll ink-detection transfer**. Any First Letters
attempt on the eligible batch that reuses Scroll-1-derived ink models — at
any resolution treatment — is expected to fail. Community effort should
prioritize chemistry-robust ink detection; resolution adaptation is a solved,
cheap add-on (20-minute recipe, this repo).

**Addendum (2026-08-25).** §9 materially re-scopes this conclusion. Every
negative rung above used models in what we later measured to be the
under-trained regime (§9.3: AUC still rising at +0.101 per doubling of
per-fragment training exposure). At matched, adequate budget the cross-scroll
penalty shrinks to −0.02…−0.04 AUC (§9.4) — small, real, and an order of
magnitude below what the 5k-iteration numbers implied. "Cross-scroll domain
shift dominates" survives as the diagnosis *for under-trained models* — which
included every publicly released model as of 2026-08. The corrected statement
of the binding constraint is: **training budget first, chemistry residual
second.** The ladder's factual results (rungs 1–10) stand; their
interpretation is refined by §§9–11.

## 7. Follow-up: the LOSO cross-chemistry benchmark (Exp 4, first results)

Using the public fragment corpus (6 classic fragments from 4 physical scrolls,
packed reproducibly by `scripts/exp4_fetch_frags.py`), we trained
leave-one-scroll-out folds (flat 64-ch UNet, 5k iterations each;
`scripts/exp4_loso.sh`) and scored held-out fragments
(mask-restricted AUC / best-threshold Dice, `scripts/exp4_score.py`):

| held-out fragment | cross AUC | cross Dice | within-scroll ceiling (AUC/Dice) |
|---|---|---|---|
| paris1 frag3 | 0.614 | 0.257 | — |
| paris1 frag4 | 0.636 | 0.239 | 0.736 / 0.336 |
| paris2 frag1 | 0.677 | 0.381 | — |
| paris2 frag2 | 0.653 | 0.350 | 0.675 / 0.369 |
| pherc1667 frag5 | 0.509 | 0.159 | — |
| pherc51 frag6 | 0.619 | 0.243 | — |

First public quantification of cross-scroll ink transfer. Read: (i) transfer
is weak-positive, not zero — there is signal to amplify; (ii) chemically
related scrolls (Paris collection) transfer nearly at their within-scroll
ceiling; (iii) PHerc 1667 sits at chance — chemistry clusters exist.
Caveat: 5k-iteration budgets limit both rows; v2 will extend budgets, add the
remaining labeled sources (0343P, 0500P2, 0009B), and vary architecture.

**Pairwise transfer matrix.** Extending to single-scroll models evaluated on
every other scroll's held-out fragment (mask-restricted AUC; `pw_*` runs,
`runs/exp4_loso/pairwise_auc.json`):

| train ↓ / test → | Paris 1 | Paris 2 | 1667 | PHerc 51 |
|---|---|---|---|---|
| Paris 1 | *0.736* | 0.668 | 0.612 | 0.735 |
| Paris 2 | 0.686 | *0.675* | 0.558 | 0.675 |
| 1667 | 0.547 | 0.537 | — | 0.595 |
| PHerc 51 | 0.741 | 0.629 | 0.627 | — |

Italic = within-scroll (different fragment). Read: Paris 1 ↔ PHerc 51 form a
tight cluster (≈0.74 both directions, at Paris 1's own ceiling); PHerc 1667 is
isolated in both directions; Paris 2 receives moderately from all but exports
less. Chemistry clusters are therefore real. **Whether they are *exploitable* for
training-set selection was tested directly after this submission was prepared,
and the answer is no**: a matched A/B (two held-out targets, one per family,
15 000 iterations each) found that selection by family lost — the carbon-family
training pair beat the dense-family pair on the dense-family target by 0.154
AUC, while winning its own target by 0.300. One training set won both. Label
density (labeled ink as a fraction of the supervised region) separates the sets
where family does not, though this corpus cannot fully disentangle the two.
An earlier version of this paragraph claimed exploitability; that claim is
retracted. Interactive figure: "Herculaneum Transfer
Map" (project artifact).

## 8. Next phase (started)

1. **Domain-robust features:** probe `scrollprize/ink_3d_dino_guided` and
   3D-DINO feature spaces on our render corpus — self-supervised features may
   carry ink contrast that supervised Scroll-1 heads discard.
2. **Direct 3D ink morphology** (open problems §6c): unsupervised anomaly
   probing on recto surfaces at letter scale, no cross-scroll labels needed.
3. **Fragment leverage:** fragments with IR-verified ink labels come from
   multiple physical scrolls — the only existing cross-chemistry labeled
   data. Inventory + co-training candidate.
4. **Phantom scan request** to the scanning team (bounds acquisition share).

*(Historical note: items 1–2 were attempted and parked — Exp 1 gave fiber
false-positives, Exp 2 was invalidated by a state-dict loading bug, both
logged in DESIGN.md. Item 3 became Exp 4 (§7) and everything after. §§9–12
supersede this list.)*

## 9. The training-budget confound (Exp 4 v2 → scaling law)

### 9.1 An apparent negative

Extending the LOSO corpus from 4 to 6 scroll-groups (adding PHerc 0343P and
0500P2; 8 fragments, 34 GB) *dropped* macro AUC from 0.618 to 0.555, with
paris2 collapsing to chance (0.480–0.500 vs 0.653–0.677 in v1). Naive read:
pooling diverse chemistries hurts. Confound: total budget was held at 5k
iterations while training fragments increased, so per-fragment exposure fell
~40 %.

### 9.2 The 2×2 disentanglement

Holding the held-out fragment fixed (paris1_f3) and crossing corpus size ×
budget:

| | 5k iters | 15k iters |
|---|---|---|
| **3 scrolls** | .614 (1250 it/frag) | **.827** (3750 it/frag) |
| **5 scrolls** | .548 (714 it/frag) | .731 (2143 it/frag) |

AUC is monotone in **iterations per training fragment** (r = +0.98), not in
corpus size or total budget separately. Corpus dilution at fixed total budget
costs −0.066 (@5k) and −0.096 (@15k) — real at both budgets.

*Errata, kept for the record:* mid-analysis we first called v2's drop "a
budget artifact, refuted" (after the 15k recovery) — premature; the fourth
cell showed dilution is real. We also briefly claimed the cross-scroll
penalty "collapses to ≈0" — overstated; §9.4 gives the measured residual.

### 9.3 Scaling curve — unsaturated

paris1_f3 AUC vs iterations-per-training-fragment: 714 → .548, 1250 → .614,
2143 → .731, 3750 → .827, 7500 → **.868** (f4 mirrors: .534/.636/.723/.794/
.857). Log-linear fit **+0.101 AUC per doubling** (r = 0.982); the last
doubling still bought +0.041. Nothing in this benchmark — or, we argue, in
the community's cross-scroll reports — has been trained to saturation.

### 9.4 Budget-matched penalty

Within-scroll "ceilings" re-measured at 15k: paris1_f4 .814 (was .736 @5k),
paris2_f2 .685 (was .675). Budget-matched cross-scroll penalty (best cross −
within@15k): **−0.020 and −0.040**. And the cross-scroll 30k model (.868)
*exceeds* the within-scroll 15k ceiling (.814): "within-scroll ceiling" is a
budget coordinate, not a ceiling.

### 9.5 Consequence

Every cross-scroll transfer number reported publicly (including our §7) is a
**lower bound set by training budget**, not by chemistry. The chemistry
residual is real (consistently negative, −0.02…−0.04) but an order of
magnitude smaller than the under-trained regime suggests. The eligible-scroll
probes of rungs 5–9 were therefore not decisive; the decisive probe needs
≥7500 it/frag multi-scroll training — that is Exp 6 (§11).

## 10. Exp 5: per-fragment ink density contrast — two families, cause unproven

Cohen's d of ink-labeled vs background voxel intensity (best layer, inside
the supervision mask; `scripts/exp5_ink_contrast.py`): paris1 −0.103/+0.063,
paris2 −0.183/+0.048, pherc51 −0.099 | 1667 **+0.268**, 343P **+0.428**,
500P2 **+0.195**. Two families: carbon-like (|d| ≤ 0.1: Paris + PHerc 51) and
denser-ink (d ≥ +0.19: 1667/343P/500P2). The tight transfer cluster
paris1↔pherc51 (§7) shares a near-identical signature; the isolated node 1667
sits in the other family. The correlation with transfer (r = −0.651, n = 6)
is entirely driven by 1667 — this explains the outlier, not a general law.

**Confounds and prior art.** Beam energy differs per fragment (53–70 keV,
343P/500P2 unknown) and attenuation is energy-dependent; the clean test
(PHerc 51's 53/70/88 keV triple scan) needs surface processing we have not
done. Known prior art: carbon ink ≈ papyrus density (foundational); lead in
two Institut-de-France fragments (Brun et al., PNAS 2016) — which are in our
*low*-contrast family, a tension arguing against naive "metal vs carbon"
labels. We therefore report **a density-contrast difference of unproven
cause**, plausibly novel only as a uniform corpus-wide measurement aligned
with transfer performance. Side result: the blank-scroll hypothesis is dead
for the benchmark — 1667 is Scroll 4, fully read by the VC team, yet our
worst transfer target.

## 11. Exp 6: the budget-adequate probe of PHerc 1447 — negative, with a passing control

**Setup.** Training corpus `frags_sim86`: 7 fragments / 5 scrolls degraded to
8.64 µm in-plane (area-mode; ink fractions preserved exactly;
`scripts/exp6_build_sim86_corpus.py`; 500P2 excluded — OOM, §Appendix A).
Target renders: 3 published 1447 segments re-rendered at `--slice-step 0.375`
so both domains sit at xy 8.64 µm / z 3.24 µm. Training: 52.5k iterations
(8h09m) — nominally 7500 it/frag, the §9.3 operating point. *Disclosed
config error:* `save_every 15000` with 52 500 total ⇒ no final checkpoint;
best surviving is ckpt_045000 (6428 it/frag).

**Results.** On 1447: frac>200 = 0.6–2.2 % (30k) and 0.5–1.8 % (45k);
letterform sheets (density-picked best 11 mm crops, `scripts/
letterform_sheet.py`) show speckle — no closed loops, no stroke-width
consistency, no baseline alignment.

**Control.** The same ckpt_045000 on a sim-8.64 µm 1667 fragment produces
clear Greek letterforms in rows (frac>200 = 4.6 %). Model, resolution
treatment and depth matching all work; 1447 is where the pipeline fails.
(Caveat: that fragment was in training — a capability control, not
generalisation; held-out generalisation is separately measured at .868, §9.3.)

**Interpretation.** This is, to our knowledge, the strongest negative on the
eligible batch: multi-scroll, budget-adequate, resolution- and depth-matched,
with a same-model positive control. Surviving explanations, ranked: (a)
**virtual-unwrapping surface error** — the training fragments are physically
exposed surfaces with IR-clean labels, while every 1447 surface descends from
m7 predictions; if the traced surface sits off the ink layer, no detector can
recover unsampled ink; (b) 1447 ink chemistry outside the 5-scroll corpus
span (the §9.4 residual is small *within* the classic corpus, but 1447 may
sit outside it); (c) genuinely blank probed regions. (a) is testable without
new labels — Exp 7.

## 12. Exp 7 — pre-registered 2026-08-25, before execution: surface-placement and segmentation-generation test on PHerc 1447

*This section was written and committed before any Exp 7 compute ran;
results will be appended, not edited in.*

**Hypotheses.**
- **H7a (placement offset):** the m7-derived 1447 surfaces are systematically
  offset along the sheet normal, so Exp 6's 65-layer slab (±105 µm) missed
  the ink-bearing layer.
- **H7b (segmentation generation):** July-2025 `auto_grown` surfaces are too
  inaccurate in shape (not merely offset); a newer-generation segmentation
  succeeds where they fail. Newly found on S3:
  `20251105093211-z_dbg_gen_00320` (Nov 2025, five months newer, bbox
  spanning ~8.2 cm of scroll axis — larger than any July segment).

**Why rung 6a does not already answer this.** The earlier offset sweep used a
single-scroll (Scroll-1-derived), under-trained model that rung 10 proved
cannot read *any* other scroll — it could not have found letters at any
offset regardless of surface quality. Exp 7 is the first offset test with a
model that passed a same-batch positive control (ckpt_045000, §11).

**Method.**
1. Re-render segments `auto_grown_20250703034159599`,
   `auto_grown_20250703025628283`, and `z_dbg_gen_00320` from the tifxyz
   meshes at `--slice-step 0.375`, but with **n = 193 slices** (span ±311 µm
   about the traced surface, vs ±105 µm in Exp 6).
2. Slide the model's 65-layer window through each render: starts
   k ∈ {0,16,32,48,64,80,96,112,128} ⇒ normal offsets
   {−207,−156,−104,−52,0,+52,+104,+156,+207} µm (window centre relative to
   surface; 65-layer window itself spans 211 µm, so coverage is continuous).
3. Infer each window with Exp 6's ckpt_045000 (`--layer-start k
   --layer-end k+65`); no other pipeline change.

**Metrics.** Per window: frac>200 and activation std (response-vs-offset
curve); letterform sheet for each segment's max-response window; panels to
the judging artifact for human inspection.

**Controls.** (i) The k = 64 (0 µm) window on the July segments must
reproduce Exp 6's negative (speckle, frac>200 ≤ ~2 %) — internal consistency
of the re-render. (ii) A peaked response curve at a common non-zero offset
across segments is itself evidence of placement bias even absent letterforms.

**Decision rules (stated in advance).**
- **P1:** letterforms at some |offset| > 0 on July segments ⇒ H7a confirmed;
  fix is a render-offset correction; re-probe all 15 segments at the found
  offset.
- **P2:** letterforms on `z_dbg_gen_00320` but not the July segments at any
  offset ⇒ H7b confirmed; the blocker was segmentation quality, not ink
  detection; switch all 1447 work to newest-generation surfaces.
- **P3:** all negative across ±207 µm and both generations ⇒ constant-offset
  placement error excluded; H7a survives only in its undulating form (surface
  wandering off-layer non-uniformly — testable only with a manually verified
  surface in the VC3D GUI, which requires an interactive session), and (b)
  chemistry / (c) blank regions gain weight.

**Budget.** ~6 GB disk (3 renders × 193 layers), ~1–2 h streaming render,
27 GPU inference runs (~1–2 h). Well within current headroom (32 GB free).

### 12.1 Results (appended 2026-08-25, after execution; pre-registration above unedited)

Response-vs-offset (frac of rendered pixels with prediction > 200), ckpt_045000:

| offset µm | −207 | −156 | −104 | −52 | 0 | +52 | +104 | +156 | +207 |
|---|---|---|---|---|---|---|---|---|---|
| seg A (034159599) | .0128 | .0124 | .0152 | **.0251** | .0186 | .0166 | .0183 | .0203 | .0110 |
| seg B (025628283) | .0091 | .0099 | .0055 | .0094 | .0073 | .0141 | .0119 | **.0194** | .0189 |

z_dbg_gen_00320 (Nov-2025 generation, 84 cm², centred): **.0022** — the
quietest response ever recorded on 1447. Full curves:
`runs/exp7_surface/exp7_scores.json`.

- **Consistency control (i) passed:** the 0 µm windows reproduce Exp 6's
  values on the fresh 193-layer renders (1.9 % / 0.7 %, inside the 0.5–2.2 %
  Exp 6 range).
- **Control (ii) negative:** response peaks exist at non-zero offsets but at
  *discordant* offsets and signs across segments (−52 µm vs +156 µm) with
  mild amplitude (≤2.6×, all values in the speckle regime) — noise, not a
  common placement bias.
- **Letterform judgment (the pre-registered criterion):** density-picked
  crops of each segment's peak window and of the Nov segment show speckle
  and fiber-direction streaks — no closed loops, no stroke-width
  consistency, no baseline rows (panels added to the judging artifact).

### 12.2 Verdict: P3

- **H7a rejected** for constant offsets: no letterforms at any window centre
  in ±207 µm (continuous slab coverage given the 211 µm window span).
- **H7b rejected:** the newer, ten-times-larger segmentation reads *quieter*,
  not better. This mirrors the rung-9 observation (cleaner sheets → less
  false-positive texture): the July segments' higher response was fold/
  texture artifact, and a better-traced surface removes it without revealing
  ink.
- Surviving explanations, re-ranked: **(b) 1447 ink chemistry outside the
  corpus span** is promoted to prime suspect; (a′) undulating surface error
  (surface wandering off-layer non-uniformly) remains possible but is now
  the only surface-shaped variant left, testable by eye in the VC3D GUI
  (via a local VC3D GUI deployment); (c) blank regions remain
  possible per-segment but strain to cover 84 cm² of the Nov surface plus
  everything else probed.

### 12.3 Post-mortem

The pre-registration held up operationally: decision rules covered the
observed outcome without amendment, and both controls did their job. One
scope note for the record: the sweep tests *rigid* placement error only —
an undulating error with amplitude comparable to the sheet spacing defeats
every fixed offset simultaneously. We judged that acceptable in advance
(P3 explicitly names it) rather than discovering it afterwards. Cost:
~4 h wall, ~6 GB disk, 19 inference runs — the cheapest experiment in the
campaign per hypothesis killed.

## 13. Exp 8 — pre-registered 2026-08-25, before execution: is the budget-adequate negative 1447-specific or batch-wide?

*Committed before any Exp 8 compute ran; results will be appended.*

**Question.** Exp 6/7 established a validated-model negative on PHerc 1447.
Rung 9's negatives on the rest of the 8.64 µm batch (1218/0800/0268) all used
under-trained single-scroll models, which §9 voids as evidence. This is the
first budget-adequate probe of those scrolls.

**Hypotheses.**
- **H8a (1447-specific):** other batch scrolls read — 1447 is the outlier
  (its chemistry/preservation outside the corpus span).
- **H8b (batch-wide):** all scrolls are negative under the validated model —
  the non-recovery is a property of the batch (acquisition protocol and/or
  shared chemistry), not of one scroll.

**Method.** Per scroll (1218, 0800, 0268): the two largest self-grown rung-9
segments (~47 cm² each, m7-seeded, grown 2026-08-18/19) rendered from S3 at
`--slice-step 0.375`, n = 65 (the exact Exp 6 protocol), inferred with
ckpt_045000. Metrics: frac>200 / frac>128 / std + density-picked letterform
sheets; panels to the judging artifact. Renders are deleted after inference
(disk headroom 27 GB); predictions and scores are retained.

**Controls.** The same checkpoint's sim-1667 positive stands as the
capability control; Exp 7 §12.1 established that this render+infer chain
reproduces Exp 6 numbers on fresh renders.

**Known confound, stated in advance.** These are m7-seeded auto-grown
surfaces; an undulating-surface failure would produce the same negative on
every scroll and is partially degenerate with H8b. Exp 7 bounds this: better
surfaces got *quieter*, and rigid offsets are excluded on 1447. If P2 (below)
obtains, one segment per scroll goes on the VC3D eyeball list alongside
1447's.

**Decision rules.**
- **P1:** letterforms on ≥1 scroll ⇒ the negative is 1447-specific; First
  Letters effort refocuses on the responsive scroll; the chemistry-span
  hypothesis sharpens (responsive scroll inside the corpus span, 1447
  outside).
- **P1b:** structured, clearly super-1447 response but no legible letters ⇒
  ambiguous; extend to more segments on that scroll before concluding.
- **P2:** all negative ⇒ batch-wide non-recovery at adequate budget with an
  in-pipeline positive control — the strongest form of the scan-team
  question (protocol vs batch chemistry); First Letters on this batch is
  de-prioritized pending their answer.

**Budget.** ~7 GB transient disk, ~3–5 h wall (6 streamed renders + 6
inference runs).

### 13.1 Results (appended 2026-08-26, after execution; pre-registration above unedited)

frac of rendered pixels > 200 (ckpt_045000; two ~47 cm² segments per scroll):

| scroll | seg 1 | seg 2 |
|---|---|---|
| PHerc 1218 | .0008 | .0012 |
| PHerc 0800 | .0016 | .0015 |
| PHerc 0268 | .0034 | .0020 |

All six are *below* every 1447 value from Exps 6–7 (0.5–2.5 %), and far below
the positive-control level (4.6 % with letterforms). Letterform sheets of the
loudest segment per scroll: sparse speckle following fiber/fold contours,
plus render-stripe artifacts on 1218 — no closed loops, no stroke-width
consistency, no rows (three panels added to the judging artifact). Scores:
`runs/exp8_batch/exp8_scores.json`.

### 13.2 Verdict: P2 — the negative is batch-wide

Every scroll of the 8.64 µm batch (1447 from Exps 6–7; 1218/0800/0268 here)
is now probed with the validated, budget-adequate multi-scroll model, and
every one is silent. H8a (1447-specific) is rejected. Note the ordering:
1447 — the scroll with the *worst* surfaces — is the loudest of the four,
and response decreases as surface quality improves, consistent with the
false-positive-texture reading from §12.

Combined interpretation across §§11–13: the non-recovery is a property of
the *batch* — its shared acquisition protocol (116 keV / 1.2 m) and/or
chemistry shared across these four scrolls but absent from the classic
corpus — and not of any single scroll, any segmentation generation, or any
surface offset. Per the pre-registered P2 rule: the scan-team question
(known-ink phantom under the 8.64 µm protocol; §5's recommendation, now
with the strongest possible evidential backing) leads the submission, and
First Letters investment on this batch is de-prioritized until it is
answered. The undulating-surface degeneracy noted in §13's confound clause
remains formally open batch-wide; the 1447 VC3D eyeball check applies to one
segment per scroll here too.

### 13.3 Post-mortem

Session-restart resilience mattered: the run died mid-batch and resumed
losslessly because predictions gate re-work (idempotent probe function);
the one flaw found was a cleanup `rm` under a root-owned parent mislabelling
successful inferences as INFER-FAIL in the log — cosmetic, fixed in the
resume script. Cost: ~4 h wall total across the interruption, disk peak
within guard. With Exps 6–8 the campaign has now spent three pre-registered
experiments converging on one question that only the scanning team can
answer — which is itself the result.

## 14. Exp 9 — pre-registered 2026-08-26, before execution: native-resolution multi-scroll probe of PHerc 1203 at 2.4 µm

*Committed before any Exp 9 compute ran; results will be appended.*

**Rationale.** All probing so far targeted the 8.64 µm batch. PHerc 1203 —
one of the nine 9.362 µm-batch eligible scrolls — uniquely also has a native
**2.403 µm / 0.2 m / 77 keV** scan (S3: `20260319130212-…`), i.e. the same
protocol class as every scan the community has ever read. Rung 10's negative
on it used a single-scroll model at ~1/10 adequate budget and is void under
§9. Probing it with an adequate multi-scroll model trained at **native
fragment resolution** removes every variable this study has exonerated —
resolution gap (none), batch acquisition protocol (none), training budget
(adequate), corpus chemistry span (6 scrolls). This is simultaneously the
best available shot at actual letters and the cleanest remaining test of the
chemistry-span hypothesis.

**Hypotheses.**
- **H9a:** the classic corpus's chemistry span covers 1203 → letterforms (or
  strongly structured text-like response) at 2.4 µm.
- **H9b:** the new scrolls' chemistry lies outside the classic labeled span →
  silence even with all other variables removed.

**Method.**
1. Train the flat `vesuvius_unet` (Exp 6 recipe unchanged) on all 8 native
   fragments / 6 scrolls (3.24 µm and 2.0 µm layers; the 2.4 µm target sits
   inside this span), 60 000 iterations = 7 500 it/frag (the §9.3 operating
   point), `save_every 10 000` so the final checkpoint exists (Exp 6 lesson).
2. Render the five self-grown 1203@2.4 µm segments (paths24, full-res
   coords; ~9.3 cm² total) at `--slice-step 1`, n = 65, S3-streamed.
3. Infer the final checkpoint on all five; frac>200/frac>128/std +
   letterform sheets; panels to the judging artifact.

**Controls.** (i) Capability: the final checkpoint inferred on one training
fragment (paris1_f3) must show letterforms (Exp 6-style; same caveat — 
capability, not generalisation; held-out generalisation at this recipe was
measured at AUC .857–.868 in §9.3). (ii) Render validation: mid-layer of the
first render must show papyrus structure (guards the paths24 coordinate-
scale inference).

**Decision rules.**
- **P1:** letterforms on 1203@2.4 µm ⇒ First Letters candidate on an
  eligible scroll (verify with organizers that letters from the 2.4 µm scan
  of an eligible scroll qualify); unlock the ladder: pseudo-label 1203@2.4 →
  transfer to 1203@9.36 via the proven degradation recipe → probe the
  remaining eight 9.36 scrolls.
- **P1b:** structured super-noise response without legible letters ⇒ grow
  more segments at the responsive band before concluding.
- **P2:** silence ⇒ chemistry-span verdict: new-scroll ink is outside the
  classic corpus span even at native resolution and classic protocol; the
  model-side lever reduces to corpus widening (in-scroll labels, Exp 10
  inventory), and the community message extends from "8.64 batch" to "the
  eligible set requires either new labeled chemistry or scanning answers."

**Budget.** ~9.5 h GPU training (overnight), ~11 GB renders (kept until
inference, then reviewed), inference ~30 min.

### 14.1 Interim result (2026-08-27): probe INVALIDATED before verdict — the segments are not sheet surfaces

Training completed (60k iterations; capability control passed — the model
reads Greek on a native-resolution training fragment). Three of five segments
were rendered and probed, all with an identical novel signature: zero
high-confidence activations, ~11.5 % mid-band, resolving visually into
fiber-flow texture plus inference-tile oscillation.

Before scoring the remaining segments, a depth-stack demonstration image
(built to answer a methods question from the PI) revealed that **the renders
show a cross-section through many windings at every depth — not a face-on
sheet**. The traced surfaces cut *across* the scroll instead of following a
sheet: the growth ran on a thin m7-L2 prediction band (L2 z ≈ 1800–2200) but
produced segments spanning z ≈ 1469–4932 — far outside the band's prediction
support. Garbage geometry in, meaningless inference out.

Consequences, recorded per protocol:
1. **The three 1203 probes are void.** No P1/P2 verdict is taken from them;
   §14's hypotheses remain open pending valid surfaces.
2. **Control insufficiency identified:** the pre-registered render check
   ("mid-layer shows papyrus structure") passes on cross-sectioned papyrus.
   The corrected control — adopted for all future renders — is *face-on
   appearance*: the mid-layer must look like a single sheet's surface, not
   striped winding cross-sections.
3. **Rung 10 is retroactively doubted.** The ladder's "decisive control"
   (§3, rung 10) used segments from this same band-limited grow on the same
   scan; its "cross-scroll transfer fails at native resolution" conclusion
   may be a segmentation artifact rather than a chemistry result. The
   conclusion is downgraded from "controlled result" to "unverified pending
   valid surfaces" — which also weakens the ladder's original headline
   attribution to scroll identity at native resolution (the 8.64 µm batch
   results, Exps 6–8, rest on different segment sources whose renders passed
   the face-on check by inspection, and are unaffected).
4. **Remedy in progress:** the full m7-L2 surface prediction for the 2.4 µm
   scan (24.5k chunks, ~5 GB effective) is being pulled so regrowth has
   complete prediction support; new segments will be validated by a fast
   cropped render + face-on check *before* full renders and inference. The
   scan team's own Sept-2025 published segments (9.36 µm frame) were also
   located as a second path.

## 15. Exp 10 — pre-registered 2026-08-27, before execution: corpus widening with in-scroll labels

**Hypothesis.** The fragment-only corpus lacks two things the eligible-scroll
domain has: wrapped-papyrus context (neighbor-sheet bleed in outer layers —
fragments are single sheets in air) and additional chemistry exposure. Adding
the community's in-scroll PHerc 1667 ink labels (100 koine-format crops from
6 scroll segments at 2 µm; bruniss `train_ink_eval`) improves cross-scroll
ink detection on in-scroll renders.

**Method.** Train the Exp 9 recipe on 108 sources (8 native fragments + 100
in-scroll crops; `scripts/exp10_build_corpus.py` packs the crops), 90 000
iterations overnight, `save_every` 5 000, resume-capable under systemd
(`exp10-train.service`). Evaluation once Exp 9's regrown 1203 surfaces pass
the face-on control: probe with both the Exp 9 model (fragments-only) and
this model — a paired comparison on identical renders.

**Decision rules.** P1: the wide model finds letterforms where the
fragments-only model does not ⇒ the missing ingredient was in-scroll domain/
chemistry; extend corpus building. P2: both silent on valid surfaces ⇒
chemistry-span verdict strengthens with the widest corpus assembled to date.
P3: both find letters ⇒ Exp 9's H9a stands and the comparison measures the
in-scroll data's gain.

## Appendix A: upstream bugs found (reported separately)

1. `vc_render_tifxyz --zarr-output` deterministic SIGSEGV (any n, local or
   remote); `--tif-output` unaffected. Full matrix: `docs/vc3d-bug-reports.md`.
2. Local zarr chunk-lookup mismatch: local `-v` volumes silently render as
   fill-value; strace shows wrong chunk indices; `--remote-url` path correct.
3. koine gotchas: stale `flat_ink_patches_*.json` cache survives dataset
   fixes; resnet3d flat-mode training needs `decoder_upscale=4` (local patch);
   labels live on a single z-plane (z=32) — resampling must preserve it at
   `shape[0]//2`.

## Appendix B: artifact index

- Rung-4 positive: `runs/ink_sim86_resnet_ft/predictions/overview_div6.png`,
  `overlay_mask_div6.png` (red = supervised regions).
- Rung-8 positive: `runs/s1_79um_probe/sim86v2ft_div10.png`.
- Rung-6b sheet: `runs/1447_probe/batch/contact_sheet.png`; rung-6a:
  `runs/1447_probe/offset_sweep/sweep_sheet.png`.
- Rung-9 sheets: `runs/PHerc1218_probe/contact_sheet.png`, `PHerc0800…`,
  `PHerc0268…`.
- Rung-10: `runs/PHerc1203_24um_probe/*_stretched.png`.
- All viewable at `the runs/ directory of the data volume`.
