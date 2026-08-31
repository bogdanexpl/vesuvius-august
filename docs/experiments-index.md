# Experiments index

Every experiment in the campaign, in order, with outcome. Full detail:
`docs/study-2026-08-cross-scroll-ink-transfer.md` (sections noted);
day-by-day log with dead ends in the main repository's `DESIGN.md`.
Per-model training details — schedule, wall time, checkpoint choice — are in
`training-runs.md`.

## The ladder (study §3) — which factor blocks ink recovery at 8.6 µm?

| rung | question | outcome |
|---|---|---|
| 1 | 2.4 µm model on native 8.6 µm render, naive | negative (patch noise) |
| 2 | physically matched rescaling enough? | negative — input rescaling doesn't transfer |
| 3/3b | closer 7.9 µm teacher; GP-era preprocessing | negative; not a normalization artifact |
| 4 | is 8.6 µm ink learnable at all? | **positive** — 20-min fine-tune on sim-8.6 µm reads Greek, 78 % outside supervision |
| 5 | rung-4 model on real 1447 | negative (blobs) |
| 6a | surface z-offset sweep (old model) | no depth helps (superseded by Exp 7) |
| 6b | all 14 published 1447 segments | negative everywhere |
| 7 | acquisition-statistics-matched sim | sim sanity holds; real 1447 unchanged |
| 8 | sim-trained model on a real 7.9 µm scan | **positive** — recipe survives sim-to-real |
| 9 | 46 self-grown segments across the 8.64 µm batch | uniformly silent |
| 10 | native-resolution cross-scroll control (1203 @ 2.4 µm) | negative — later downgraded (§14.1: segment validity unverified) |

## Numbered experiments

| exp | name | one-line description | status / outcome |
|---|---|---|---|
| 1 | 3D-DINO probe | community 8-scroll direct-3D ink model on a 1203 2.4 µm region | fiber false-positives, no letters; projection caveat |
| 2 | DINO ink-likeness | ViT feature similarity to known ink | **invalid** — state-dict mis-load, positive control failed; redo checklist recorded |
| 3 | fragment inventory | which physical scrolls have public IR-verified ink labels | ≥7 scrolls; enabled Exp 4 |
| 4 | LOSO benchmark (§7, §9) | first cross-scroll ink benchmark: leave-one-scroll-out + 4×4 pairwise; then budget 2×2, scaling curve, matched ceilings | AUC monotone in it/frag (r=.98), +0.101/doubling unsaturated; matched penalty −0.02…−0.04 |
| 5 | ink density families (§10) | Cohen's d of ink vs background intensity per fragment | two families tracking transfer clusters; beam-energy confound stated, cause unproven |
| 6 | budget-adequate 1447 probe (§11) | 5-scroll sim-8.64 model at the scaling operating point on depth-matched 1447 renders | negative with passing positive control |
| 7 | surface-placement test (§12) | pre-registered offset sweep ±207 µm + Nov-2025 segmentation generation | P3: both surface hypotheses rejected; chemistry promoted |
| 8 | batch generality (§13) | same protocol on 1218/0800/0268 (six ~47 cm² segments) | P2: batch-wide silence; response *decreases* with better surfaces |
| 9 | native-resolution 1203 probe (§14) | 8-fragment native-res model (60 k iters, capability control passed) on 1203 @ 2.4 µm | **invalidated before verdict** — probe segments were winding cross-sections; face-on control adopted; grower bug filed |
| 10 | corpus widening (§15) | retrain with 100 in-scroll Scroll 4 ink-label crops added (108 sources, 90 k iters) | trained; evaluation via Exp 11 |
| 11 | paired 9.36 µm probe (§15 eval) | fragments-only vs wide model, both sim-9.36 fine-tuned, on validated published 1203 segments | in flight |

## Cross-cutting contributions

- **Face-on render control** (§14.1): mid-layer must look like a single sheet;
  in-segment dark-gap fraction separates cross-sections from sheets ~8×.
- **Four upstream bug reports** (`vc3d-bug-reports.md`): render zarr-output
  segfault; local-zarr chunk lookup returning fill; koine cache eviction
  race; grower producing cross-cutting surfaces on coarse binary predictions.
- **Headless probe pipeline**: seed→grow→S3-streamed render→pack→infer,
  ~10 min/segment at 8.64 µm, no GUI required.
