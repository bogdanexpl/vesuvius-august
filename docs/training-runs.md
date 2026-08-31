# Training runs

Every model trained for the results in this submission, with the question it
served, the data it saw, the architecture and schedule, wall time, and which
checkpoint was used and why. Hardware throughout: one RTX A6000 (48 GB),
mixed precision fp16, data streamed from the Vesuvius Challenge open-data
bucket. Configurations are in `configs/`; the two architectures are the koine
`vesuvius_unet` (2.5D UNet, 64-channel flat mode) and `resnet3d`
(ResNet3D-50 segmentation head).

Every checkpoint used for a reported result was selected by one of two rules,
stated per run below: **final** (the last checkpoint of a completed schedule)
or **last surviving** (where a save-interval mistake meant no final
checkpoint existed). No checkpoint was ever selected by scoring several
against the target — the eligible scrolls have no labels, so there is nothing
to select on, and doing so would manufacture the result.

## Phase 0 — the anchor

| | |
|---|---|
| **Question** | Can we reproduce a known read end to end on this hardware, so later negatives mean something? |
| **Data** | Scroll 1 (PHerc. Paris 4) ink tutorial dataset, segment w00, 2.4 µm, labels on one z-plane |
| **Model** | `vesuvius_unet`, patch 64×256×256, batch 2, LR 0.01, from scratch |
| **Schedule** | 20 000 iterations, checkpoint every 1 000 |
| **Wall time** | 3 h 07 m training, 53 min full-segment inference |
| **Checkpoint** | `ckpt_020000` — **final**. Verified by reading legible Greek rows on w00 before being used as the anchor for rungs 1–2 |

## Rung 4 — is 8.6 µm ink learnable at all?

| | |
|---|---|
| **Question** | Does the ink signal survive 8.6 µm sampling, or is resolution the blocker? |
| **Data** | Scroll 1 tutorial data degraded to 8.6 µm-equivalent (3.6× in-plane, 65→18 z), built by `downsample_to_sim86.py` |
| **Model** | `resnet3d`, patch 18×256×256, batch 8, LR 5e-4, initialised from the public `resnet50_7.9um_scroll1_frags` weights wrapped into koine format |
| **Schedule** | 5 000 iterations, checkpoint every 1 000 |
| **Wall time** | 20 m 20 s |
| **Checkpoint** | `ckpt_005000` — **final**. Used for rungs 4, 5, 6a, 6b |

The short schedule is the point: this rung asks whether resolution adaptation
is cheap, and 20 minutes was enough to read text the model had never seen.

## Rung 7 — acquisition-statistics matching

| | |
|---|---|
| **Question** | Does matching the eligible renders' second-order image statistics close the sim-to-real gap? |
| **Data** | Rung-4 corpus rebuilt with 3D gaussian PSF blur and a quantile LUT toward measured eligible-render statistics (`build_sim86v2.py`) |
| **Model** | `resnet3d`, patch 18×256×256, batch 8, LR 3e-4, initialised from the rung-4 checkpoint |
| **Schedule** | 5 000 iterations, checkpoint every 1 000 |
| **Wall time** | ~20 m |
| **Checkpoint** | `ckpt_005000` — **final**. Used for rungs 7, 8, 9 |

## Exp 4 — the cross-scroll benchmark and the budget sweep

All folds share one recipe — `vesuvius_unet`, patch 64×256×256, batch 2,
LR 0.01, trained from scratch — so that iterations and corpus size are the
only variables. Data is the 8-fragment corpus built by `exp4_fetch_frags.py`
from public IR-labeled fragments; each fold trains on some scrolls and is
scored on a held-out scroll's fragment with `exp4_score.py`.

| run | training corpus | iterations | wall time | purpose |
|---|---|---|---|---|
| `exp4_loso` (4 folds) | 3 of 4 scroll groups | 5 000 | ~45 m each | first LOSO benchmark |
| pairwise (2 extra models) | 1 scroll group each | 5 000 | ~45 m each | 4×4 transfer matrix |
| `exp4_loso_v2` (6 folds) | 5 of 6 scroll groups | 5 000 | ~45 m each | corpus extended to 6 scrolls |
| `exp4_budget15k` | 5 scroll groups | 15 000 | 2 h 19 m / 2 h 16 m | budget arm of the 2×2 |
| `exp4_cell_3scroll15k` | 3 scroll groups | 15 000 | 2 h 16 m | the 2×2's fourth cell |
| `exp4_ceiling15k` (2 runs) | 1 fragment each | 15 000 | 2 h 16 m / 2 h 15 m | budget-matched within-scroll ceilings |
| `exp4_scale30k` | 3 scroll groups | 30 000 | 4 h 30 m | the scaling curve's largest point |

**Checkpoint rule:** every fold used its **final** checkpoint. Comparisons
are only meaningful if the schedule is the variable, so no fold's checkpoint
was chosen by inspection.

Roughly 20 GPU-hours across this experiment. The scaling curve is read
across runs, not within one: 714, 1 250, 2 143, 3 750 and 7 500 iterations
per training fragment come from combining schedule length with corpus size.

## Exp 6 — the budget-adequate probe model (used for Exps 6, 7, 8)

| | |
|---|---|
| **Question** | Do the eligible scrolls stay silent under a model trained at the operating point Exp 4 identified? |
| **Data** | `frags_sim86`: 7 fragments from 5 scrolls degraded to 8.64 µm in-plane, ink fractions preserved exactly (`exp6_build_sim86_corpus.py`). PHerc 500P2 excluded — its 27 160 × 14 990 plane exceeded memory during inference |
| **Model** | `vesuvius_unet`, patch 64×256×256, batch 2, LR 0.01, from scratch |
| **Schedule** | 52 500 iterations planned (7 500 per fragment, the Exp 4 operating point), checkpoint every 15 000 |
| **Wall time** | 8 h 09 m |
| **Checkpoint** | `ckpt_045000` — **last surviving**, and the reason is a mistake worth stating: `save_every` was set to 15 000 against a 52 500-iteration schedule, so no checkpoint was written at the end. The best available is 45 000 iterations, or 6 428 per fragment against the 7 500 intended — 86 % of the planned exposure |

This checkpoint carries the positive control that gives the negatives their
meaning: applied to a simulated-8.64 µm PHerc 1667 fragment it produces clear
Greek letterforms (4.6 % strong activations) while returning 0.08–2.5 % on
every eligible-scroll segment. Because that fragment was in training, the
control tests capability rather than generalisation; held-out generalisation
for this recipe was measured separately in Exp 4 at AUC 0.868.

## What is not here

Inference runs are not tabulated — there are several hundred across the
campaign, each a few minutes to ~25 minutes depending on segment size, all
using the checkpoints above with no per-target tuning. The eligible-scroll
probes have no labels, so nothing about them was fitted or selected; the
render protocol per probe is recorded in the study report.
