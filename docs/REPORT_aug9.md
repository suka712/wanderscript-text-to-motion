# Progress Report — 2025-08-09

Consolidated record of achieved state. Supersedes STEP1_plumbing.md and
STEP2_baseline_calibration.md.

## 1. What we're building
Text-to-motion for indoor scenes, general full-body motion including scene interaction
(sit/lie/reach). Built on T2M-GPT (VQ-VAE tokenizer + autoregressive transformer).
Body-level, 22 joints. See CLAUDE.md for the full pipeline and contributions.

## 2. Data pipeline — DONE, validated
- **HUMANISE join:** 3 sources (pure_motion / align_data / contact_motion) joined
  19,648 / 19,648 = 100%, full scale.
- **Converter** (`src/motion_features.py`): 22-joint positions → 263-dim HumanML3D format.
  Reuses HumanML3D's own extractor; we wrote only adapters (Z-up→Y-up relabel,
  reference-skeleton offsets, drift-free local-position reader). Input is 22-joint
  positions (SMPL-X→joint reduction happens upstream in HUMANISE's prep, external).
  **Verified: 0.80 mm mean MPJPE exact-match vs HumanML3D's shipped 263; foot-contact
  bit-exact.**
- **Two-track storage:** canonicalized 263 (for tokenizer) + world-frame trajectory
  (x,y,yaw per frame, for placement/chaining/metrics). World-frame rebuilt from
  pure_motion + align_data, validated by floor-overlay on 150/150 scenes. HUMANISE
  confirmed Z-up.
- **Scene meshes:** 643 ScanNet meshes load; BEV renderer world→pixel error 0.67 px.

## 3. Tokenizer characterization — DONE
Eval harness reproduces T2M-GPT (recon FID 0.066 vs paper 0.070; R@1/2/3 match) — trusted.

Per-category reconstruction, FROZEN tokenizer (MPJPE, mm):
| category | HUMANISE | reference |
|---|---|---|
| walk | 47.9 | H3D baseline 45.3 |
| stand up | 67.6 | |
| sit | 72.6 | |
| lie | 139.8 | H3D-lie 90.1 |

Reading: locomotion reconstructs at baseline; interaction worse, lie most. The H3D-lie
control (90.1) shows lying is intrinsically ~2× hard even in-distribution, and HUMANISE
lie is a further ~55% worse — a HUMANISE-specific gap.

## 4. How far along
Foundation complete: data prepared and verified, harness trusted, base tokenizer
characterized. No model trained yet. Architecture is fully specified (CLAUDE.md).

Verification framing used throughout (two pipelines):
- A (converter only): 22-joint → 263 → invert → compare. Isolates converter. 0.80 mm.
- B (with VQ-VAE): 263 → encoder → quantize → decoder → 263 → compare. Converter held
  constant, so error is the tokenizer. Produces the section-3 numbers.

## 5. Next — two parallel diagnostic tracks
Open question: is a tokenizer finetune actually needed? Two independent experiments,
run in parallel, reconciled after. Both are diagnostic, not the committed build.

- **Track 1 — grounding (3090, branch `track1-grounding`).** Frozen tokenizer; add
  start-pose + goal conditioning to the transformer; probe whether generated motion
  reaches the goal. Answers "is the tokenizer even the bottleneck?"
  Specs: docs/track1_grounding/.
- **Track 2 — tokenizer (4090, branch `track2-tokenizer`).** Joint finetune on
  HumanML3D + HUMANISE; does HUMANISE-lie improve 140→~90 without regressing general
  motion? Also diagnoses whether the lie gap is real coverage or an upstream artifact.
  Spec: docs/track2_tokenizer/.

Reconcile:
| Track 1 grounding | Track 2 finetune | verdict |
|---|---|---|
| works | improves | tokenizer optional — ship frozen, finetune is a quality bump |
| works | won't improve | frozen tokenizer, drop finetune, investigate upstream data |
| fails | either | grounding is the real problem — pivot to trajectory-first |

Track 2's changed codebook invalidates Track 1's tokens — do not merge the tracks. Learn
from both, then do the real build once.

## 6. Known caveat
The upstream SMPL-X→22-joint reduction (HUMANISE's own, external to us) is not
independently verified. It underlies all HUMANISE numbers. Track 2's result helps
diagnose it: if HUMANISE-lie won't improve under finetune, suspect this step.