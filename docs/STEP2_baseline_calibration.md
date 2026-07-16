# Step 2 — Baseline Calibration

Status: **IN PROGRESS, blocked on GPU availability (not a design problem).**
Build-order item 2 in CLAUDE.md. Purpose: reproduce T2M-GPT's published numbers on
HumanML3D so the eval harness is trusted before any downstream number (including the
Step 3 VQ-VAE joint-finetune results) is taken at face value.

Note on scope: this step was originally framed around a "locomotion-only MVP"
decision that has since been reversed (see CLAUDE.md — the project now targets full
interaction, not navigation-only). That reversal doesn't change what Step 2 itself
needs to do: it's pure harness calibration against the *original* HumanML3D-only
T2M-GPT checkpoint, independent of scope decisions. The locomotion filter built
during this step (below) is no longer the planned training-data cut, but the code
and its numbers remain available if a locomotion-only ablation is ever useful.

---

## Ask

1. **Reproduce the paper's reported metrics** — FID and R-precision (Top-1/2/3) via
   T2M-GPT's own eval harness, HumanML3D test set, official checkpoints. Compare to
   the paper; if it's off, the harness or checkpoints are wrong and nothing
   downstream can be trusted yet.
2. **Reconstruction-FID calibration** — encode→decode only (no GPT sampling),
   compare to T2M-GPT's reported recon FID. This resolves whether Step 1's H3D
   MPJPE=137.3mm number is a metric-convention artifact (recon FID is what the paper
   reports; MPJPE conventions vary) or a genuine harness problem.
3. **Locomotion filter for HUMANISE** (historical — see scope note above) — label-based
   filter keeping walk/stand, dropping sit/lie, with per-category counts logged.

## Findings so far

**Task 3 — done.** `src/humanise_join.py::locomotion_filter()` +
`scripts/verify/check9_locomotion_filter.py`, committed. Result over all 19,648
HUMANISE clips:

| Action | Count | Kept? |
|---|---|---|
| walk | 8,264 | yes |
| stand up | 3,463 | yes |
| sit | 5,578 | no |
| lie | 2,343 | no |
| **locomotion total** | **11,727 / 19,648 (59.7%)** | |

**Task 2 — done.** Found and fixed the 2 NaN-corrupted HumanML3D files (see Step 1
doc). Three H3D walk reconstructions were rendered for visual sanity
(`scripts/verify/check10_h3d_walk_renders.py`), MPJPE in the 50–340mm range depending
on clip complexity, consistent with Step 1's baseline.

Reconstruction-FID run (`VQ_eval.py`, `repeat_time` reduced from the paper's 20 to 3
for the same shared-GPU reason as Task 1 — see below; finished cleanly in ~4.5 min
despite the contention, since VQ-only encode/decode is much lighter than AR
generation):

| Metric | Ours (3 repeats) | Paper |
|---|---|---|
| Reconstruction FID | **0.066 ± .001** | 0.070 ± .001 |
| Diversity | 9.740 ± .074 | — |
| R@1 | 0.496 ± .008 | 0.491 ± .001 |
| R@2 | 0.692 ± .003 | 0.680 ± .003 |
| R@3 | 0.787 ± .003 | 0.775 ± .002 |
| Matching score | 3.063 ± .011 | — |

**Verdict: recon FID matches the paper within reproduction variance. The harness and
checkpoint are sound.** This resolves the open question from Step 1: the H3D
MPJPE=137.3mm figure was a metric-convention difference (MPJPE isn't what the paper
reports; FID is), not a harness problem.

**Correction — STEP1b's sit/lie MPJPE figures do NOT stand as reported.** While
extending this into a per-category HUMANISE recon-FID breakdown (`check12`), the
FID pipeline initially produced nonsensical results (categories inverted relative to
MPJPE — `lie` looked *better* than `walk`). Root-caused via a control test: this
VQ-VAE checkpoint has its own normalization stats at
`checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/{mean,std}.npy`, distinct from
the `H3D_ROOT` `Mean.npy`/`Std.npy` used everywhere in Step 1 — running the same
code on H3D itself gave FID=5.25 with the wrong stats, FID=0.072 (matches the paper)
with the right ones.

Re-ran Step 1's MPJPE canary with the corrected stats, all categories
(`scripts/verify/check14_mpjpe_evaluator_norm.py`):

| Category | MPJPE (corrected) | vs. H3D baseline | (original, wrong-norm) |
|---|---|---|---|
| H3D baseline | **45.3mm** | — | 137.3mm |
| walk | 47.9mm | 1.06x | ~112mm |
| stand up | 67.6mm | 1.49x | ~192mm (1.4x) |
| sit | 72.6mm | 1.60x | ~293mm (2.1x) |
| lie | **139.8mm** | **3.09x** | ~703mm (5.1x) |

Then rendered 10 random `lie` clips (`scripts/verify/check15_lie_ten_samples.py`) to
check whether 3.09x is uniform or driven by a few broken clips: per-clip MPJPE
65-246mm, mean 135.6mm, std 56mm, **no catastrophic outliers**, and every
reconstruction visually preserved the same overall pose family as its real
counterpart (moderate joint-angle drift, not structural breakage).

**Verdict: "lie is structurally broken (5.1x)" was substantially a normalization
artifact.** The corrected, real finding is "lie is the worst category by a real
margin, 3.09x, moderately degraded, not broken." CLAUDE.md section 2b has been
updated to state this correctly rather than quote the original number. The
per-category FID from `check12` itself (after the same normalization fix) is
walk=2.24, stand-up=0.70, sit=1.02, lie=0.41 — **this ranking should NOT be used
for anything**, since it's confounded by wildly different intrinsic diversity per
category (trace of the real-embedding covariance: walk 14.9, stand-up 6.9, sit 7.6,
lie 1.6 — `lie`'s real motions are nearly a single point in the evaluator's
embedding space, so FID can't meaningfully discriminate reconstruction quality for
that category). MPJPE is the trustworthy per-category comparison here, not FID.

**Task 1 — still not completed after four attempts. Blocked on shared-GPU
contention, not a design issue.** Attempts: 20 repeats (stalled, killed), 3 repeats
(stalled, killed), a 45-minute bounded retry (`timeout 2700`, ran to its bound with
zero log progress, self-terminated cleanly). All traced to another process
(~16–19GB) sharing the same 4090 at 98-100% utilization throughout. **CLAUDE.md's
environment section has been corrected: the 4090 is not exclusively ours.** Lighter
jobs (Task 2's VQ-only eval, the per-category FID/MPJPE work above) reliably get
usable GPU cycles even under this contention; Task 1's autoregressive generation
loop specifically has not, across four tries.

## Next steps

- Retry Task 1 (`GPT_eval_multi.py`, already reduced to 3 repeats,
  `T2M-GPT/pretrained/` checkpoints already on disk) when the GPU allows — ideally
  confirm the other job has actually finished first (`nvidia-smi
  --query-compute-apps`) rather than attempting blind again.
- Once Task 1 lands: if generation FID/R-precision also land near the paper, Step
  2's done-criterion is fully met and Step 3 (VQ-VAE joint finetune) is unlocked. If
  not, investigate before proceeding — do not build Step 3 on an uncalibrated
  harness.
- Step 3 planning should use the corrected per-category MPJPE numbers above (3.09x
  for lie), not the original Step 1 canary numbers, when deciding how much the
  joint finetune needs to close the gap.
