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
reports; FID is), not a harness problem. STEP1b's sit/lie MPJPE figures stand as
reported.

**Task 1 — not yet completed. Blocked on shared-GPU contention, not a design issue.**
Two earlier attempts (`GPT_eval_multi.py`, first at the paper's 20 repeats, then
reduced to 3 — the script has `repeat_time = 3` hardcoded with a comment explaining
the reduction) both stalled for hours at 100% GPU utilization with zero log
progress, traced to another process (~16–19GB) sharing the same 4090. Both were
killed rather than left to run indefinitely. **CLAUDE.md's environment section
should be treated as needing a correction: the 4090 is not exclusively ours.** Task
2's success just now shows the GPU does yield usable cycles for lighter jobs even
under contention — Task 1 (autoregressive generation, much heavier per sample) may
still need genuine free time to complete.

## Next steps

- Retry Task 1 (`GPT_eval_multi.py`, already reduced to 3 repeats,
  `T2M-GPT/pretrained/` checkpoints already on disk) when the GPU allows.
- Once Task 1 lands: if generation FID/R-precision also land near the paper, Step
  2's done-criterion is fully met and Step 3 (VQ-VAE joint finetune) is unlocked. If
  not, investigate before proceeding — do not build Step 3 on an uncalibrated
  harness.
