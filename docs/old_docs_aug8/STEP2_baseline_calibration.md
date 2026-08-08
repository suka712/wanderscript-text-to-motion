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

**Task 2 — partial.** Found and fixed the 2 NaN-corrupted HumanML3D files (see Step
1 doc). A reconstruction-FID run was started (`VQ_eval.py`, repeat-averaged) but was
never carried to a final number — see blocker below. Three H3D walk reconstructions
were rendered for visual sanity (`scripts/verify/check10_h3d_walk_renders.py`),
MPJPE in the 50–340mm range depending on clip complexity, consistent with Step 1's
baseline.

Reference numbers pulled from the T2M-GPT paper (arXiv, Table 1 + Table 3,
HumanML3D, τ=0.5 config matching the downloaded `VQTransformer_corruption05`
checkpoint) to compare against once eval numbers land:

| Metric | Paper |
|---|---|
| Reconstruction FID | 0.070 ± .001 |
| Generation FID | 0.116 ± .004 |
| R@1 | 0.491 ± .001 |
| R@2 | 0.680 ± .003 |
| R@3 | 0.775 ± .002 |

**Task 1 — not completed. Blocked on shared-GPU contention, not a design issue.**
Two attempts (`GPT_eval_multi.py`, first at the paper's 20 repeats, then reduced to
3 — the script now has `repeat_time = 3` hardcoded with a comment explaining the
reduction) both stalled for hours at 100% GPU utilization with zero log progress,
traced to another process (~16–19GB) sharing the same 4090. Both attempts were
killed rather than left to run indefinitely. **CLAUDE.md's environment section
should be treated as needing a correction: the 4090 is not exclusively ours.**

## Next steps

- Confirm GPU availability before retrying (`nvidia-smi --query-compute-apps`).
- Re-run Task 1 (`GPT_eval_multi.py`, already reduced to 3 repeats,
  `T2M-GPT/pretrained/` checkpoints already on disk) and Task 2's recon-FID run to a
  finished number.
- Write the actual verdict once numbers land: harness trusted (proceed to Step 3) or
  not (fix before proceeding — do not build the VQ-VAE finetune on an uncalibrated
  harness).
