# tokenizer_finetune — joint VQ-VAE finetune (diagnostic)
Updated: 2026-08-09 · Branch: track2-tokenizer · Runs on 4090, parallel to Track 1

## Question
Does joint finetune move HUMANISE-lie 140→~90 without regressing general motion — and was
the lie gap real codebook coverage or an upstream artifact? Diagnostic only: does NOT
re-extract tokens or touch the transformer. Branch does NOT merge on completion.

## Branch
On a clean, committed `master` (**not `main`** — `origin/main` on this remote is an
unrelated old repo; `master` is the real branch, confirm with `git branch -a` if unsure):
`git checkout -b track2-tokenizer`. Commit here.

## Setup
- Unfreeze the VQ-VAE (encoder + codebook + decoder).
- Joint dataloader: HumanML3D + HUMANISE, balanced sampling (start 1:1, tunable) so neither
  dominates (14.6k vs 19.6k).
- Use the validated 263 inputs (converter verified 0.80mm; upstream SMPL-X→joint step NOT
  verified — keep in mind when reading results).

## Finetune
Joint finetune on both datasets. Checkpoint regularly; keep the frozen baseline for comparison.

## Validation — per category, held-out (MPJPE)
- HUMANISE: walk / stand / sit / lie
- HumanML3D: overall + lie (controls: baseline 45.3, H3D-lie 90.1)
- **Normalization**: all reference numbers here (45.3, 47.9, 90.1, 139.8, etc.) were
  computed with the FROZEN checkpoint's own mean/std
  (`checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/{mean,std}.npy`), not
  `H3D_ROOT/Mean.npy`/`Std.npy` — using the latter inflates every number ~2-3x and has
  already caused two false alarms in this project. For the finetuned checkpoint,
  decide explicitly whether to keep using the frozen checkpoint's original norm stats
  (finetuning warm-starts from that embedding space) or recompute them — and state
  which you used when reporting results, since this comparison is meaningless otherwise.

## Success — BOTH must hold
1. Interaction improves: HUMANISE-lie drops from 140 toward ~90.
2. No forgetting: HumanML3D overall and HUMANISE walk do not regress (walk ~47.9, H3D ~45.3).

## Decision gate (report explicitly)
- Lie improves + general holds → gap was real coverage; finetune works; tokenizer track stays
  viable pending Track 1.
- Lie won't improve (stays ~140) → likely the upstream SMPL-X→joint ARTIFACT, not coverage.
  Finetune is the wrong fix — STOP and flag for upstream investigation.
- General regresses → sampling balance off / unstable; tune the ratio before concluding.

## Do NOT
- Do not re-extract transformer tokens or retrain the transformer.
- Do not merge this branch on completion — report; merge decided at reconciliation.

## Deliverable
docs/track_2/RESULTS.md: per-category MPJPE before vs after, both criteria evaluated,
a few lie reconstruction renders (moderate vs broken), clear verdict on the gate.