# 03 — Tokenizer joint finetune (Stage A)

**Status: DONE. It works.** The sit/lie reconstruction gap was real codebook coverage, not
an upstream data artifact. Final checkpoint:
`track2_checkpoints/track2_joint_finetune_run1/net_iter020000.pth` (4090, gitignored).

## Result

Held-out, evaluator-consistent normalization, frozen → finetuned:

| category | before (mm) | after (mm) | change |
|---|---|---|---|
| H3D baseline | 56.11 | 56.2 | +0.2% (flat) |
| H3D-lie (n=11, noisy control) | 117.46 | 117.4 | flat |
| HUMANISE walk | 50.20 | 34.1 | **−32.1%** |
| HUMANISE stand up | 66.98 | 53.0 | **−20.9%** |
| HUMANISE sit | 69.55 | 47.6 | **−31.6%** |
| HUMANISE lie | 136.91 | 96.3 | **−29.6%** |

Both gates met: interaction improved, and nothing regressed — H3D held flat while every
HUMANISE category improved. Lie sits in a 92–101 mm band across the last 9 of 11
checkpoints, so this is a plateau, not a lucky dip. The two lr decays (12k, 18k) moved
nothing.

**This also answers the upstream question.** Finetuning the codebook alone, with zero
change to the SMPL-X → 22-joint step, closed most of the lie gap. So that step is not the
culprit. It remains unverified (see `01_data_pipeline.md`), just no longer suspect.

## How

- **Data**: HumanML3D + HUMANISE, 1:1 balanced sampling per batch (`h3d_frac=0.5`)
  regardless of the underlying 14.6k / 19.6k imbalance.
- **Splits**: official `train.txt` / `test.txt` for both datasets, always. Trains on train,
  evaluates on test.
- **Normalization**: the frozen checkpoint's own meta mean/std throughout, train and eval.
  The finetune warm-starts from that embedding space, so keeping its normalization keeps
  the encoder in-distribution from iteration 0 and keeps every number comparable to
  `02_baseline_calibration.md`.
- **Recipe**: unchanged from T2M-GPT's own VQ-VAE run (`commit=0.02`, `loss_vel=0.5`,
  `recons_loss=l1_smooth`), except lr 2e-5 instead of 2e-4 — 10× lower because this is a
  finetune of a converged model, not training from scratch. 20k iters, batch 256, window 64.

## Two things that would have silently ruined this

**`QuantizeEMAReset` does not survive a checkpoint load.** It tracks `init`, `code_sum` and
`code_count` as plain Python attributes, not registered buffers. So
`load_state_dict(strict=True)` restores the `codebook` tensor but leaves `init=False`, and
the first `train()` forward runs the from-scratch init path and **overwrites all 512 codes
from a single batch** — verified empirically before any real training. Anyone resuming or
finetuning this VQ-VAE must seed the EMA accumulators from the loaded codebook first
(`prepare_quantizer_for_finetune`).

**The 64-frame window drops most of HUMANISE.** HUMANISE clips are much shorter than
HumanML3D's, so requiring 64 frames keeps only 6334/16523 (38.3%) of HUMANISE train clips —
lie 49.0%, sit 50.5%, stand-up 22.9%, walk 33.4%. Sit and lie survive better than walk, so
this does not bias against the categories of interest, but the effective HUMANISE pool is
smaller than the split suggests. Kept at 64 to match the architecture recipe exactly; a
shorter window would be a second confounding change and would truncate the very sit-down /
lie-down transitions this training is for.

## Qualitative

Held-out lie clips spanning the frozen model's error range: best 65.1→55.1 mm, median
117.8→**43.4** mm, worst 321.7→279.2 mm. The worst case is an atypical curled pose that
finetuning improves but does not fix — distorted limb placement, but no NaN and no
exploded skeleton in either model. Moderate degradation, not catastrophic.

## Code

`src/joint_vqvae_dataset.py`, `scripts/track2/{precompute_humanise_263,
eval_per_category_mpjpe,train_vqvae_joint_finetune,render_lie_comparison,render_lie_video}.py`
