# 02 — Baseline calibration

**Status: DONE for reconstruction; the harness is trusted.** Generation-FID reproduction
is still open but blocks nothing.

## Result

Reproducing T2M-GPT's **reconstruction** numbers with our harness and their checkpoint:

| metric | ours | paper |
|---|---|---|
| reconstruction FID | 0.066 ± .001 | 0.070 ± .001 |
| R@1 / R@2 / R@3 | all within variance | — |

This is what makes every downstream reconstruction number believable. It is scoped to
reconstruction — encode/decode only, no autoregressive sampling.

## Frozen-tokenizer characterization (held-out)

Per-category MPJPE of the off-the-shelf VQ-VAE. This is the "before" that
`03_tokenizer_finetune.md` improves on.

| category | MPJPE (mm) |
|---|---|
| H3D baseline | 56.11 |
| H3D-lie (control, n=11) | 117.46 |
| HUMANISE walk | 50.20 |
| HUMANISE stand up | 66.98 |
| HUMANISE sit | 69.55 |
| HUMANISE lie | **136.91** |

Reading: locomotion reconstructs at roughly baseline, interaction worse, lying worst.

## Normalization — the one thing to get right

The VQ-VAE checkpoint has its **own** expected mean/std at
`checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/{mean,std}.npy`. These are **not** the
raw H3D data-prep `Mean.npy`/`Std.npy`. Using the wrong pair inflates every number ~2-3×
(the H3D baseline alone reads 137 mm instead of 45 mm) for both MPJPE and FID. Every number
in this repo uses the checkpoint's own stats. This has caused three separate false alarms;
assume it is the first thing wrong if a reconstruction number looks absurd.

## Held-out discipline

Numbers above come from each dataset's official `test.txt` only. An earlier H3D baseline of
**45.3 mm was leaked** — it sampled all of `new_joint_vecs`, including clips the model
trained on. That was fine for characterizing a frozen model that trained on none of it, but
it is not a valid reference once we are the ones training. **56.11 mm is the honest figure;
do not quote 45.3.** Same for the H3D-lie control: 117.46 held-out, not 90.1.

## Still open (not blocking)

Generation FID / R-precision reproduction — autoregressive sampling, not just encode/decode.
Root-caused to a self-inflicted timeout, not GPU contention or a code fault. See
`archive/STEP2_baseline_calibration.md` Task 1 before re-attempting, to avoid rediscovering
the same trap. Neither completed track needed it.
