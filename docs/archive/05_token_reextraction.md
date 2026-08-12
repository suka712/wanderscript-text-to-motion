# 05 — Token re-extraction

**Status: DONE and verified.** All 19,648 HUMANISE clips re-tokenized with the finetuned
VQ-VAE. Mandatory gate between Stage A and any transformer training (CLAUDE.md 2b): tokens
are indices into a specific codebook, so Stage A invalidated every previously extracted
token.

Frozen tokens stay at `track1_probe/tokens/`, finetuned at `track1_probe/tokens_finetuned/`.
Separate directories, never overwritten, so `04_grounding.md` stays reproducible.

## Verification

File sizes are identical between the two token sets, which proves nothing — token *count*
per clip depends only on clip length. What was actually checked:

| check | result |
|---|---|
| codebook rows changed vs frozen | **512 / 512**, max abs diff 44.75 |
| test clips with identical token sequences | 80 / 3125 |
| per-token agreement, frozen vs finetuned | 45.3% |
| goals / starts unchanged (must not depend on tokenizer) | identical ✓ |
| clips processed | 19,648 (16,523 train + 3,125 test), 0 skipped |

**Oracle floor** — ground-truth tokens → decode → SE(2) place → distance to that clip's own
fed goal, each tokenizer against its own tokens, same 200 held-out clips:

| tokenizer | oracle floor |
|---|---|
| frozen | 0.124 m |
| finetuned | **0.107 m** (−13.6%) |

The better tokenizer reconstructs trajectories better, which is what `04_grounding.md`
predicted from the displacement breakdown.

## The hazard this step exists to prevent, quantified

Feeding the **new tokens through the old decoder** gives **0.321 m** — 3× worse than the
correct pairing, with no error raised anywhere. Stale tokens decode to something plausible
rather than crashing, so a skipped re-extraction shows up as a mediocre model, not a bug.
Always name the tokenizer explicitly: `--tokens-dir` and `--vqvae-ckpt` must agree.

## Downstream effect — the probe, re-run on the new tokenizer

Both models retrained on the new tokens (the frozen-tokenizer models could not be reused:
they emit frozen-codebook tokens, which is exactly the 0.321 m mismatch above). Same 200
held-out clips, same seed:

| | goal-error | median | SEM | corr(commanded, achieved) |
|---|---|---|---|---|
| NULL — stay at start | 0.627 m | 0.378 m | 0.050 | — |
| unconditioned | 0.549 m | 0.362 m | 0.039 | +0.035, +0.577 |
| **conditioned, relative frame** | **0.132 m** | **0.092 m** | 0.010 | **+0.921, +0.986** |
| ORACLE — ground-truth tokens | 0.107 m | 0.074 m | 0.008 | — |

Goal-error **0.164 → 0.132 m** (−19.5%) versus the frozen tokenizer, and conditioning now
cuts error by **76%** against its own baseline (was 67%). The model/oracle ratio tightened
slightly, 1.32 → 1.23.

This is the composition claim in `04_grounding.md` confirmed end-to-end: goal accuracy was
limited by tokenizer reconstruction, so Stage A improved goal-reaching **without any change
to the grounding mechanism**. Stage A paid for itself downstream.

Not over-read: the unconditioned baseline is nominally worse than on the frozen tokenizer
(0.549 vs 0.490 m), but that gap is ~1.5 SEM and should be treated as noise, not as
evidence the finetune hurt unconditioned generation.

## Reproduce

```
python scripts/track1/prepare_probe_data.py \
    --ckpt   <track2_checkpoints>/net_iter020000.pth \
    --tokens-dir <probe_root>/tokens_finetuned
python scripts/track1/train_probe.py --conditioned --cond-mode rel \
    --tokens-dir <probe_root>/tokens_finetuned --out-name conditioned-rel-ft
python scripts/track1/eval_probe.py --ckpt-name unconditioned-ft conditioned-rel-ft \
    --tokens-dir <probe_root>/tokens_finetuned \
    --vqvae-ckpt <track2_checkpoints>/net_iter020000.pth
```
