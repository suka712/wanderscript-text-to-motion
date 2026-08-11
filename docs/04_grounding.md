# 04 — Grounding probe

**Status: DONE. Grounding works.** A transformer told where to go, goes there. This retires
what was the #2 project risk and confirms the architecture in CLAUDE.md 2b. Do **not** pivot
to trajectory-first generation.

## Result

Frozen tokenizer, single segment, 200 held-out HUMANISE clips, greedy decoding, SE(2)-placed
at the fed start pose:

| | goal-error mean | median | SEM | corr(commanded, achieved) |
|---|---|---|---|---|
| NULL — stay at start | 0.627 m | 0.378 m | 0.050 | — |
| unconditioned | 0.490 m | 0.313 m | 0.037 | +0.068, +0.620 |
| conditioned, **absolute** frame | 0.515 m | 0.318 m | 0.038 | −0.123, +0.604 |
| conditioned, **relative** frame | **0.164 m** | **0.108 m** | 0.014 | **+0.903, +0.972** |
| ORACLE — ground-truth tokens | 0.124 m | 0.082 m | 0.010 | — |

The relative-frame model lands **within 4 cm of the tokenizer's own reconstruction floor** —
a 67% reduction against the unconditioned baseline, ~9 SEM. Its token accuracy is also
higher (86.3% vs 77.5% / 77.0%): once the goal is in a usable frame it genuinely predicts
which tokens come next.

The correlation column is the cleanest read. The unconditioned model's +0.62 on the second
axis is just the "people walk forward" prior; its +0.07 laterally means it has no idea where
the goal is. Absolute-frame conditioning is no better. Relative-frame is +0.90/+0.97.

## The one design decision that matters

**Feed the goal in the frame the model generates in** — start-relative and heading-aligned
(`se2_utils.world_to_local_xy`), not absolute world coordinates.

The motion representation is canonicalized: the model generates in a frame whose origin and
heading are the start pose. Handing it absolute coordinates means it must compute
`R(yaw₀)ᵀ·(goal − start)` — bilinear in its own inputs — through a single `Linear`, which
cannot represent a rotation. It does not learn this, and the result (0.515 m) is
indistinguishable from not conditioning at all.

In the relative frame the start pose carries no information — it is (0,0) at heading 0 by
construction — so it is **not** a model input. It enters only at inference, as the SE(2)
placement.

**This generalizes to chaining.** The prefix from segment *k* must likewise be expressed in
segment *k+1*'s own start frame.

## Does it hold at range?

Mean displacement in HUMANISE is only 0.63 m, so a good aggregate could hide a model that
only handles near-stationary clips. Binned over 600 held-out clips:

| \|goal−start\| | n | model | oracle | null |
|---|---|---|---|---|
| 0.00–0.25 m | 222 | 0.071 m | 0.049 m | 0.068 m |
| 0.25–0.50 m | 113 | 0.098 m | 0.077 m | 0.344 m |
| 0.50–1.00 m | 118 | 0.181 m | 0.159 m | 0.729 m |
| 1.00–2.00 m | 110 | 0.286 m | 0.228 m | 1.471 m |
| > 2.00 m | 37 | 0.508 m | 0.385 m | 2.399 m |

Error grows with distance, but **the oracle grows with it in the same proportion** — the
model/oracle ratio stays at 1.1–1.4 in every bin. Goal-following itself is scale-invariant;
what degrades at range is tokenizer trajectory reconstruction. That is a direct argument for
carrying `03_tokenizer_finetune.md` forward: it raises exactly this ceiling.

**It did.** Re-run on the finetuned tokenizer (`05_token_reextraction.md`), the oracle floor
drops to 0.107 m and goal-error follows it to 0.132 m — with no change to the grounding
mechanism at all.

## Method notes

- **Model**: T2M-GPT's `Text2Motion_Transformer`, unmodified. The goal is concatenated to
  the CLIP text feature before the existing `cond_emb` projection — `clip_dim` 512 → 514.
  No architecture surgery, no block_size change. `cond_emb` is warm-started by copying the
  pretrained weight into the first 512 columns; everything else loads `strict=True`.
- **Text encoder is CLIP ViT-B/32, not T5** — it is what is already pretrained in this
  codebase. Orthogonal to what the probe tests.
- **Splits**: HUMANISE's official `train.txt` / `test.txt`, 16523 / 3125, 0 skipped.
- **Both controls are mandatory.** ORACLE pushes each clip's own ground-truth tokens
  through the identical decode + place path — it is the measurement's noise floor, and if
  it is not small relative to the effect, there is no measurement. NULL is what a "never
  move" policy scores. Start-error is *not* a result: SE(2) placement puts frame 0 on the
  fed start by construction, so it is exactly 0.0 for any model. It is asserted, not
  reported.
- **Placement rotates by `yaw0 + π/2`**, not `yaw0` — the canonicalized frame 0 faces +Z in
  HumanML3D's Y-up frame, which is −Y in the Z-up world frame, while `compute_track2`
  defines yaw 0 as +X. Using `yaw0` rotates every trajectory 90° about the fed start and is
  invisible to a start-error check; the oracle catches it (0.862 m vs 0.124 m).

## What this does not establish

Single segment only — no chaining, which remains the project's one open research risk. No
scene features, no collision awareness: the model reaches coordinates in empty space and
does not know a wall is there. And every fed goal is in-distribution (displacement std
~0.35 / 0.73 m); a commanded 5 m walk is beyond the evidence, with the >2 m bin (n=37) at
the edge of it.

## Code

`scripts/track1/{prepare_probe_data,train_probe,eval_probe,se2_utils,render_probe_video}.py`.
Train: `train_probe.py --conditioned` (`--cond-mode rel` is the default; `abs` reproduces
the absolute-frame run). Eval: `eval_probe.py --ckpt-name <a> <b> ...` evaluates any number
of checkpoints on identical clips and always appends both controls.
