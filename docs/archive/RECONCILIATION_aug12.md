# Reconciliation — 2026-08-12

Both diagnostic tracks finished. This is the orchestrator's verdict on each, after
auditing the code and re-running the parts that did not hold up. Supersedes the "Next"
section of `REPORT_aug9.md`.

**Headline: both tracks pass. Track 2 as reported; Track 1 only after its measurement was
fixed and its experiment re-posed — it had reported FAILS on an invalid measurement.**

- Track 2 (tokenizer finetune): decision gate MET. Lie 136.9 → 96.3 mm, no forgetting.
- Track 1 (grounding probe): reported FAILS. **Retracted.** Its goal-error metric was
  dominated by a 90° SE(2) placement bug, and its conditioning was posed in a frame that
  made the task require a rotation from a single linear layer. With both fixed, goal-error
  drops 0.490 → **0.164 m** against a 0.124 m noise floor. Grounding works.

Consequence: **do not pivot to trajectory-first.** The architecture in CLAUDE.md 2b stands.

---

## Track 2 — joint VQ-VAE finetune: ACCEPTED, merged to master

Decision gate MET on both criteria, held-out throughout.

| category | frozen (held-out) | finetuned | change |
|---|---|---|---|
| H3D baseline | 56.11 | 56.2 | +0.2% (flat) |
| H3D-lie (n=11, noisy control) | 117.46 | 117.4 | flat |
| HUMANISE walk | 50.20 | 34.1 | **-32.1%** |
| HUMANISE stand up | 66.98 | 53.0 | **-20.9%** |
| HUMANISE sit | 69.55 | 47.6 | **-31.6%** |
| HUMANISE lie | 136.91 | 96.3 | **-29.6%** |

Lie is stable in a 92-101mm band across the last 9 of 11 checkpoints, so this is a
plateau, not a lucky dip. The sit/lie gap was **real codebook coverage**. This also
removes the main piece of evidence that would have motivated an urgent investigation of
the unverified upstream SMPL-X→22-joint reduction (`REPORT_aug9.md` §6) — finetuning the
codebook alone, with no change to that step, closed most of the gap. The step is still
not independently verified; it is just no longer the prime suspect.

### Two corrections this track produced that propagate project-wide

1. **The H3D baseline of 45.3mm was leaked and should stop being quoted.** `check14`
   sampled from all of `new_joint_vecs`, including clips the frozen model trained on. The
   honest held-out figure is **56.11mm**. Every "did general motion regress" claim from
   here on is checked against 56.11, not 45.3. The same applies to the H3D-lie control
   (90.1 → 117.46 held-out, n=11, high variance — treat as indicative only).
2. **`QuantizeEMAReset` does not survive a checkpoint load.** It tracks `init`,
   `code_sum`, `code_count` as plain Python attributes rather than registered buffers, so
   `load_state_dict(strict=True)` restores the `codebook` tensor but leaves `init=False`.
   The first `train()` forward pass then runs the from-scratch init path and overwrites
   all 512 codes from a single batch — verified empirically before any real training.
   Anyone resuming or finetuning this VQ-VAE must seed the EMA accumulators from the
   loaded codebook first (`prepare_quantizer_for_finetune`).

---

## Track 1 — grounding probe: reported FAILS, actually PASSES

The branch reported "grounding FAILS, pivot to trajectory-first." That conclusion was
drawn from a measurement dominated by a bug, on an experiment posed in a frame that made
the task nearly unlearnable. Both are fixed below; the corrected answer is the opposite.

### The bug: a 90° error in SE(2) placement

HumanML3D's `process_file` canonicalizes frame 0 to face **+Z in its Y-up frame** ("All
initially face Z+"). Under this repo's own `zup_to_yup_hml` relabel (`y_hml = z_zup`,
`z_hml = -y_zup`), that direction is **−Y in the Z-up world frame**, i.e. world yaw −π/2.
But `humanise_join.compute_track2` defines yaw = 0 as facing **+X**. So composing a
recovered local trajectory onto a world start pose must rotate by `yaw0 + π/2`.
`se2_place` rotated by `yaw0`, placing every generated trajectory 90° off about the fed
start.

It was invisible to the probe's only sanity check. Start-error is exactly 0.0 under any
rotation, because frame 0 sits at the local origin by construction — the report correctly
noted start-error "can't be anything else," but then treated it as evidence the placement
code was right. It is not: it is insensitive to precisely this class of error.

The control that catches it is a **ground-truth-token round trip** — decode a clip's own
tokens, place them, compare to that clip's own fed goal. It should be near zero. On 200
held-out clips:

| placement yaw offset | mean err | median |
|---|---|---|
| 0° (as shipped) | **0.862 m** | 0.525 m |
| **+90° (correct)** | **0.124 m** | 0.082 m |
| 180° | 0.923 m | 0.527 m |
| −90° | 1.255 m | 0.706 m |

The as-shipped oracle error (0.862 m) is the same magnitude as the goal-errors both
models scored (0.828 / 0.822 m). **The probe was measuring its own placement bug.** The
renders read as "both models walk confidently in the wrong direction" because every
trajectory was rotated 90°.

The branch did run a version of this control during debugging and got 0.46 m, but read it
as a normalization result and moved on. A 0.46 m residual against a mean start-to-goal
distance of 0.63 m should have stopped the run — it is the loudest possible signal that
the pipeline, not the model, was producing the error.

### Re-running the eval with the fix

Training was never affected (placement is eval-only), so the checkpoints stand. Same
checkpoints, same 200 clips, same seed:

| | mean | median | n |
|---|---|---|---|
| ORACLE (GT tokens) | 0.124 m | 0.082 m | 200 |
| conditioned (absolute frame) | 0.515 m | 0.318 m | 200 |
| unconditioned | 0.490 m | 0.313 m | 200 |
| NULL (stay at start) | 0.627 m | 0.378 m | 200 |

What changes: the "both models lose to a trivial stay-in-place policy" claim was an
artifact — both beat it. The measurement now has real headroom (oracle 0.12 ≪ signal 0.5).

What survives at this stage: absolute-frame conditioning still does not help — 0.025 m
*worse* than unconditioned, inside noise (SEM 0.038). So fixing the metric alone does not
rescue the original experiment.

### Why that null result still did not settle the question

The probe fed start and goal as **absolute ScanNet world coordinates**. The model
generates in a canonicalized frame whose origin and heading are the start pose, so to use
that conditioning it must compute `R(yaw₀)ᵀ · (goal − start)` — a **bilinear** function of
its own inputs. It was handed that job through a single `Linear`, which cannot represent a
rotation, via 6 raw dims out of 518 that warm-start near zero, for 4000 iterations. A null
result under those conditions is close to uninformative about whether grounding is
feasible.

The fix is to feed the goal in the frame the model already generates in: start-relative
and heading-aligned (`se2_utils.world_to_local_xy`). That is the exact quantity the
canonicalized representation needs, precomputed rather than demanded of a linear layer.
In that frame the start pose carries no information (it is (0,0) at heading 0 by
construction), so it is dropped from the conditioning vector and enters only at inference,
as the SE(2) placement. This is `train_probe.py --cond-mode rel` (clip_dim 514, now the
default); the original absolute-frame path is kept as `--cond-mode abs` for
reproducibility. The unconditioned baseline is unaffected and is reused, not retrained.

**This is the single most important design lesson from Track 1, and it generalizes:
anything the model must know about geometry should be handed to it in the frame it
generates in.** The same applies to the goal conditioning in the real transformer build,
and to the prefix/continuation conditioning for chaining (CLAUDE.md 2d) — the tail of
segment k must be expressed relative to segment k+1's own start frame, not in world
coordinates.

### Relative-frame result: grounding WORKS

Same 200 held-out clips, same seed, same frozen tokenizer, same 4000 iterations. The only
change is the frame the goal is expressed in.

| | goal-error mean | median | SEM | corr(commanded, achieved) |
|---|---|---|---|---|
| NULL (stay at start) | 0.627 m | 0.378 m | 0.050 | — |
| unconditioned | 0.490 m | 0.313 m | 0.037 | +0.068, +0.620 |
| conditioned, **absolute** frame | 0.515 m | 0.318 m | 0.038 | −0.123, +0.604 |
| conditioned, **relative** frame | **0.164 m** | **0.108 m** | 0.014 | **+0.903, +0.972** |
| ORACLE (GT tokens) | 0.124 m | 0.082 m | 0.010 | — |

**The relative-frame model lands 0.164 m from the fed goal against a noise floor of
0.124 m.** It is within 4 cm of the best any model could do through this tokenizer. Token
accuracy also jumped (86.3% vs 77.5% absolute-frame, 77.0% unconditioned) — the goal is
genuinely informative about which tokens come next once it is expressed in a frame the
model can use.

The correlation column is the cleanest read. The unconditioned model's +0.62 on the second
axis is not grounding, it is the "people mostly walk forward" prior; its +0.07 on the
lateral axis shows it has no idea where the goal is. The absolute-frame model is no better
(−0.12, +0.60) — confirming it extracted essentially nothing from the coordinates it was
handed. The relative-frame model is at +0.90/+0.97: it goes where it is told.

### Does it hold at longer range?

Mean displacement in this data is only 0.63 m, so a good aggregate could hide a model that
only handles near-stationary clips. Binned by commanded displacement (600 held-out clips):

| \|goal−start\| | n | model err | oracle err | null |
|---|---|---|---|---|
| 0.00–0.25 m | 222 | 0.071 m | 0.049 m | 0.068 m |
| 0.25–0.50 m | 113 | 0.098 m | 0.077 m | 0.344 m |
| 0.50–1.00 m | 118 | 0.181 m | 0.159 m | 0.729 m |
| 1.00–2.00 m | 110 | 0.286 m | 0.228 m | 1.471 m |
| > 2.00 m | 37 | 0.508 m | 0.385 m | 2.399 m |

Grounding holds across the whole range. Error grows with distance — but **the oracle grows
with it in almost exactly the same proportion** (model/oracle ratio stays at 1.1–1.4 in
every bin). The model is not getting worse at following the goal at longer range; the
tokenizer is getting worse at reconstructing longer trajectories. Goal-following itself is
scale-invariant here.

### Verdict: Track 1 PASSES, on the corrected experiment

The decision gate in `001_grounding_probe.md` asks whether conditioning meaningfully
reduces goal-error versus an unconditioned baseline. It does: 0.490 → 0.164 m, a 67%
reduction, ~9 SEM. The original "FAILS → pivot to trajectory-first" verdict is withdrawn.
**Do not pivot to trajectory-first.** The mechanism CLAUDE.md 2b specifies — explicit
start/goal as learned embeddings concatenated to the transformer's conditioning — works as
designed, provided the goal is given in the model's own generation frame.

---

## The two tracks compose

Reconciliation table row (`REPORT_aug9.md`): **grounding works + finetune improves →
"tokenizer optional — ship frozen, finetune is a quality bump."** That is the right call,
and the displacement breakdown above sharpens *why* the finetune is worth taking: at every
range, what limits goal accuracy is tokenizer reconstruction, not grounding. Track 2 cut
interaction reconstruction error ~30% with no forgetting. So the two results are not
competing — Track 2 raises exactly the ceiling Track 1 is now pressed against.

Recommended build: **use the finetuned tokenizer**, re-extract tokens with it (mandatory,
CLAUDE.md 2b strict ordering), and train the real transformer with relative-frame goal
conditioning.

### What is still open — do not overclaim from this

The probe deliberately tested one thing. All of the following are untested:

- **Single segment only.** No chaining, no conditional continuation. That remains the
  hardest open problem (CLAUDE.md 5.1) and nothing here speaks to it.
- **No scene conditioning**, no collision awareness. The model reaches coordinates in
  empty space; it does not know a wall is there.
- **Goals are in-distribution.** Every fed goal is the clip's own endpoint, drawn from a
  displacement distribution with std ~0.35/0.73 m. Generalization to goals the training
  distribution does not cover (a commanded 5 m walk) is not established — the >2 m bin
  (n=37) is the edge of the evidence.
- **CLIP text, not T5.** The probe used what is already pretrained in this codebase.

---

## Standing hazard: this project keeps shipping frame/normalization bugs

Four now, all the same shape — a silent convention mismatch that degrades a number without
crashing anything:

1. Y-up vs Z-up assumption for HUMANISE raw joints (STEP1, caught).
2. Wrong mean/std in the MPJPE canary — 137mm read as real, actually 45mm (STEP2, caught).
3. The same normalization error again, in Track 1's encode/decode (caught by that track).
4. The 90° SE(2) yaw-convention error (this document — **not** caught by that track).

The pattern in the misses is that each was checked with a metric that could not detect it.
Start-error cannot see a rotation. A per-frame MPJPE cannot see cumulative trajectory
drift. The countermeasure is cheap and should be mandatory from here: **every pipeline
that produces a number gets an oracle control** — push ground truth through the identical
path and confirm the result is near-perfect before reading any model number off it. Track
2 did this by construction (its iter-0 eval is the frozen model through the same code) and
produced no such bug.
