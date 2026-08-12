# 08 — Transformer finetune (all conditioning)

**Status: model trained and healthy. Scene conditioning is UNEVALUATED AND CANNOT BE
EVALUATED ON THIS DATA** — established, not assumed; see "why no metric works here". The
blocker is the dataset, not the metric and not the model.

First model trained on all conditioning together, at 5× the probe budget:
CLIP text (512) + relative goal (2) + seam pose (66) + occupancy crop (784) = 1364-d,
20000 iters, on `tokens_finetuned/`.

## Result

300 held-out clips, finetuned tokenizer.

| | seam err | goal err | token acc |
|---|---|---|---|
| **full** (with occupancy) | 71.7 mm | 0.0584 m | 99.6% |
| **noscene** (ablation, identical otherwise) | 71.4 mm | 0.0564 m | 99.5% |
| ORACLE (ground-truth tokens) | 70.9 mm | 0.0544 m | — |
| canonicalization floor | 21.6 mm | — | — |
| *(reference)* step-07 probe, 4000 iters, no scene | 71.2 mm | 0.0573 m | 98.6% |

**The combination does not degrade.** Adding scene conditioning and 5× the training budget
lands within noise of the 4000-iteration step-07 probe. A prior recorded before the run
(`docs/IN_FLIGHT.md`) predicted the combination would underperform the individual probes; it
did not. That prior was wrong and is recorded as wrong.

**Ignore the token accuracy.** 99.5% is expected and uninformative: the seam pose in the
conditioning vector nearly determines the early part of the target. It is not a quality
signal and should never be quoted as one.

## Why the ablation is void

Both models sit **at** the oracle: 0.8 mm of headroom on seam, 4 mm on goal. There is
nothing left for an ablation to move. `noscene` scoring marginally *better* than `full` is
noise, not evidence that occupancy hurts — and equally, **this is not evidence that scene
conditioning fails.** The instrument cannot resolve the question. 2¼ hours of compute went
into an ablation that was structurally incapable of answering it. That is a design error,
not a model result.

Worse, goal error is the wrong question for a scene arm in the first place. Occupancy tells
the model where the furniture is; goal error only asks whether a coordinate was reached and
is perfectly satisfied by a path straight through a sofa.

## The collision metric, and why it also failed

The obvious fix — measure non-collision — was built (`scripts/continuation/eval_collision.py`)
and is **invalid on HUMANISE as defined**. Ground-truth motion, collision rate by action:

| action | whole clip | first third | n |
|---|---|---|---|
| lie | **100.0%** | 100.0% | 50 |
| sit | 86.4% | 85.6% | 120 |
| stand up | 47.6% | 57.7% | 53 |
| walk | 24.6% | 24.0% | 177 |

Real human lying-down motion collides 100% of the time, because the person is on the bed.
On an interaction dataset, **being inside furniture is the objective**, so a
root-in-occupied-pixel metric scores success as failure. The whole-clip and first-third
columns barely differ, so this is not a matter of trimming terminal frames either.

### Fixing the metric — two attempts, one works

**Attempt 1, exclude the target instance: BLOCKED.** The principled fix is to remove the goal
object's ScanNet instance from the occupancy map. Neither box has the instance segmentation —
the local mirror holds only `*_vh_clean_2.ply`, with no `*.aggregation.json` or `*.segs.json`,
and HUMANISE's `pred_contact` is a distance field, not a mask. Would need ScanNet download
credentials.

**Attempt 2, connected-component proxy: FAILED, and was abandoned rather than tuned.**
Exclude the occupied component the goal sits in — if the goal is on a bed, that blob is the
bed. In practice furniture touches walls, so **331 of 400 clips** produced a component larger
than a quarter of all occupied space. Only 1.1% of occupied area got excluded and `lie`
stayed at 98%. Raising the size guard would have "worked" by excluding the walls themselves,
which is why it was dropped. Kept as `src/target_occupancy.py` with the failure documented.

**Attempt 3, tall-obstacle map: WORKS.** Raise the obstacle height threshold so low furniture
drops out and walls survive. Ground-truth collision by threshold (`render_occupancy` now takes
`obstacle_height_m`):

| threshold | lie | sit | stand up | walk | ALL | raster occupied |
|---|---|---|---|---|---|---|
| 0.12 m (default) | 100.0% | 88.6% | 44.5% | 30.2% | 56.7% | 22.7% |
| 0.60 m | 88.5% | 19.1% | 22.6% | 0.1% | 17.8% | 12.3% |
| **0.90 m** | **20.1%** | **3.9%** | **0.0%** | **0.1%** | **3.2%** | **6.1%** |
| 1.20 m | 11.5% | 2.6% | 0.0% | 0.0% | 1.9% | 2.9% |
| 1.50 m | 0.0% | 2.6% | 0.0% | 0.0% | 0.8% | 1.5% |

0.90 m is the pick: walls survive at 6.1% of the raster while beds/sofas/chairs drop out.
Higher thresholds look better only because the map is emptying — at 1.50 m just 1.5% is
occupied and nothing can be detected. `lie` keeps a 20% residual, most likely high beds and
headboards clearing 0.9 m. Cached for all 643 scenes at `~/wander_data/bev_tall_cache`.

## Why no metric works here — it is the data

With a correctly-defined collision map, the ablation still returns nothing:

| | collision rate |
|---|---|
| full (with scene) | 3.09% |
| noscene | 3.09% |
| ORACLE | 3.09% |
| NULL (never move) | 3.25% |

Identical to two decimals, and **NULL is in the same place** — which is the tell. With a mean
start-to-goal displacement of 0.63 m, the root barely leaves the cell it started in, so
collision is a property of the START POSE, not of the path.

Restricting to clips that actually navigate confirms it: **only 25 of 2962 clips (0.8%) are
`walk` with displacement > 1.5 m.** On those, full 0.66% / noscene 0.50% / ORACLE 0.43% /
NULL 0.00% — the models are marginally worse than ground truth and `full` marginally worse
than `noscene`, at n=25, which supports no conclusion whatsoever.

**This single fact explains all three failed evaluations.** Goal error saturates because the
goals are 0.6 m away. Collision is start-determined because nothing moves. Scene conditioning
has almost nothing to do, because there is almost no navigating to do. It is not evidence
that occupancy conditioning fails — it is the absence of a test.

### What this implies

- **HUMANISE alone cannot demonstrate scene-aware navigation.** Any claim of that form needs
  longer trajectories than this data contains per segment.
- **Chaining is the unlock, not just the next step.** Composing segments is what produces
  multi-metre paths, and therefore the first setting where a scene arm can be measured at all.
  Evaluate scene conditioning again after step 9, on chained rollouts, not before.
- **Check how PSMo / AffordMotion define non-collision on HUMANISE** before quoting any
  number against them. Given the above, their protocol cannot be the naive one, and the
  definition will determine comparability.

## Status of each conditioning input

| input | evidence |
|---|---|
| relative-frame goal | **validated** — `04`, `05` |
| seam pose | **validated** — `07` |
| occupancy scene | **representation** validated (`06`); contribution to generation **untestable on this data** — needs chained, multi-metre rollouts |

## Code

`scripts/track1/train_probe.py --cond-mode full`, data from
`scripts/continuation/prepare_continuation_data.py --bev-dir`.
Checkpoints: `~/wander_data/step8/checkpoints/{full,noscene}/`.

```
WANDER_TRACK1_PROBE_ROOT=<step8> python scripts/track1/train_probe.py \
    --conditioned --cond-mode full --tokens-dir <step8>/tokens \
    --out-name full --iters 20000 --save-every 5000
python scripts/continuation/eval_continuation.py --ckpt-name full noscene \
    --ckpt-root <step8>/checkpoints --tokens-dir <step8>/tokens \
    --vqvae-ckpt <track2>/net_iter020000.pth
```
