# 08 — Transformer finetune (all conditioning)

**Status: model trained and healthy. The scene-conditioning question is UNRESOLVED, and the
ablation built to answer it does not work.** Read the "why the ablation is void" section
before quoting anything here.

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

### What a valid metric requires

**Collision must be measured against everything EXCEPT the target object.** HUMANISE gives
`object_id` per clip and ScanNet meshes carry instance segmentation, so the occupancy raster
can be rebuilt with the goal instance removed. Then "walked through a sofa on the way to the
chair" counts and "sat on the chair" does not.

This is a prerequisite for step 10 (collision-guided decoding) and for any PSMo /
AffordMotion comparison — those benchmarks must define non-collision this way, and reporting
the naive number would have produced something incomparable to published work while looking
like a result.

## Status of each conditioning input

| input | evidence |
|---|---|
| relative-frame goal | **validated** — `04`, `05` |
| seam pose | **validated** — `07` |
| occupancy scene | **representation** validated (`06`); **contribution to generation UNEVALUATED** |

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
