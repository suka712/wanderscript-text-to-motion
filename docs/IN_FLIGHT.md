# In flight — read this if you are picking up cold

Volatile state that is NOT captured by RESULTS.md: what is running, where things live on the
boxes, and the next concrete action. **Update or delete this file when its work lands.**

Last updated: 2026-08-13, after step 10.

---

## Nothing is running.

```
ssh train-3090 'pgrep -af "train_probe|demo_rollout|render_"; nvidia-smi --query-gpu=memory.used --format=csv,noheader'
```

**Wait-loop gotcha.** `while pgrep -f foo.py; do sleep 60; done` never exits — the loop's own
command line contains "foo.py" so pgrep matches itself. Use `pgrep -f "[f]oo.py"` or poll a
log sentinel (`grep -q '^saved ' log`).

## Next action — collision-guided decoding (step 11)

**It is the only thing gating a demo.** Goal-following and chaining are both fixed; the model
goes where it is told and chains cleanly, but it does not steer around furniture.

Target, already measured: beat the **1.09%** straight-line control. Current models sit at
2.07-2.61%, i.e. worse than walking directly between waypoints.

Approach per CLAUDE.md 2e: at each AR step take top-k tokens, decode each candidate's root
movement, check against the 0.9 m tall-obstacle map, re-rank to prefer non-colliding.
**Rejection sampling over whole chained rollouts is the guaranteed floor** — generate N
chains, keep the lowest-collision one — and is worth measuring first as a baseline.

Build on `~/wander_data/step10/checkpoints/goalaug` (the goal-augmented model, best available).
Measure with `scripts/chaining/demo_rollout.py --n-rollouts 30 --min-step 0.6 --max-step 1.2`,
which prints the straight-line control alongside.

## Ready to use

- **Best model**: `~/wander_data/step10/checkpoints/goalaug` — full conditioning + goal
  augmentation. 0.374 m on arbitrary goals, 0.0560 m on familiar ones, 78 mm seams.
- **Collision map**: `~/wander_data/bev_tall_cache` (0.9 m threshold, 643 scenes).
  **Never score on the 0.12 m map** — it counts sitting on the target as a collision.
- **Chaining**: `scripts/chaining/{rollout,eval_accumulation,demo_rollout,render_chain_video}.py`
- **Data with goal augmentation support**: `~/wander_data/step10/tokens` (carries `xy_traj`)

## Not worth redoing

- **Target-instance exclusion for collision** — blocked, no ScanNet instance segmentation on
  either box. The connected-component proxy failed (331/400 clips merged furniture with walls);
  it survives as `src/target_occupancy.py` with the failure documented.
- **Scene ablation on single HUMANISE segments** — every metric is saturated or
  start-pose-determined there (RESULTS §8). Only chained rollouts resolve it.
- **Randomising the goal while keeping the motion** — teaches the model to ignore the goal.
  Truncation augmentation is the correct form (RESULTS §10).

## Where things live on the 3090

All paths under `~/wander_data/` unless noted. None of it is in git.

| path | what |
|---|---|
| `motion_data/` | H3D, HUMANISE, scannet, `HUMANISE_263_cache` |
| `motion_data/track2_checkpoints/net_iter020000.pth` | **the finetuned VQ-VAE** — used by everything downstream |
| `track1_probe/tokens/` | tokens from the FROZEN tokenizer (RESULTS §4) |
| `track1_probe/tokens_finetuned/` | tokens from the finetuned tokenizer (RESULTS §5) |
| `track1_probe/checkpoints/` | `unconditioned`, `conditioned`, `conditioned-rel`, `unconditioned-ft`, `conditioned-rel-ft` |
| `continuation/tokens`, `continuation/checkpoints/` | RESULTS §7 — `noprefix`, `continuation` |
| `step8/tokens`, `step8/checkpoints/` | step 8 — `full` and `noscene`, both complete |
| `bev_cache/` | 643 scene renders (rgb + occupancy + extent), ~1.6 s/scene to regenerate |
| `deps/` | DINOv2 weights (ViT-S/14, ViT-B/14) + the patched repo — **moved out of /tmp** |
| `report_videos/`, `report_gallery/` | rendered outputs, synced to the Mac at `~/Documents/wander-output` |

### DINOv2 gotchas (env is Python 3.8 / torch 1.12)

`deps/dinov2_repo` is the upstream repo **patched** to run here — every file has
`from __future__ import annotations` prepended (upstream uses 3.10 `X | None` syntax).
It also needs a `scaled_dot_product_attention` shim (torch ≥2.0 API); that lives in
`scripts/scene_probe/scene_probe.py::_install_sdpa_shim`. Weights load `strict=True`,
which is the check that the architecture matches rather than silently falling back to
random init.

### Network

The per-flow throttle in CLAUDE.md §8 is real but is **not** a bandwidth cap. `aria2c -x8 -s8`
pulled 88 MB in <25 s and 346 MB in ~2 min. Use it for large single files.

---

## Known gaps, in priority order

1. **Scene conditioning is unevaluated and untestable on HUMANISE** — see RESULTS §8. Only
   0.8% of clips walk >1.5m; needs chained rollouts.
2. **Nothing has been chained.** Every result is single-segment or one-seam.
   Accumulation over N segments is the open research question (CLAUDE.md risk #1).
3. **No comparison to published work.** Generation FID unreproduced after 5 attempts;
   PSMo / AffordMotion untouched. Done-criterion #1 is half met.
4. **Single seed, single config everywhere.** Every probe is one run. Large effects
   (155.8→71.2, DINOv2≈raw pixels) would survive a reseed; smaller ones quoted in the docs
   (0.164→0.132, ratio 1.32→1.23) are point estimates with no run-to-run variance measured.
5. **No independent review.** The 90° SE(2) bug was found by auditing someone else's code.
   The code written since has been checked only by its own author, with oracle controls as
   the main defense.
