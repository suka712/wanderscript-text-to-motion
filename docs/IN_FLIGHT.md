# In flight — read this if you are picking up cold

Volatile state that is NOT captured by RESULTS.md: what is running, where things live on the
boxes, and the next concrete action. **Update or delete this file when its work lands.**

Last updated: 2026-08-13, end of reporting cycle. Steps 1-10 done.

---

## Nothing is running.

```
ssh train-3090 'pgrep -af "train_probe|demo_rollout|render_"; nvidia-smi --query-gpu=memory.used --format=csv,noheader'
```

**Wait-loop gotcha.** `while pgrep -f foo.py; do sleep 60; done` never exits — the loop's own
command line contains "foo.py" so pgrep matches itself. Use `pgrep -f "[f]oo.py"` or poll a
log sentinel (`grep -q '^saved ' log`).

## Next action — INTERACTION IN A CHAIN. Do this before collision steering.

**Done-criteria 4 and 5 are NOT met and this is why.** Every chained rollout so far is
WALK-ONLY, with goals sampled on free floor. Both criteria require scene *interaction*, and
CLAUDE.md §1 states a walk-only demo is a failure. Chaining itself works — the gap is that
sit/lie inside a chain has never been run.

It is also the **largest remaining unknown**. Sitting down ends in a body pose nothing like
mid-stride, and every continuation result was measured on walking. If that pose transfer
degrades, it is a training-data problem and costs a cycle. Cheap to find out.

What to change in `scripts/chaining/demo_rollout.py`:
1. It filters to `action == "walk"` — remove that.
2. Goals come from free floor (`sample_waypoints`). For interaction, take the goal from the
   clip's own target object instead: HUMANISE gives `object_id` / `object_label` per clip and
   the world-frame track's endpoint is where the person actually interacts.
3. Text is hardcoded `"walk to the target"`. Give each segment action-appropriate text —
   the clip's own utterance for interaction segments.
4. Chain something like walk -> sit -> stand -> walk and check the seams into and out of the
   interaction specifically, not just the average.

Then render with `render_chain_video.py` — a watchable sit-in-a-real-room clip is
done-criterion 5.

## After that — collision-guided decoding

The only other thing gating a demo. Target already measured: beat the **1.09%** straight-line
control (current models sit at 2.07-2.61%, i.e. worse than walking directly).

Approach per CLAUDE.md 2e: at each AR step take top-k tokens, decode each candidate's root
movement, check against the 0.9 m tall-obstacle map, re-rank to prefer non-colliding.
**Rejection sampling over whole chained rollouts is the guaranteed floor** — generate N
chains, keep the lowest-collision one — worth measuring first as a baseline.

Build on `~/wander_data/step10/checkpoints/goalaug`. Measure with
`scripts/chaining/demo_rollout.py --n-rollouts 30 --min-step 0.6 --max-step 1.2`, which
prints the straight-line control alongside.

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
