# In flight — read this if you are picking up cold

Volatile state that is NOT captured by RESULTS.md: what is running right now, where things
live on the boxes, and the next concrete action.
**Update or delete this file when the work it describes lands.**

Last updated: 2026-08-12, after step 9.

---

## Nothing is running. Box is idle (270 MiB, no jobs).

```
ssh train-3090 'pgrep -af "train_probe|demo_rollout|render_"; nvidia-smi --query-gpu=memory.used --format=csv,noheader'
```

**Gotcha when writing a wait loop.** `while pgrep -f foo.py >/dev/null; do sleep 60; done`
never exits: the loop's own command line contains "foo.py", so pgrep matches the watcher
itself. Several waiters this session hung on that and had to be killed by PID. Use a pattern
that cannot match the watcher — e.g. `pgrep -f "[f]oo.py"` — or poll a sentinel in the log
(`grep -q '^saved ' log`) instead of the process table.

## Next action — goal generalization, BEFORE steps 10/11

RESULTS §9 found the model follows in-distribution goals (−78.9% vs a never-move null) but
**arbitrary goals not at all** (+4.5%, i.e. no better than standing still). Real clip text
does not fix it, so it is the goals, not the text. This gates the demo and step 11, because
an MLLM planner emits arbitrary coordinates.

Cheapest test of the leading hypothesis — the model never saw a goal that was not the endpoint
it was already reaching:

1. Retrain `--cond-mode full` with **goal augmentation**: with probability p, replace the
   training goal with a perturbed or resampled one and keep the target motion. If goal-error
   on free waypoints drops below the null, the diagnosis is confirmed and the fix is data.
2. Re-measure with `scripts/chaining/demo_rollout.py --n-rollouts 30 --min-step 0.6
   --max-step 1.2`, which prints the straight-line control alongside.
3. The never-move null for those settings is ~0.925 m. Beat it by a wide margin or the
   problem is not solved.

Waypoint spacing must stay a **band** at the trained displacement scale; "anywhere beyond
min_step" gives ~4 m hops that the model ignores, and the resulting error says nothing.

Then step 10 (collision-guided decoding) — it now has a measured justification: passive scene
conditioning does not steer, and both models collide MORE than the 1.09% straight-line
control. That is the number to beat.

## Ready to use

- Collision metric: `~/wander_data/bev_tall_cache` (0.9 m threshold, 643 scenes);
  `scripts/continuation/eval_collision.py` points at it. **Never score on the 0.12 m map.**
- Chaining: `scripts/chaining/{rollout,eval_accumulation,demo_rollout,render_chain_video}.py`.
- Step-8 models: `~/wander_data/step8/checkpoints/{full,noscene}/`.

## Not worth redoing

- **Target-instance exclusion for collision** — blocked, no ScanNet instance segmentation on
  either box. The connected-component proxy failed (331/400 clips merged furniture with
  walls); it survives as `src/target_occupancy.py` with the failure documented.
- **Scene ablation on single HUMANISE segments** — every metric is saturated or
  start-pose-determined there (RESULTS §8). Only chained rollouts can resolve it.

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
