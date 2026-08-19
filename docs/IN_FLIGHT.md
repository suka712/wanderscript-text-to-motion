# In flight — read this if you are picking up cold

Volatile state that is NOT captured by RESULTS.md: what is running, where things live on the
boxes, and the next concrete action. **Update or delete this file when its work lands.**

Last updated: 2026-08-18. Steps 1-11 done. **Step 11 (interaction in a chain) is DONE —
done-criteria 4 and 5 MET.** Watchable walk→sit→stand→walk clips at
`~/wander_data/step11_demo_multiseg/` (best: `demo_00` scene0151/couch, `demo_05`
scene0694/coffee-table, both 0% collision). Best interaction model:
`~/wander_data/step11/checkpoints/action`.

---

## Nothing is training right now. (An eval may be running — check.)

```
ssh train-3090 'pgrep -af "[t]rain_probe.py|[d]emo_interaction|[e]val_"; nvidia-smi --query-gpu=memory.used --format=csv,noheader'
```

**Wait-loop gotcha.** `while pgrep -f foo.py; do sleep 60; done` never exits — the loop's own
command line contains "foo.py" so pgrep matches itself. Use `pgrep -f "[f]oo.py"` or poll a
log sentinel (`grep -q '^saved ' log`).

## Step-11 result, reproduce commands

Best model `~/wander_data/step11/checkpoints/action` (`cond_mode=full_action`, goal-aug 0.5
walk-only, walk-prefix-aug 0.5). Recipe:
`WANDER_TRACK1_PROBE_ROOT=~/wander_data/step11 train_probe.py --conditioned --cond-mode
full_action --iters 20000 --lr 1e-4 --goal-aug 0.5 --walk-prefix-aug 0.5 --tokens-dir
~/wander_data/step10/tokens --out-name action`.

- Capability (pelvis height, NOT goal error which is z-blind): `eval_sit_capability.py --ckpts
  ~/wander_data/step11/checkpoints/action --vqvae-ckpt <ft-vqvae> --n 60 --actions sit "stand up"
  --prefix-mode walk` → sit **85%** (was 0%); `--prefix-mode own` → **83%** (was 42%).
- Demo: `demo_interaction.py --ckpt ~/wander_data/step11/checkpoints/action --vqvae-ckpt
  <ft-vqvae> --out <dir> --seed-action sit --n-demos 10 --start-dist 3.0 --front 0.3` →
  **5/10** chains SAT and STOOD. The walk-up AUTO-SPLITS into ≤1.1 m hops (a single 3 m walk
  undershoots and leaves the sit goal too long → the model walks instead of sitting).
  `--seed-action lie` for a lie demo (untried — worth a run).

## Next: step 12 (collision-guided decoding) or demo polish

Done-criteria 4/5 are met, so the demo gate is cleared. Remaining build-order items:
- **Step 12 — collision-guided decoding** (the last SWAPPABLE contribution). Target: beat the
  **1.09%** straight-line control; the `full_action` chains sit at ~0–8% collision (some scenes
  worse, e.g. 24% once). Rejection sampling over chained rollouts is the guaranteed floor.
- **Demo polish** (optional): close the composed SAT gap (50% → higher) with end-of-walk
  prefixes in `--walk-prefix-aug` (currently mid-stride only) or a higher aug probability; a
  `lie` demo; nicer camera. See RESULTS §11 "honest gap".
- **Step 13 — Qwen JSON** end-to-end now emits `{action, goal_coord}` per segment, which maps
  directly onto `demo_interaction`'s (action, goal) segments — the action one-hot is exactly the
  MLLM's `action` field.

## Open limitation — sit orientation (the model ignores which way furniture faces)

The model sits without knowing the furniture's facing, so it can sit backwards. Diagnosed with
`scripts/chaining/diag_sit_facing.py`: it FOLLOWS THE APPROACH DIRECTION (|sit facing − GT| 31°)
and IGNORES a commanded sit facing (14° of 180° flip). Cause: no orientation signal anywhere —
occupancy is a footprint. **Do NOT "fix" the demo by approaching from the GT seated direction —
that is a hack (peeks at GT, doesn't touch the model, doesn't generalize), rejected on
2026-08-20.** Proper fix = an orientation-aware scene rep the model consumes (RGB render /
oriented-object map), or making it depend on an explicit orientation input. RESULTS §11.

## Heading (moonwalk) — FIXED at inference 2026-08-19 (RESULTS §11)

The body did not turn to face travel (|facing−travel| 78°, "moonwalking" on free chains).
- Target-heading CONDITIONING (`cond_mode=full_action_head`, model `step11/checkpoints/head_action`)
  was tried and FAILED: helped at 2k (56°) but the converged 20k model ignored it (76°) — the
  heading target is redundant with goal+prefix on GT, so it gets no gradient. Don't re-add it.
- The INFERENCE re-orient fixes it: `rollout(..., reorient=True)` (exposed as `--reorient` on
  `demo_interaction.py` and `diag_heading.py`) rotates each walk segment's start to face its
  goal. |facing−travel| 78°→**3°**, keeps clean seams (prefix is heading-canonicalized).
  Reoriented demo: `~/wander_data/step11_demo_reorient/`.
- `head_action` and the earlier `action` model behave the same on heading (conditioning ignored);
  use either with `--reorient`. Diagnose with `scripts/chaining/diag_heading.py [--reorient]`.

## The finding, in one paragraph (so a cold reader gets it)

Done-criteria 4/5 were blocked by TWO things, both invisible to goal error (which is (x,y)-only
and cannot see sitting, a z event — RESULTS §11): (1) goal augmentation (§10) trained pure
xy-reaching and suppressed sitting 75%→42%; (2) the walk→sit seam is out-of-distribution —
HUMANISE sit clips start from a standstill, so a mid-stride walking prefix drops sit 70%→0% and
the model just keeps walking. Fix (training-time only, no re-tokenize): `cond_mode=full_action`
(a 4-way action one-hot as an explicit "sit now" signal) + `--walk-prefix-aug` (swap a walking
prefix onto interaction clips to synthesize the missing seam) + `--goal-aug` restricted to walk
clips. Validated at 2k iters: walking-prefix sit 0%→85%.

## New/changed artifacts this cycle

- `scripts/chaining/demo_interaction.py` — composed walk→sit→stand→walk demo, pelvis-z SAT/STOOD
  structure check (no oracle exists for a composed chain, so structure is the check). Has
  `--start-dist` (synthesize a far chain start so the walk-up is real) and an AUTO-SPLIT walk-up
  into ≤1.1 m hops so the body is delivered to the furniture before the sit.
- `scripts/chaining/eval_sit_capability.py` — single-segment sit/stand by pelvis height, with
  `--prefix-mode {own,walk}` (the prefix-isolation control) and multi-ckpt compare.
- `scripts/chaining/eval_accumulation.py` — additive `--by-action` flag (default output unchanged);
  now loads BEV for `full_action`/`full_action_head` too.
- `scripts/chaining/diag_heading.py` — measures body-facing vs travel direction (the moonwalk
  diagnostic); `--reorient` to test the inference heading fix.
- `scripts/track1/add_goal_heading.py` — adds the target-heading field to a token manifest
  (geometry only, no re-tokenize); produced `~/wander_data/step11/tokens_head`.
- `scripts/track1/train_probe.py` — `cond_mode=full_action` and `full_action_head`,
  `--walk-prefix-aug`, goal-aug now walk-only. `scripts/chaining/rollout.py` —
  `rollout(..., actions=[...], reorient=...)`, `build_cond` action + heading args.
- Best models: `~/wander_data/step11/checkpoints/head_action` (latest; action+heading, use with
  `--reorient`) and `.../action` (action only) — the two behave the same on heading since the
  heading CONDITIONING is ignored at convergence (RESULTS §11). `step10/checkpoints/goalaug`
  remains best for pure navigation but sits only 42%.
  `step10/checkpoints/goalaug` remains best for pure navigation but sits only 42%.

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
