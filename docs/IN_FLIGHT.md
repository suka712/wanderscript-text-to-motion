# In flight — read this if you are picking up cold

Volatile state that is NOT captured by the numbered stage docs: what is running
right now, where things live on disk, and what the next concrete action is.
**Update or delete this file when the work it describes lands.**

Last updated: 2026-08-12, after step 8 finished.

---

## Nothing is running.

Step 8 completed (both `full` and `noscene`, 20000 iters each). Results and the reasons the
ablation is void are in `docs/08_transformer_finetune.md`.

Check the box is idle before starting anything:
```
ssh train-3090 'pgrep -af train_probe.py; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'
```

## Next action — pick one

**Step 9, chaining.** Multi-segment rollout + SE(2) + seam blend on the step-8 `full` model.
Two things ride on it, not one:

1. Accumulation over N segments — the open research question. The continuation mechanism is
   proven at ONE seam only (docs/07). Feed the DECODED pose forward, never a blended one
   (CLAUDE.md 2d point 2).
2. It is the only way to evaluate scene conditioning. Chained rollouts are what produce
   multi-metre paths; HUMANISE segments alone average 0.63m, which is why every step-8
   evaluation came back empty (docs/08). Re-run `eval_collision.py` on chained rollouts.

The collision metric is ready: `~/wander_data/bev_tall_cache` (0.9m threshold, 643 scenes),
`scripts/continuation/eval_collision.py` already points at it.

Not worth doing: target-instance exclusion. Attempted and blocked — no ScanNet instance
segmentation on either box. The connected-component proxy was tried and failed (331/400 clips
merged furniture with walls); it survives as `src/target_occupancy.py` with the failure
documented so nobody retries it.

## Where things live on the 3090

All paths under `~/wander_data/` unless noted. None of it is in git.

| path | what |
|---|---|
| `motion_data/` | H3D, HUMANISE, scannet, `HUMANISE_263_cache` |
| `motion_data/track2_checkpoints/net_iter020000.pth` | **the finetuned VQ-VAE** — used by everything downstream |
| `track1_probe/tokens/` | tokens from the FROZEN tokenizer (docs/04) |
| `track1_probe/tokens_finetuned/` | tokens from the finetuned tokenizer (docs/05) |
| `track1_probe/checkpoints/` | `unconditioned`, `conditioned`, `conditioned-rel`, `unconditioned-ft`, `conditioned-rel-ft` |
| `continuation/tokens`, `continuation/checkpoints/` | docs/07 — `noprefix`, `continuation` |
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

1. **Scene conditioning is unevaluated** — see docs/08. Needs target-excluded occupancy.
2. **Nothing has been chained.** Steps 4/6/7 are single-segment or one-seam probes.
   Accumulation over N segments is the open research question (CLAUDE.md risk #1).
3. **No comparison to published work.** Generation FID unreproduced after 5 attempts;
   PSMo / AffordMotion untouched. Done-criterion #1 is half met.
4. **Single seed, single config everywhere.** Every probe is one run. Large effects
   (155.8→71.2, DINOv2≈raw pixels) would survive a reseed; smaller ones quoted in the docs
   (0.164→0.132, ratio 1.32→1.23) are point estimates with no run-to-run variance measured.
5. **No independent review.** The 90° SE(2) bug was found by auditing someone else's code.
   The code written since has been checked only by its own author, with oracle controls as
   the main defense.
