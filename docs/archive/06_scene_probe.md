# 06 — Scene representation probe

**Status: DONE. Verdict: drop DINOv2 and the RGB BEV render. Use occupancy geometry.**

CLAUDE.md §4 listed the scene encoder as SWAPPABLE and flagged a specific worry — DINOv2 is
trained on natural images, so top-down renders are out of its distribution — that had never
been tested. This is that test, run before the transformer work for the same reason the
grounding probe ran before it: a scene representation that carries nothing is far cheaper to
discover now than at step 9.

## Result

4-way action classification (walk / sit / lie / stand up) from scene context at the goal,
**no text**. Split by scene, 410 train / 175 test scenes, 3298 / 1502 clips. Linear head.

| arm | test acc | walk | sit | lie | stand up |
|---|---|---|---|---|---|
| prior (majority class) | 26.4% | — | — | — | — |
| DINOv2 ViT-B/14 | 34.3% | 0.37 | 0.35 | 0.40 | 0.26 |
| RGB crop, **no encoder** | 36.4% | 0.33 | 0.40 | 0.41 | 0.32 |
| DINOv2 ViT-S/14 | 37.6% | 0.39 | 0.32 | 0.48 | 0.32 |
| **occupancy crop** | **63.0%** | 0.69 | 0.64 | 0.67 | 0.51 |

**DINOv2 is doing no work.** Both variants land within ±2 points of feeding the downsampled
RGB crop directly with no pretrained encoder at all. And the bigger model is *worse* than
the smaller one, which rules out "the encoder was too small" — this is a domain problem, not
a capacity problem, exactly as §4 suspected.

**Occupancy geometry nearly doubles it**, and we already compute it, from the same renderer,
for free.

63% is not a great absolute number, but it is close to the ceiling for a geometry-only
representation on this task: **sit and stand-up happen at the same furniture** and differ
only in the direction of motion, so no static scene crop can separate them. That pair is
where the errors are (stand-up recall 0.51, the lowest), while walk and lie — which do have
distinct geometry — reach 0.69 and 0.67.

## Decision

Replace the BEV-RGB + DINOv2 scene encoder with the **binary occupancy raster, cropped in
the agent's own frame**. This is a SWAPPABLE component being swapped on evidence, which is
what §4 invited.

Three things this buys beyond accuracy:
- **One representation serves two purposes.** The occupancy raster is already required for
  collision-guided decoding and the non-collision metric. The scene encoder now uses the
  same raster instead of a second, parallel one.
- **No ViT in the inference loop**, no 350 MB dependency, no OOD question hanging over the
  scene branch.
- **The RGB render can be dropped from the pipeline entirely** — it is the expensive half
  of rendering (needs an EGL context; occupancy is rasterized directly from mesh triangles).
  Keep `render_rgb` for figures and debugging, not for conditioning.

## Method notes

- **Crops are in the agent's frame** — centered on the goal, rotated by the start heading —
  the same lesson as `04_grounding.md`. A world-axis-aligned crop would force the probe to
  solve a rotation it never faces at inference. Verified independently of scene content:
  a full-extent crop reproduces the raster at 98.6% (residual is nearest-neighbour
  resampling), and a +90° heading rotates the crop by exactly 90° (`rot90 k=3` at 100.0%,
  and *not* k=0, which is what "heading silently ignored" would look like).
- **The split is by scene, not by clip.** Many clips share a scene, so a clip-level split
  lets the probe memorize per-scene appearance and score well while learning nothing
  transferable — the same class of leak as the 45.3 mm baseline in
  `02_baseline_calibration.md`.
- **Text is deliberately excluded.** "lie on the bed" makes the label trivial and would
  measure nothing about the scene features. HUMANISE's own `object_label` confirms the
  labels are affordance-bearing (lie→bed, sit→chair/sofa).
- **The head is linear** and identical across arms, so the result is about the
  representation, not about a head strong enough to learn the task from anything.
- **DINOv2 on torch 1.12**: the repo needs `from __future__ import annotations` (it uses
  3.10 union syntax) and a `scaled_dot_product_attention` shim (torch ≥2.0 API). Both are
  in `scene_probe.py`. Weights load `strict=True`, which is what confirms the architecture
  matches rather than silently falling back to random init.

## What this does not establish

CLS token with a linear head. Patch tokens, a nonlinear head, or a DINOv2 finetuned on
top-down renders could all extract more. But the damning comparison is DINOv2 ≈ raw pixels,
which holds regardless of head — a representation that cannot beat its own unencoded input
is not contributing. Action classification is also a proxy for affordance, not the end task;
it says a representation *can* separate what the goal affords, not that the motion model
will use it well.

## Code

`scripts/scene_probe/render_bev_cache.py` (one-time render cache, 643/643 scenes, 0 failures,
~1.6 s/scene), `scripts/scene_probe/scene_probe.py`.

```
python scripts/scene_probe/render_bev_cache.py --out-dir <bev_cache>
python scripts/scene_probe/scene_probe.py --bev-dir <bev_cache> \
    --cache <crops.pkl> --dinov2-weights-large /tmp/dinov2_vitb14.pth
```
