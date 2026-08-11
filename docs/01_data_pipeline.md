# 01 — Data pipeline

**Status: DONE, validated at full scale.** Produces the two-track representation everything
downstream consumes. Nothing here is open.

## Result

| what | outcome |
|---|---|
| HUMANISE 3-source ID join (pure_motion / align_data / contact_motion) | 19,648 / 19,648 = 100% |
| 22-joint → 263-dim conversion, vs HumanML3D's own shipped 263 | **0.80 mm** mean MPJPE, foot-contact bit-exact |
| HUMANISE → 263 at full scale | 19,648 / 19,648 converted, 0 NaN, 0 exceptions |
| World-frame track validated by floor overlay on scene meshes | 150 / 150 scenes |
| ScanNet meshes load (trimesh) | 643 / 643 |
| BEV renderer world→pixel error | 0.67 px |

## How

**Two-track storage** (CLAUDE.md 2a). Every clip is kept as (1) canonicalized 263-dim for
the tokenizer, and (2) a world-frame root trajectory `(x, y, yaw)` per frame for placement,
chaining and all scene metrics. The 263-dim is position-invariant, so track 2 is not
redundant — it is the only thing that knows where the motion is.

**Converter** (`src/motion_features.py`). Wraps HumanML3D's own feature extractor rather
than reimplementing it, so the 263 layout is guaranteed to match what the tokenizer
expects. We wrote only adapters: the Z-up→Y-up relabel, reference-skeleton offsets, and a
drift-free local-position reader. The 0.80 mm figure above is that wrapper checked against
HumanML3D's published vectors — it isolates the converter, with no VQ-VAE involved.

**World-frame track** (`src/humanise_join.py`, `compute_track2`). Rebuilt from
pure_motion's SMPL-X translation + orientation composed with align_data's rigid transform.
Validating it by overlaying trajectories on scene-mesh floors is what caught a missing
`scene_translation` offset.

## Conventions that bite

- **HUMANISE raw joint data is Z-up.** Empirically verified; the original Y-up assumption
  was wrong. World frame = ScanNet Z-up, yaw about Z.
- **Yaw is always `(sin, cos)`**, never a raw scalar.
- **`compute_track2` defines world yaw 0 = facing +X**, but the canonicalized frame 0 faces
  −Y in that frame (yaw −π/2). The 90° difference is load-bearing — see `04_grounding.md`
  and CLAUDE.md 2a.

## Known caveats

- The upstream **SMPL-X → 22-joint reduction is HUMANISE's own and is not independently
  verified.** It underlies every HUMANISE number in this repo. It is no longer the prime
  suspect for anything (see `03_tokenizer_finetune.md`), but it has never been checked.
- **2 NaN-corrupted files** exist in the official HumanML3D release — skip or handle them.

## Code

`src/motion_features.py`, `src/humanise_join.py`, `src/bev_render.py`,
`scripts/verify/check*.py`. Checks 18/19 isolate the converter; 7/14/20 add the VQ-VAE.
