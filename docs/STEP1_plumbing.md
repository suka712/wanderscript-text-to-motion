# Step 1 — Verify Plumbing

Status: **DONE.** Referenced from CLAUDE.md section 3 ("what is validated so far —
do not redo, do not re-litigate"). This file is the historical record: what was
asked, what was found, what was built.

---

## Ask

Before writing any pipeline/model code, confirm four load-bearing assumptions about
the data on disk (`/media/user/2tb/motion_data/`):

1. Does a **world-frame** per-frame absolute `(x, y, yaw)` root trajectory exist for
   HUMANISE, separate from the canonicalized HumanML3D 263-dim tensor? This is the
   single most important check — everything downstream (placement, chaining,
   start/goal conditioning, collision metrics) needs it.
2. Has HUMANISE motion been converted to the exact **263-dim HumanML3D format** the
   VQ-VAE expects?
3. Does the T2M-GPT VQ-VAE **reconstruct HUMANISE motion acceptably** (encode/decode
   canary), or does HUMANISE fall outside HumanML3D's codebook coverage?
4. Can a **BEV occupancy raster** (aligned to world frame) actually be rendered from
   the ScanNet meshes?

Rule: any FAIL gets surfaced and stopped on, never silently worked around.

## Findings

**1. World-frame trajectory — not in the preprocessed tensors, reconstructable from raw data.**
`HUMANISE/contact_motion/motions/*.npy` (the tensor actually used for training) is
canonicalized — no absolute position. But the raw ingredients exist and were used to
build a faithful reconstruction:
- `HUMANISE/pure_motion/<action>.zip/<clip>/motion.pkl` — raw per-frame SMPL-X root
  translation `(T,3)` + orientation `(T,3)` axis-angle.
- `HUMANISE/align_data_release/<action>.zip/<clip>/anno.pkl` — the rigid transform
  (scalar yaw + translation + scene_translation) placing a clip into its ScanNet scene.

Built as `src/humanise_join.py`: reconstructs per-frame `(x, y, yaw)` via this rigid
transform, yaw stored as `(sin, cos)`. **ID-join validated at full scale across all
three sources: 19,648 / 19,648 clips, 0 misses** (`scripts/verify/check6_id_join_full.py`).

**Axis convention — corrected during this work.** HUMANISE's raw joint data is
natively **Z-up** (verified via anatomical joint-height ordering: head z≈1.5, feet
z≈0.05, and cross-checked against HUMANISE's own placement code). The original
assumption that raw data was Y-up and needed conversion was **wrong** — there is no
Y-up→Z-up step; HUMANISE's own alignment code already rotates about Z. World frame =
ScanNet Z-up throughout.

Floor-overlay validation (`scripts/verify/check8_track2_overlay.py`): reconstructed
Track-2 trajectories overlaid on ScanNet mesh floor bounds, **150/150 scenes pass**
(trajectories sit on the floor, not floating/sunk). This check caught and fixed a
missing `scene_translation` offset in the placement transform.

**2. 263-dim conversion — built by reuse, not hand-written.**
`src/motion_features.py` feeds HUMANISE's `(T,22,3)` joint positions into
HumanML3D's own feature-extraction code (vendored, not reimplemented), so the 263-dim
layout is guaranteed to match what the VQ-VAE expects. **All 19,648 clips convert
cleanly, 0 NaN/exceptions.**

Also found: **2 files in the official HumanML3D release are NaN-corrupted**
(`007975.npy` and its mirror `M007975.npy`) — unrelated to HUMANISE, a pre-existing
data defect. Patched `T2M-GPT/dataset/dataset_TM_eval.py` to skip non-finite motions
with a logged warning.

**3. Reconstruction canary — codebook coverage gap, uneven by action.**
Ran the frozen T2M-GPT VQ-VAE (checkpoint fetched via the repo's own
`download_model.sh`) encode→decode on both H3D and converted HUMANISE
(`scripts/verify/check7_vqvae_canary.py`):

| Category | MPJPE (original) | vs. H3D baseline |
|---|---|---|
| H3D baseline (walk-like) | 137.3mm | — |
| HUMANISE walk | ~112mm | fine |
| HUMANISE stand-up | ~192mm | 1.4x |
| HUMANISE sit | ~293mm | 2.1x |
| HUMANISE lie | ~703mm | 5.1x, visually broken |

This result has since been reinterpreted (see CLAUDE.md): it was the reason the
VQ-VAE gets jointly finetuned on HumanML3D + HUMANISE rather than a reason to scope
sit/lie out of the project.

**Correction (found during STEP2's per-category FID work, `check12`-`check15`):** the
normalization above (`H3D_ROOT`'s `Mean.npy`/`Std.npy`) is NOT what this VQ-VAE
checkpoint's own evaluator expects — the checkpoint has its own stats at
`checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/{mean,std}.npy`, which differ
enough (max abs diff 0.028 in mean, 0.34 in std) to substantially inflate error. Found
via a control test: the exact same eval code run on H3D itself gave FID=5.25 with the
wrong stats and FID=0.072 (matches the paper's 0.070) with the right ones. Re-ran the
MPJPE canary with the correct stats, all categories (`scripts/verify/check14_mpjpe_evaluator_norm.py`):

| Category | MPJPE (corrected) | vs. H3D baseline |
|---|---|---|
| H3D baseline | **45.3mm** (was 137.3mm) | — |
| HUMANISE walk | 47.9mm | 1.06x |
| HUMANISE stand-up | 67.6mm | 1.49x |
| HUMANISE sit | 72.6mm | 1.60x |
| HUMANISE lie | **139.8mm** (was ~703mm) | **3.09x** (was 5.1x) |

H3D's own baseline dropped 137.3→45.3mm, confirming the normalization mismatch affected
the *original* canary too, not just the later FID work. Lie is still the worst category
by a real margin, but "5.1x, structurally broken" was substantially a normalization
artifact — the corrected picture is "3.09x, moderately worse." A 10-clip visual
spot-check of `lie` (`scripts/verify/check15_lie_ten_samples.py`) found per-clip MPJPE
ranging 65-246mm with no catastrophic outliers (mean 135.6mm, std 56mm) — every
reconstruction preserved the same overall pose family as its real counterpart, with
moderate joint-angle deviations, not structural breakage. This reads as uniform moderate
degradation, not a broken subset dragging the average up.

**This matters:** the corrected numbers are a materially weaker version of the
justification CLAUDE.md originally gave for finetuning the VQ-VAE rather than keeping it
frozen. The directional finding stands (interaction reconstructs worse than locomotion,
worst on lie) but the magnitude does not support "structurally broken" as stated. See
CLAUDE.md section 2b for how this is now worded, and docs/STEP2_baseline_calibration.md
for the FID side of this same correction.

**4. BEV meshes — present; renderer built and validated (done opportunistically
while GPU-blocked on Step 2).**
All 643 HUMANISE scene IDs have a matching ScanNet `_vh_clean_2.ply` mesh
(`scripts/verify/check4_mesh_check.py`), all load via `trimesh`, Z-up, floor near
z≈0.

Renderer: `src/bev_render.py`, using `pyrender` for the RGB pass (headless via EGL —
confirmed working through the NVIDIA driver's `libEGL`, no display needed) and direct
matplotlib triangle rasterization for occupancy (not the GL depth buffer — pyrender's
orthographic depth had unusable metric precision on this scene scale; rasterizing the
mesh's own height-sliced triangles is exact and sidesteps it entirely). Both rasters
share one square world-frame extent and pixel resolution, so they are pixel-aligned by
construction.

- RGB: top-down photographic render with geometry above 2m clipped (so the camera
  sees floor/furniture, not the ceiling it would otherwise be looking at).
- Occupancy: floor-level triangles (≤0.12m above the scene floor) mark walkable area;
  triangles between 0.12m and 2m mark occupied obstacles, drawn on top. Anything
  outside the room's footprint or a scan hole defaults to occupied (safe default).
- World→pixel mapping: `col = (x-xmin)/(xmax-xmin)*W`, `row = (ymax-y)/(ymax-ymin)*H`.
  Verified empirically (not just derived): a marker placed at a known world position
  lands within **0.67px** of the formula's prediction, consistently across scenes
  (`scripts/verify/check11_bev_render.py`).
- Batch-tested across 33 scenes (every 20th, sorted): **0/33 failures**, ~0.8s/scene,
  occupied-area fraction 11–41% (mean ~22%) — a plausible range for room scans with
  furniture concentrated along walls.
- Visual spot-check: occupancy correctly hugs walls and furniture footprints (bed,
  couches) while leaving open floor clear — see composite overlays in
  `scratch_outputs/bev/` (gitignored, regenerate via check11).

## Verdict

Checks 1, 2, 3 all surfaced real, buildable gaps rather than dead ends — none forced
a design change at the time. Check 3's uneven-reconstruction finding is what later
triggered the VQ-VAE-frozen → VQ-VAE-finetuned pivot (see CLAUDE.md). Step 1 is
closed; nothing here should be redone.
