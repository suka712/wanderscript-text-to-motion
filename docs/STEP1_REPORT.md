# STEP 1 — Plumbing Verification Report

Environment: `python3` = anaconda3 build, `numpy` 1.26.4 available. `torch` NOT
installed in this environment. `trimesh` was not installed but installed cleanly
via pip (internet reachable) and used for the mesh check.

Scripts used (committed under `scripts/verify/`):
- `check1_2_inspect.py` — shape/key inspection of HUMANISE, H3D, PROX, Mean_Std_* files.
- `check1_raw_world_frame.py` — reconstructs the raw-mocap + alignment-transform path
  for the world-frame trajectory and contrasts it with the canonicalized
  `contact_motion/motions` tensor.
- `check4_mesh_check.py` — confirms ScanNet mesh coverage and loadability for all
  HUMANISE scene ids.

---

## Check 1 — World-frame root trajectory: PARTIAL / NOT IN PREPROCESSED TENSORS, RECOVERABLE FROM RAW DATA

**The tensors actually used for training (`HUMANISE/contact_motion/motions/*.npy`,
shape `(T, 22, 3)`) are canonicalized, not world-frame.** Verified directly: for
`00003.npy` (a "lie down" transition clip) the root joint (index 0) only spans
`[0.36, 0.39, 0.84]` meters across the whole clip, and for near-static clips
(`00000.npy`) it spans centimeters. This is a per-clip-centered local joint-position
representation, structurally analogous to HumanML3D's canonicalized root — it carries
**no absolute scene position and very likely no yaw** (values sit near a fixed local
origin regardless of clip). So: if the question is "does the *preprocessed* tree
already contain a ready-to-use `(x, y, yaw)` tensor," the answer is **no**.

However, this is *not* the "only canonicalized 263-dim exists, world frame discarded"
failure case the spec warns about. The raw ingredients to reconstruct a genuine
world-frame trajectory exist, split across two untouched raw zip archives:

1. **`HUMANISE/pure_motion/<action>.zip` → `<clip>/motion.pkl`** (a 9-tuple):
   - `[1] transl`: `(T, 3)` float32, raw per-frame SMPL-X root translation,
     **Y-up**, meters. Verified smoothly varying across frames for a walk clip
     (X: 3.54→1.53 m, Y: ~0.73–0.88 m constant height, Z: 0.83–0.92 m small
     lateral drift) — this is genuine frame-by-frame motion, not a static/centered
     value.
   - `[2] global_orient`: `(T, 3)` float32, raw per-frame root axis-angle
     orientation (yaw is recoverable from this).
   - Also present: `betas` (16,), `body_pose` (T,63), and other fields not needed
     for the trajectory.

2. **`HUMANISE/align_data_release/<action>.zip` → `<clip>/anno.pkl`** (list with
   one dict) — the rigid transform that places a raw mocap clip into its ScanNet
   scene's coordinate frame:
   - `scene_translation`: `(3,)` — matches `scene_trans_x/y/z` in
     `HUMANISE/annotations.csv`.
   - `translation`: `(3,)` — per-instance placement offset.
   - `rotation`: **scalar float, radians** (e.g. `0.349` rad ≈ 20°) — a single
     yaw applied about the up-axis to place the whole clip in the scene.
   - Plus `scene`, `action`, `utterance`, `object_id` metadata for matching.

   Spot-checked end to end for `scene0000_00`, action "walk to the couch":
   `rotation=0.349`, `scene_translation=[-4.111, -4.178, -0.061]`,
   `translation=[-2.105, 1.258, -0.025]`.

Reconstructing world-frame `(x, y, yaw)` per frame is therefore a **rigid transform**
(`world_xyz[t] = R(rotation) @ transl[t] + translation + scene_translation`,
`world_yaw[t]` from `global_orient[t]` composed with `rotation`), not integration of
local velocities — structurally safer than trying to recover it from the 263-dim
canonicalized representation, which does drift.

**Coordinate convention (inferred, not documented in the data tree):**
- Units: meters (both `transl` values and ScanNet mesh bounds are in the same
  ~0–10 m room-scale range).
- Raw SMPL-X `transl`/`global_orient` (`pure_motion` pkls) are **Y-up**.
- ScanNet `.ply` meshes (`scannet/scans/*_vh_clean_2.ply`) are **Z-up**, floor near
  z≈0 (verified: `scene0000_00` bounds z∈[-0.0006, 3.02]).
- These two conventions do not match as-is — some axis remap must happen inside
  HUMANISE's own alignment step (`align_data_release`) or in a downstream loader
  not present in this data tree. **This must be pinned down explicitly (e.g. by
  reconstructing one full clip's world trajectory and overlaying it on the scene
  mesh bounds) before Step 3 pipeline code is written.** Not attempted here — out
  of scope for a plumbing check, and doing it wrong silently would be exactly the
  kind of workaround the spec says not to do.
- `yaw` zero-direction: not determinable without doing the axis-remap /
  reconstruction above.

**Verdict for Check 1: does not fail outright, but does not pass cleanly either.**
No ready-made world-frame tensor exists in the preprocessed tree. A faithful,
non-drifting reconstruction path exists in raw form but requires:
(a) joining `pure_motion` clip ids to `align_data_release` clip ids to
`contact_motion/motions` row indices (only one clip was spot-checked end-to-end,
general ID-matching was not verified across the full 19,648-clip set), and
(b) resolving the Y-up/Z-up axis mismatch between mocap and scene mesh. Both are
concrete, boundable preprocessing tasks — not "impossible" — but they are real,
unbuilt work, not a plumbing check pass. **This should be treated as a design
decision to make explicitly at the start of Step 3, not silently assumed away.**

---

## Check 2 — HUMANISE → HumanML3D 263-dim conversion: **FAIL (does not exist)**

- `H3D/new_joint_vecs/*.npy` **is** the standard HumanML3D 263-dim representation:
  shape `(T, 263)` confirmed (`000000.npy` → `(116, 263)`), with matching
  `H3D/Mean.npy` / `H3D/Std.npy` both shape `(263,)`. The layout math checks out
  for 22 joints: `1 (root ang vel) + 2 (root lin vel xz) + 1 (root height) + 21*3
  (local joint pos) + 21*6 (local joint 6D rot) + 22*3 (joint vel) + 4 (foot
  contact) = 1+2+1+63+126+66+4 = 263`. ✓.
- **But `H3D` is the original HumanML3D dataset, not a HUMANISE conversion.**
  Evidence: `H3D/new_joint_vecs` file count is 29,228 (≈ HumanML3D's 14.6K base +
  mirrored augmentation), and `H3D/texts/*.txt` contain generic, scene-free action
  descriptions ("a man kicks something or someone with his left leg", "he is flying
  kick with his left leg") — not HUMANISE's scene-grounded instructions.
- HUMANISE's own motion representation, `HUMANISE/contact_motion/motions/*.npy`,
  is `(T, 22, 3)` — canonicalized **joint positions only** (66-dim per frame if
  flattened), matching `Mean_Std_CM_HUMANISE_pos.npz` (`mean`/`std` shape `(1,
  66)`). File count is 19,648, exactly matching `HUMANISE/all.txt` line count — so
  this is HUMANISE's real per-clip representation, and it is **not** 263-dim, has
  no rotations/velocities/contacts baked in, and is not the T2M-GPT VQ-VAE's
  expected input format.
- A repo-wide search for any other `263`-dim array or a `new_joint_vecs`-style
  directory under HUMANISE turned up nothing.
- No conversion script, README, or metadata file describing a HUMANISE→263-dim
  step was found anywhere in `/media/user/2tb/motion_data/`.

**Verdict for Check 2: FAIL.** HUMANISE motion has **not** been converted to the
263-dim HumanML3D format the frozen T2M-GPT VQ-VAE expects. Only the original
HumanML3D dataset (`H3D/`) is in that format. HUMANISE exists in a different,
bespoke `(T, 22, 3)` joint-position + contact-point format (paired with
`contact_motion/contacts/*.npz` — 8192-point scene point clouds with
per-point distance/mask fields — clearly built for a different, contact-prediction
style pipeline, not for a frozen HumanML3D VQ-VAE). **Writing a HUMANISE → 263-dim
converter (joint positions → root-relative decomposition → rotations/velocities/
contacts, per the official HumanML3D `motion_process.py` recipe) is unbuilt
work and is on the critical path before Step 2/3 can touch HUMANISE at all.**

---

## Check 3 — Reconstruction canary: **BLOCKED (no checkpoint located)**

- Model code **is** present: `/home/user/Khiem-ssh/T2M-GPT` (a full clone of the
  T2M-GPT repo, including `models/vqvae.py`), but it contains **no checkpoint
  files** (`.pth`/`.tar`) anywhere in the repo.
- Broad search of `/media/user/2tb/motion_data/`, the home directory, and common
  checkpoint locations found no T2M-GPT / VQ-VAE checkpoint. The only relevant
  `.pth` under the data root is `POINTTRANS_C_N8192_E300/model.pth`, which is a
  Point Transformer checkpoint (8192-point scene cloud encoder, consistent with
  the `contact_motion/contacts` 8192-point fields) — unrelated to the T2M-GPT
  VQ-VAE.
- `torch` is not installed in the current Python environment, so even if a
  checkpoint were found, inference could not run without an environment setup
  step first.

**Verdict for Check 3: cannot be executed. Blocked on: (1) locating/downloading a
T2M-GPT VQ-VAE checkpoint, (2) installing torch.** This is expected/acceptable per
the spec — surfaced, not silently skipped.

---

## Check 4 — BEV occupancy raster achievability: **PASS (meshes present); renderer not built**

- `HUMANISE/scenes/` lists 643 scene ids; `scannet/scans/` has exactly 643 scan
  directories, and **all 643 HUMANISE scene ids have a matching ScanNet scan
  directory** (`comm` diff = 0 missing).
- Each scan directory contains one cleaned mesh, e.g.
  `scannet/scans/scene0000_00/scene0000_00_vh_clean_2.ply` (3.3 MB, standard
  ScanNet `vh_clean_2` format). Spot-checked 5 scenes: all exist, nonzero size,
  and load successfully via `trimesh` (`scene0000_00`: 81,369 vertices, bounds
  ≈8.4 × 8.7 × 3.0 m; other scenes similar room-scale bounds). Format is
  standard triangle mesh PLY, Z-up, floor near z≈0.
- **No BEV rendering script, occupancy-raster script, or renderer of any kind was
  found** anywhere in `/media/user/2tb/motion_data/` or the current repo
  (`/home/user/Khiem-ssh/wander`). It needs to be built from scratch — e.g. an
  orthographic top-down render via `trimesh`/`pyrender`/`open3d`, rasterizing RGB
  and a walkable/occupied binary mask from the same camera and extent. Not
  attempted here per spec (Check 4 explicitly excludes actual BEV rendering from
  this step).
- World→pixel mapping is not yet defined (no renderer exists to define it against).
  Given the mesh bounds are already in ScanNet's native Z-up world-frame meters,
  the natural design is a per-scene orthographic projection with a fixed
  meters-per-pixel scale and the scene's own `bounds[:, :2]` (i.e. XY) as the
  raster extent — but this is a Step 4/6 design decision, not verified/built here.

**Verdict for Check 4: meshes present and loadable — recoverable/buildable, but
the BEV renderer itself does not exist yet anywhere in the tree.** Per spec this
is a surfaced-but-recoverable finding, not a hard blocker.

---

## Overall Verdict

**NOT ALL PASS.** Summary:

| Check | Status |
|---|---|
| 1. World-frame root trajectory | **PARTIAL** — absent from preprocessed tensors; recoverable from raw `pure_motion` + `align_data_release` pkls via a rigid transform, but reconstruction is unbuilt and has an unresolved Y-up/Z-up axis mismatch against the ScanNet meshes. Treat as a design decision to make explicitly, not a silent given. |
| 2. HUMANISE → 263-dim conversion | **FAIL** — does not exist. Only the original HumanML3D (`H3D/`) is in 263-dim format. HUMANISE is stored as `(T,22,3)` canonicalized joint positions + separate contact-point data, built for a different pipeline. A converter must be written. |
| 3. VQ-VAE reconstruction canary | **BLOCKED** — no checkpoint located anywhere on disk; torch not installed in current env. |
| 4. BEV occupancy raster | **RECOVERABLE** — ScanNet meshes present for all 643 HUMANISE scenes and load correctly; no renderer exists yet, must be built. |

**Per the spec: "Do not begin Step 2 until Check 1 passes."** Check 1 did not pass
cleanly — it surfaced a concrete but unbuilt reconstruction path with an open axis-
convention question. Check 2 is an outright fail requiring new conversion code
before HUMANISE can feed the frozen VQ-VAE at all. **Recommend stopping here for a
design decision covering both**, specifically:
1. Whether to invest in the raw-pkl world-frame reconstruction (rigid transform +
   axis-remap) as the source of truth for `(x, y, yaw)`, and how to validate it
   (e.g. overlay a reconstructed trajectory on a scene mesh render and visually
   confirm it stays on the floor/walkable area).
2. Who/what writes the HUMANISE → HumanML3D 263-dim converter (adapting the
   official `motion_process.py` recipe to HUMANISE's `(T,22,3)` joint positions),
   since this is required before Check 3 can even be meaningful for HUMANISE data
   (today only `H3D`/HumanML3D can feed the frozen VQ-VAE, not HUMANISE).

Checks 3 and 4 are individually recoverable (get a checkpoint / install torch;
write a renderer) and are not architecture-changing, consistent with the spec's
expectation that they may be "recoverable but must be surfaced."

---
---

# STEP 1b — Extend Plumbing: Report

Implements STEP1b_extend.md Tasks 0-3. Code lives under `src/` (preprocessor
modules) and `scripts/verify/` (standalone check scripts, each runnable
directly). Data/checkpoints/renders are gitignored; only code and this report
are committed.

## Task 0 — Unblock Check 3 (fetch)

- **torch**: installed `torch==2.7.1+cu118` via
  `pip install torch --index-url https://download.pytorch.org/whl/cu118`.
  `torch.cuda.is_available()` → `True` on the RTX 4090 (CUDA 11.8 toolchain
  present via `nvcc`).
- **Checkpoint**: downloaded via the T2M-GPT repo's own official script,
  `T2M-GPT/dataset/prepare/download_model.sh` (`gdown 1LaOvwypF-jM2Axnq5dc-Iuvv3w_G-WDE`
  → `VQTrans_pretrained.zip`, unzipped). Repo commit at time of use: `b1446f1`
  ("Update t2m extractor", `Mael-zys/T2M-GPT`).
  - **Path used**: `/home/user/Khiem-ssh/T2M-GPT/pretrained/VQVAE/net_best_fid.pth`
  - Loads with `strict=True` into `models.vqvae.HumanVQVAE` (19,436,807 params).
    `run.log` shipped alongside the checkpoint records `dataname=t2m,
    quantizer=bailando, nb_code=512, code_dim=512, down_t=2` (training-time
    config); the current repo's quantizer classes don't include `"bailando"`,
    but the checkpoint's only quantizer buffer is `vqvae.quantizer.codebook`,
    which matches the `QuantizeEMAReset` ("ema_reset") class's buffer name --
    used that at inference time (weights-only concern, no retraining).
  - No model code changed.

## Task 1 — Unified HUMANISE preprocessor

Module: `src/humanise_join.py` (Track 2 / world-frame + ID-join) and
`src/motion_features.py` (Track 1 / 263-dim adapter over the unmodified
HumanML3D extractor).

### Track 2 — world-frame root (built and validated first)

**ID-join at full scale** (`scripts/verify/check6_id_join_full.py`):
HUMANISE's join key is not value-matching (the top-level `HUMANISE/annotations.csv`
is a header-only stub, 0 data rows -- unusable) but **positional**, reproduced
from the reference implementation
`/home/user/jered/T2M_test/afford-motion/prepare/datasets/HUMANISE/HUMANISE.py`:
flatten `align_data_release/<action>/<clip>/anno.pkl` in `natsorted` **action**
order (`lie, sit, stand up, walk`), `natsorted` **clip** order within each
action, and each anno.pkl's placement-dict **list** in list order (list
lengths vary 1-9, not always 1). The 0-based position in that flattened stream
*is* the motion_id used everywhere else (`contact_motion/motions/{id:05d}.npy`,
`all.txt`, `contact_motion/anno.csv` row order).

Result, run over **all** clips, not a sample:
- `align_data_release` flattened entries: **19,648**
- `contact_motion/anno.csv` rows: **19,648**
- Field mismatches (scene_id + utterance) across all 19,648 joined rows: **0**
- `pure_motion` clip references that fail to resolve: **0**
- **Coverage: 19,648 / 19,648 = 100.0000%, 0 misses.**

**World-frame reconstruction** (`humanise_join.compute_track2`): rigid
transform per HUMANISE's own placement code
(`_transform_smplx_from_origin_to_sampled_position`), applied directly to raw
`pure_motion` joint positions (not full SMPL-X reposing, since a rigid
transform commutes with skinning for our purposes):
1. Recenter horizontal (XY) using the clip's `anchor_frame` pelvis position
   (anchor frame is action-dependent: `stand up` anchors frame 0, all others
   anchor the last frame -- taken verbatim from HUMANISE.py).
2. Rotate about world Z by `anno.pkl['rotation']`.
3. Translate by `anno.pkl['translation']`.
4. **Empirically-found correction**: subtract the per-scene constant
   `anno.pkl['scene_translation']`. HUMANISE.py's own placement transform
   does *not* apply `scene_translation` when producing its SMPL-X params (it
   only stores it as metadata), but the ACTUAL ScanNet mesh files on this
   machine are offset from that placement frame by exactly this vector.
   Found via the Task 3 floor-overlay test itself (see below): without this
   correction, Z (height) already matched the floor almost exactly, while
   XY trajectories landed meters outside the room footprint -- exactly the
   signature of a missing constant horizontal offset, since
   `scene_translation`'s Z component is ~centimeters but its X/Y components
   are several meters. Confirmed the fix against 165 clips across 6 scenes
   (see Task 3 numbers).

**Axis-convention finding (revises the a-priori assumption in CLAUDE.md /
STEP1b_extend.md's "LOCKED DECISION")**: empirical inspection of HUMANISE's
own joint tensors -- both `contact_motion/motions/*.npy` and the raw
`pure_motion` `joints_traj` -- shows column **index 2**, not index 1, is
anatomically "up" (e.g. a walking clip: head z≈1.53, feet z≈0.05-0.09,
essentially constant across the whole clip, while columns 0 and 1 vary by
1-3m as the person walks). HUMANISE's own placement code is consistent with
this: it recenters only the XY (indices 0,1) components of the pelvis and
rotates about axis `[0,0,1]` (Z). **HUMANISE's raw SMPL-X data is natively
Z-up already** (X, Y horizontal; Z up) -- matching ScanNet's world frame
directly, not the Y-up SMPL/AMASS convention STEP1 guessed from a single
ambiguous walk clip. There is therefore **no Y-up→Z-up rotation to insert**
in Track 2 (the rigid transform above operates entirely within this native
Z-up frame). This is exactly the kind of thing the spec's floor-overlay test
is designed to catch if assumed wrong; see Task 3 for the empirical
confirmation. Yaw is derived from hip/shoulder joint geometry (a Z-up
adaptation of HumanML3D's own face-direction formula), not by decomposing
`global_orient`, avoiding a second independent up-axis assumption. Yaw is
stored as `(sin, cos)` per CLAUDE.md.

### Track 1 — canonicalized 263-dim (for the frozen VQ-VAE)

**Joint order: CONFIRMED MATCH, no reindexing needed.** HUMANISE's 22-joint
tensors come from `smplkit.SMPLXLayer` truncated to the first 22 joints
(`prepare/smplx_to_vec.py` in the afford-motion / HUMANISE data-prep repos),
which is `smplkit`'s `JOINTS_NAME['SMPLX'][:22]` -- index-for-index identical
to T2M-GPT/HumanML3D's SMPL joint order (`utils/paramUtil.py`'s
`t2m_kinematic_chain`): pelvis, l/r hip, spine1, l/r knee, spine2, l/r ankle,
spine3, l/r foot, neck, l/r collar, head, l/r shoulder, l/r elbow, l/r wrist.

**Extractor: reused unmodified, not hand-written.** The forward 263-dim
extractor (`process_file` / `extract_features`, `uniform_skeleton`) does
**not** exist in `T2M-GPT/utils/motion_process.py` (that file only ships the
*inverse* functions, `recover_from_ric` etc.) or anywhere under
`/media/user/2tb/motion_data`. It was located, still on this machine, at
`/home/user/jered/T2M_test/motion-diffusion-model/data_loaders/humanml/scripts/motion_process.py`
(and an identical copy in `OmniControl`), whose `paramUtil.py` is
byte-identical to T2M-GPT's own. `src/motion_features.py` imports this module
directly (`sys.path` insert, no code copied) and calls `process_file()`
unmodified. Two things were added, both non-mathematical:
1. A compat shim (`np.float = float`) for the vendored 2022-era script's use
   of a numpy alias removed in numpy≥1.24 -- not a math change.
2. Explicit global-variable plumbing (`tgt_offsets`, `face_joint_indx`,
   `fid_r`/`fid_l`, `l_idx1`/`l_idx2`) that the original script only sets when
   run as `__main__`; the target-skeleton offsets are recovered from H3D's own
   published `new_joint_vecs/000021.npy` (the canonical `t2m_tgt_skel_id`) via
   the inverse function, rather than needing a raw-joints file this machine
   doesn't have.

**Up-axis adapter** (the "thin adapter" the spec asked for): given Track 2's
axis finding (HUMANISE positions are Z-up), and `process_file`'s hard
assumption that column 1 is up (`root_y = positions[:,0,1]`,
`floor_height = positions[...,1].min()`), `motion_features.zup_to_yup_hml`
applies the proper rotation `(x,y,z)_zup -> (x, z, -y)_yup` (a -90° rotation
about X, determinant +1 -- preserves handedness, unlike a bare column swap).

**Run-at-scale validation** (`scripts/verify/check5_track1_full_scale.py`,
all 19,648 clips): **19,648 / 19,648 (100.00%) produced finite 263-dim
output, 0 exceptions, 0 NaN/Inf, 0 too-short (<3 frames)**. Elapsed 209s.

**Keeping the two tracks separate / not conflating scene-contact data**: Track
1's foot-contact bits come only from `process_file`'s internal velocity
threshold on the extracted local joints (standard HumanML3D behavior,
untouched). HUMANISE's separate `contact_motion/contacts/*.npz` (8192-point
scene point clouds) is never read by `motion_features.py` -- kept fully
out of the 263-dim pipeline, reserved for the later scene-contact metric per
CLAUDE.md.

## Task 2 — Reconstruction canary (`scripts/verify/check7_vqvae_canary.py`)

**Methodology note (a real bug found and fixed during this task):** an
initial attempt measured MPJPE on `recover_from_ric`'s globally-reconstructed
positions and got nonsense numbers (H3D baseline MPJPE ~7.9 **meters**).
Diagnosis: `recover_from_ric` reintroduces world heading/position via a
`torch.cumsum` over the *whole clip* -- a tiny per-frame rotation-velocity
error compounds into meters of drift by frame 100+, which swamps any signal
about local pose/reconstruction fidelity and is not what "the frozen VQ-VAE
reconstructs motion acceptably" is asking about. Fixed by evaluating MPJPE on
**local, root-relative, heading-canonicalized per-frame joint positions**
(`motion_features.local_joint_positions`, read directly from the 263 vector's
root-height + `ric_data` columns -- no cross-frame integration at all, pure
per-frame slicing). `recover_from_ric` is still used for the qualitative
rendered samples, where visually-inspectable drift is fine.

**Stage 1 — H3D baseline** (n=200 random clips):
- MPJPE mean **137.3 mm**, median 111.9 mm, max 503.2 mm.
- Sanity: this is the frozen checkpoint + eval harness working end-to-end on
  its own native training distribution -- confirms Task 0's setup is correct.

**Stage 2 — converted HUMANISE** (n=200 random clips):
- MPJPE mean **256.5 mm** (1.87x H3D), median 179.6 mm, max 1004.7 mm.
- **Breakdown by action** (n=400, separate larger sample):

  | action    | n   | MPJPE mean | vs H3D baseline |
  |-----------|-----|-----------:|-----------------:|
  | walk      | 189 | 112.3 mm   | 0.82x (in line)   |
  | stand up  | 60  | 191.9 mm   | 1.40x             |
  | sit       | 104 | 292.7 mm   | 2.13x             |
  | **lie**   | 47  | **702.9 mm** | **5.12x**       |

- **Visual confirmation**: a rendered "lie on the bed" sample
  (`scratch_outputs/canary/humanise_lie_{orig,recon}.png`, gitignored --
  paths reported here since the files aren't committed) shows the original
  motion is a correctly flat, lying pose (vertical extent ~0-0.25 m). The
  VQ-VAE reconstruction is **structurally broken**: the pose collapses into a
  contorted, non-flat shape, and the (global, for-visualization-only)
  position drifts to absurd coordinates (axis ranges reaching -10 m) within
  ~50 frames.

**Verdict for Task 2: walk and stand-up reconstruct acceptably (roughly in
line with, or moderately worse than, the H3D baseline). Sit is degraded
(~2x). Lie is poor -- 5x the baseline error and visually broken.** This is
exactly the spec's flagged watch item: "sitting/complex scene-contact
motions may fall outside HumanML3D's codebook distribution." Lie and sit
together are ~40% of HUMANISE's clips (lie 2,343 / sit 5,578 of 19,648 =
~40.3%), so this is not a negligible edge case.

## Task 3 — Validate the world-frame track (`scripts/verify/check8_track2_overlay.py`)

Overlaid `compute_track2` root trajectories on their scene's ScanNet mesh
(top-down XY footprint + a floor/ceiling-referenced height plot), for 5
scenes x 3 clips = 15 clips, **all 15 PASS** (trajectory XY stays within the
mesh's XY bounds + 1m margin, and pelvis Z stays within
`[floor - 0.15m, floor + 2.2m]` for the whole clip). A broader random sample
of 150 clips across many scenes/actions (no rendering, numeric bounds check
only) is **150/150 PASS**.

This *is* the axis-fix correctness test: it's what surfaced the missing
`scene_translation` offset (see Track 2 above) -- before that fix, height
already matched the floor (confirming Z-up was right) but XY trajectories
landed meters outside every room's footprint (confirming a horizontal offset
bug, not an axis-convention bug). One example
(`scratch_outputs/overlay/scene0006_00_overlay.png`, gitignored) shows a
"lie down" clip's pelvis height dropping smoothly from ~1.7 m (standing) to
~0.85 m (lying on the bed) while its XY trajectory stays inside the room --
both the up-axis and the horizontal placement check out together.

## Overall STEP 1b Verdict

| Requirement (per STEP1b_extend.md) | Status |
|---|---|
| World-frame track built, ID-join at scale (log coverage) | **PASS** -- 19,648/19,648, 0 misses |
| Floor overlay passes | **PASS** -- 15/15 rendered + 150/150 broader sample |
| 263 converter reuses HumanML3D extraction, layout confirmed | **PASS** -- joint order match confirmed, unmodified extractor reused, 19,648/19,648 run cleanly |
| Reconstruction canary: H3D passes | **PASS** -- 137.3 mm mean local MPJPE, sane end-to-end |
| Reconstruction canary: HUMANISE reconstructs acceptably | **FAIL for `lie` (5.1x H3D error, visually broken) and degraded for `sit` (2.1x)**; `walk`/`stand up` are acceptable |

**VERDICT: STOP for a design decision.** Three of the four Task-3/Task-1
requirements are cleanly green (world-frame track, ID-join, floor overlay,
263-conversion reuse+layout are all solid, no workarounds). But per the
spec's explicit watch item, the frozen VQ-VAE does **not** reconstruct
HUMANISE's `lie` motions acceptably (5x the baseline error, structurally
collapsed pose in the rendered sample) and reconstructs `sit` only
moderately well (2x baseline). Per STEP1b_extend.md: *"Do NOT preempt this.
Report the numbers and STOP for a design decision if HUMANISE reconstruction
is poor. Do not silently start finetuning."* Accordingly, **Step 2 (T2M-GPT
baseline repro) is safe to start independently** (it only touches H3D, which
passes cleanly), but **HUMANISE-dependent work (Step 3 scene conditioning)
should not proceed until a decision is made** on one of:
1. Accept degraded quality for `sit`/`lie` segments (they're still usable,
   just noisier -- e.g. down-weight their loss contribution or exclude them
   from the earliest scene-conditioning milestone and add them back later).
2. Finetune the VQ-VAE decoder on HUMANISE's contact-heavy motions --
   **this changes CLAUDE.md's "VQ-VAE frozen" locked decision** and needs
   explicit sign-off, not a silent code change.
3. Investigate whether a different/larger T2M-GPT VQ-VAE variant (if one
   exists) or a preprocessing fix (e.g. the codebook may be undertrained on
   near-static/floor-contact motions specifically, not sit/lie geometry per
   se -- worth a quick check on H3D's own near-static clips before
   concluding it's a `sit`/`lie`-specific gap) closes the gap without
   touching the frozen weights.
