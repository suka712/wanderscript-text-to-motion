#!/usr/bin/env python3
"""
Check 1 (deep dive): does a world-frame per-frame (x, y, yaw) trajectory exist
anywhere in the HUMANISE data tree?

Finding: NOT in the *preprocessed* training tensors (HUMANISE/contact_motion/motions
is (T, 22, 3) canonicalized local joint positions -- root stays near origin, see
span/displacement check below). It DOES exist in recoverable raw form, split across
two separate zip archives that must be joined by motion id:

  1. HUMANISE/pure_motion/<action>.zip -> <clip_id>/motion.pkl
     A 9-tuple of raw (pre-canonicalization) SMPL-X params for the mocap clip:
       [0] gender (str)
       [1] transl        (T, 3)  float32  -- raw per-frame root translation (Y-up, meters)
       [2] global_orient  (T, 3)  float32  -- raw per-frame root axis-angle orientation
       [3] betas          (16,)
       [4] body_pose      (T, 63)
       [5] ??? (T, 90)     -- likely hand/face pose, not needed here
       [6] (T, 3) all-zero in the sample checked
       [7] (T, 6) all-zero in the sample checked
       [8] (T, 127, 3)     -- likely full SMPL-X joint/vertex set

  2. HUMANISE/align_data_release/<action>.zip -> <clip_id>/anno.pkl
     A 1-element list containing a dict with the rigid transform that places the
     raw mocap clip into the ScanNet scene coordinate frame:
       scene_translation (3,)  -- scene-level offset (matches annotations.csv scene_trans_x/y/z)
       translation        (3,)  -- per-instance placement offset
       rotation            float -- single scalar yaw (radians) applied about the up-axis
       + action/scene/object metadata

Reconstructing a world-frame (x, y, yaw) trajectory is therefore:
  world_xyz[t] = R(rotation) @ transl[t] + translation + scene_translation
  world_yaw[t] = yaw(R(rotation) @ R(global_orient[t]))  (up-axis component only)

This is a RIGID transform (rotate + translate), not integration of local velocities,
so it does not drift -- structurally different (and safer) than trying to recover an
absolute frame from the canonicalized HumanML3D 263-dim representation.

Caveats to resolve before writing real pipeline code (not done here, per Step-1 scope):
  - Axis convention mismatch: raw SMPL-X transl/orient above is Y-up; the ScanNet
    .ply meshes (see check4) are Z-up with floor near z=0. Some axis remap happens
    somewhere in HUMANISE's own alignment pipeline -- must be pinned down exactly
    (inspect a full clip's world-frame output against the scene mesh bounds) before
    trusting sign/axis conventions.
  - `rotation` in anno.pkl is a single scalar per clip (rigid placement of the whole
    clip into the scene), not a per-frame yaw offset -- per-frame world yaw still
    needs global_orient decomposed frame-by-frame and composed with this scalar.
  - Matching motion ids between pure_motion/*.zip (six-digit_uuid folder names) and
    align_data_release/*.zip (scene0000_XXXXXX_uuid folder names) and the
    contact_motion/motions/00000.npy row-index convention needs an explicit join key
    -- not verified here beyond spot-checking one clip end-to-end.

Run this script to reproduce the spot check.
"""
import pickle
import zipfile
import numpy as np
import io

ROOT = "/media/user/2tb/motion_data/HUMANISE"

def load_pkl_from_zip(zip_path, member_name):
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member_name) as f:
            return pickle.load(io.BytesIO(f.read()))

if __name__ == "__main__":
    # raw mocap clip
    clip = "walk/005788_5866b366-0386-42d5-aa1a-846f47ac2372/motion.pkl"
    d = load_pkl_from_zip(f"{ROOT}/pure_motion/walk.zip", clip)
    gender, transl, global_orient = d[0], d[1], d[2]
    print("gender", gender)
    print("transl shape", transl.shape, "range", transl.min(0), transl.max(0))
    print("global_orient shape", global_orient.shape)

    # alignment transform for a (different, but representative) clip
    anno_member = "walk/scene0000_012402_ec1d6660-781a-42cc-ac52-0a990f601075/anno.pkl"
    anno = load_pkl_from_zip(f"{ROOT}/align_data_release/walk.zip", anno_member)[0]
    print("\nalign anno keys:", list(anno.keys()))
    print("scene_translation", anno["scene_translation"])
    print("translation", anno["translation"])
    print("rotation (rad)", anno["rotation"])
    print("scene", anno["scene"], "utterance", anno["utterance"])

    # confirm contact_motion/motions is canonicalized (root barely moves), NOT world-frame
    cm = np.load(f"{ROOT}/contact_motion/motions/00003.npy")
    root = cm[:, 0]
    print("\ncontact_motion/motions/00003.npy root span (should be small, sub-meter, "
          "if canonicalized):", (root.max(0) - root.min(0)))
