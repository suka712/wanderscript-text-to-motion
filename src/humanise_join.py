"""
Unified HUMANISE preprocessor: ID-join + world-frame (Track 2) reconstruction.

Join logic (validated at full scale in scripts/verify/check6_id_join_full.py,
19648/19648, 0 misses): iterate align_data_release/<action>.zip in natsorted
action order (lie, sit, stand up, walk), natsorted clip order within each
action, flatten each anno.pkl's list of placement dicts in list order. The
resulting 0-based index equals the motion_id used everywhere else
(HUMANISE/contact_motion/motions/{idx:05d}.npy, HUMANISE/all.txt,
HUMANISE/contact_motion/anno.csv row order).

Track 2 (world-frame root trajectory) reproduces the placement transform from
the reference implementation in
/home/user/jered/T2M_test/afford-motion/prepare/datasets/HUMANISE/HUMANISE.py
(_transform_smplx_from_origin_to_sampled_position), simplified to operate
directly on joint positions (we don't need full SMPL-X reposing for a root
trajectory, just a rigid transform of joint positions -- rotation and
translation commute with skinning for a *rigid* per-clip transform).

Axis-convention finding (see src/motion_features.py docstring for the full
argument): HUMANISE's raw joint data (both pure_motion's joints_traj and the
derived contact_motion/motions) is natively Z-up (X, Y horizontal; Z is
"up" -- verified by anatomical joint-height ordering, e.g. head z~1.5, feet
z~0.05). HUMANISE's own alignment code rotates about axis=[0,0,1] (Z) and
recenters only the XY (indices 0,1) components of the pelvis -- consistent
with Z already being up in the pre-placement frame. So Track 2's rigid
transform is applied entirely within this native Z-up frame; there is NO
separate Y-up-to-Z-up rotation to insert (that assumption in
STEP1b_extend.md's "LOCKED DECISION" section does not hold for HUMANISE's raw
data -- see STEP1b report for the full evidence trail). This is exactly the
kind of thing Task 3's floor-overlay check is designed to catch if wrong.

Yaw is derived from joint geometry (hip/shoulder cross product, mirroring
HumanML3D's own face-direction formula, adapted to a Z-up world) rather than
by decomposing the SMPL-X axis-angle global_orient -- this avoids a second,
independent up-axis assumption about a value we don't directly need.
"""
import io
import pickle
import zipfile
from dataclasses import dataclass
from typing import Optional

import numpy as np
from natsort import natsorted

ROOT = "/media/user/2tb/motion_data/HUMANISE"
ACTIONS = natsorted(["walk", "sit", "stand up", "lie"])
ANCHOR_FRAME = {"sit": -1, "stand up": 0, "walk": -1, "lie": -1}

# joint indices within the 22-joint SMPL order (see motion_features.py)
J_PELVIS, J_LHIP, J_RHIP, J_LSHOULDER, J_RSHOULDER = 0, 1, 2, 16, 17


@dataclass
class ClipRecord:
    index: int
    scene: str
    action: str
    motion_id: str
    translation: np.ndarray  # (3,) align placement translation
    rotation: float  # radians, yaw about world Z
    scene_translation: np.ndarray  # (3,) per-scene offset, see compute_track2 note
    utterance: str
    object_id: int
    anchor_frame: int


_flat_cache: Optional[list] = None


def build_flat_join() -> list:
    """Returns the list of raw align_data placement dicts in motion_id order
    (index i in this list == motion_id i == contact_motion/motions/{i:05d}.npy).
    """
    global _flat_cache
    if _flat_cache is not None:
        return _flat_cache
    flat = []
    for a in ACTIONS:
        with zipfile.ZipFile(f"{ROOT}/align_data_release/{a}.zip") as zf:
            members = natsorted(n for n in zf.namelist() if n.endswith("anno.pkl"))
            for m in members:
                with zf.open(m) as f:
                    d = pickle.load(io.BytesIO(f.read()))
                for p in d:
                    flat.append(p)
    _flat_cache = flat
    return flat


def get_record(index: int) -> ClipRecord:
    flat = build_flat_join()
    p = flat[index]
    return ClipRecord(
        index=index,
        scene=p["scene"],
        action=p["action"],
        motion_id=p["motion"],
        translation=np.asarray(p["translation"], dtype=np.float64),
        rotation=float(p["rotation"]),
        scene_translation=np.asarray(p["scene_translation"], dtype=np.float64),
        utterance=p["utterance"],
        object_id=int(p["object_id"]),
        anchor_frame=ANCHOR_FRAME[p["action"]],
    )


_pure_motion_cache = {}


def load_pure_motion(action: str, motion_id: str):
    """Loads the raw (pre-placement) SMPL-X motion.pkl 9-tuple for a clip.
    Returns joints_traj (T, 127, 3) among other fields; cached by
    (action, motion_id) since multiple placements can reuse one raw clip.
    """
    key = (action, motion_id)
    if key in _pure_motion_cache:
        return _pure_motion_cache[key]
    with zipfile.ZipFile(f"{ROOT}/pure_motion/{action}.zip") as zf:
        with zf.open(f"{action}/{motion_id}/motion.pkl") as f:
            d = pickle.load(io.BytesIO(f.read()))
    # keep the cache small-ish: only store what we need
    gender, transl, global_orient, betas, body_pose, hand_pose, jaw_pose, eye_pose, joints_traj = d
    entry = {
        "transl": transl,
        "global_orient": global_orient,
        "joints_traj": joints_traj[:, :22, :].astype(np.float32),  # first 22 = SMPL joints
    }
    _pure_motion_cache[key] = entry
    return entry


def rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def compute_track2(record: ClipRecord):
    """Rigid placement transform applied to all 22 raw joints (Z-up native
    frame throughout). Returns:
      joints_world: (T, 22, 3) placed joint positions, Z-up, ScanNet world frame
      xy: (T, 2) root (pelvis) world position
      yaw: (T,) radians, world yaw (0 = facing +X... see sincos below for the
           canonical (sin, cos) form used everywhere downstream)
      sincos: (T, 2) [sin(yaw), cos(yaw)] -- the CLAUDE.md-mandated yaw encoding
    """
    pm = load_pure_motion(record.action, record.motion_id)
    joints = pm["joints_traj"].astype(np.float64)  # (T, 22, 3), Z-up native
    anchor = record.anchor_frame
    pelvis_anchor_xy = joints[anchor, J_PELVIS, :2].copy()

    # T1: recenter horizontal (XY) using the anchor frame's pelvis XY
    recentered = joints.copy()
    recentered[..., 0] -= pelvis_anchor_xy[0]
    recentered[..., 1] -= pelvis_anchor_xy[1]

    # T2: rotate about world Z by record.rotation
    R = rot_z(record.rotation)
    rotated = recentered @ R.T  # (T, 22, 3)

    # T3: translate by the align-data placement translation
    joints_world = rotated + record.translation[None, None, :]

    # Empirical correction (Task 3 finding): HUMANISE.py's placement transform
    # (as vendored in afford-motion's HUMANISE.py) places the human into a
    # scene-centering frame that is offset from the ACTUAL ScanNet mesh's own
    # coordinate frame by exactly `scene_translation` (the same constant
    # per-scene value recorded in align_data's anno.pkl / annotations.csv).
    # Confirmed by the floor-overlay test in scripts/verify/check8: without
    # this correction, XY trajectories land meters outside the scene mesh's
    # footprint while Z (height) already lines up with the floor (since
    # scene_translation's Z component is tiny, ~a few cm, vs several meters
    # in X/Y) -- i.e. only the horizontal placement was off, exactly what a
    # missing scene-frame offset would produce. Subtracting scene_translation
    # brings trajectories onto the mesh footprint (see check8 pass rate).
    joints_world = joints_world - record.scene_translation[None, None, :]

    xy = joints_world[:, J_PELVIS, :2]

    # yaw from hip/shoulder geometry (mirrors HumanML3D's own face-direction
    # formula: across = (r_joint - l_joint)_hip + (r_joint - l_joint)_shoulder,
    # forward = up x across), adapted to a Z-up world (up = +Z, not +Y).
    across = (joints_world[:, J_RHIP, :2] - joints_world[:, J_LHIP, :2]) + \
             (joints_world[:, J_RSHOULDER, :2] - joints_world[:, J_LSHOULDER, :2])
    norm = np.linalg.norm(across, axis=-1, keepdims=True)
    norm[norm < 1e-8] = 1e-8
    across = across / norm
    # forward = Z x across (2D: rotate across by -90deg in-plane), matching
    # the right-handed cross(up, across) construction used by process_file
    forward = np.stack([-across[:, 1], across[:, 0]], axis=-1)
    yaw = np.arctan2(forward[:, 1], forward[:, 0])
    sincos = np.stack([np.sin(yaw), np.cos(yaw)], axis=-1)

    return joints_world.astype(np.float32), xy.astype(np.float32), yaw.astype(np.float32), sincos.astype(np.float32)
