"""
Thin adapter around the ORIGINAL HumanML3D forward feature-extraction pipeline
(EricGuo5513's process_file / extract_features), which converts raw (T, 22, 3)
joint positions into the 263-dim HumanML3D representation the frozen T2M-GPT
VQ-VAE was trained on.

Per STEP1b_extend.md: "Do NOT hand-write the 263 feature math. Reuse HumanML3D's
own motion_representation extraction code." This module does NOT reimplement
that math -- it imports it unmodified from a vendored clone found on this
machine (motion-diffusion-model), and only adds:
  1. A thin joint-order / axis adapter for HUMANISE's (T, 22, 3) tensors.
  2. Global-state plumbing that process_file() expects (it reads several
     module-level globals -- tgt_offsets, face_joint_indx, fid_r, fid_l,
     l_idx1/l_idx2, n_raw_offsets, kinematic_chain -- that the original repo
     only sets when __main__ runs as a script; we set them once, explicitly).

Source of the forward extractor (verified present, byte-compatible with
T2M-GPT's own utils/paramUtil.py which only ships the *inverse* functions):
    /home/user/jered/T2M_test/motion-diffusion-model/data_loaders/humanml/scripts/motion_process.py
    /home/user/jered/T2M_test/motion-diffusion-model/data_loaders/humanml/utils/paramUtil.py
    /home/user/jered/T2M_test/motion-diffusion-model/data_loaders/humanml/common/skeleton.py

Joint order confirmation (see STEP1b report): HUMANISE's contact_motion/motions
22-joint order (from smplkit's SMPL-X joint truncation, used by
prepare/smplx_to_vec.py in the afford-motion / HUMANISE data-prep repos) is
index-for-index identical to HumanML3D/T2M-GPT's t2m_kinematic_chain SMPL joint
order (pelvis, left_hip, right_hip, spine1, left_knee, right_knee, spine2,
left_ankle, right_ankle, spine3, left_foot, right_foot, neck, left_collar,
right_collar, head, left_shoulder, right_shoulder, left_elbow, right_elbow,
left_wrist, right_wrist). NO joint reindexing is needed -- MATCH confirmed.

Axis convention finding (load-bearing, revises the a-priori Y-up assumption in
CLAUDE.md / STEP1b_extend.md): empirical inspection of HUMANISE's own joint
tensors (both contact_motion/motions/*.npy and the raw pure_motion
joints_traj) shows column index 2 (not index 1) is anatomically "up" --  e.g.
for a walking clip, head sits at z~1.53, feet at z~0.05, with columns 0 and 1
varying by 1-3m as the person walks (horizontal). HumanML3D's process_file
assumes the opposite: column index 1 is up (root_y = positions[:,0,1], floor
computed from positions[...,1]). So HUMANISE's raw joint data is natively
**Z-up already** (X, Y horizontal; Z vertical) -- consistent with ScanNet's
world frame, and requiring NO Y-up conversion at the raw-data level. The only
conversion needed is the axis *relabeling* below, to satisfy process_file's
Y-up input contract. This is a proper rotation (a -90 deg rotation about X),
not a naive column swap, so it preserves handedness:
    x_hml =  x_zup
    y_hml =  z_zup      (up)
    z_hml = -y_zup
"""
import os
import sys
import numpy as np
import torch

MDM_ROOT = "/home/user/jered/T2M_test/motion-diffusion-model"
if MDM_ROOT not in sys.path:
    sys.path.insert(0, MDM_ROOT)

# Compat shim only (NOT a math change): the vendored 2022-era script uses the
# deprecated `np.float` alias removed in numpy>=1.24. Restore the alias so the
# unmodified extraction code runs unchanged on this machine's numpy 1.26.
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

from data_loaders.humanml.scripts import motion_process as _mp  # noqa: E402
from data_loaders.humanml.utils import paramUtil as _paramUtil  # noqa: E402
from data_loaders.humanml.common.skeleton import Skeleton as _Skeleton  # noqa: E402

H3D_ROOT = "/media/user/2tb/motion_data/H3D"
NUM_JOINTS = 22

# HumanML3D t2m constants (from the official recipe's __main__ block in
# motion_process.py, reused verbatim -- not re-derived).
_L_IDX1, _L_IDX2 = 5, 8
_FID_R, _FID_L = [8, 11], [7, 10]
_FACE_JOINT_INDX = [2, 1, 17, 16]  # r_hip, l_hip, r_shoulder, l_shoulder

_initialized = False


def zup_to_yup_hml(positions_zup: np.ndarray) -> np.ndarray:
    """Convert (T, 22, 3) joint positions from HUMANISE's native Z-up frame
    (X, Y horizontal, Z up) to the Y-up frame HumanML3D's process_file expects
    (X, Z horizontal, Y up). Proper rotation (det = +1), not a bare swap.
    """
    out = np.empty_like(positions_zup)
    out[..., 0] = positions_zup[..., 0]
    out[..., 1] = positions_zup[..., 2]
    out[..., 2] = -positions_zup[..., 1]
    return out


def _compute_tgt_offsets() -> torch.Tensor:
    """Recover the canonical HumanML3D target skeleton (t2m_tgt_skel_id =
    '000021', per T2M-GPT's paramUtil.py) from its already-published 263-dim
    vector via the inverse function (recover_from_ric), then measure its bone
    offsets. This reproduces exactly what HumanML3D's own preprocessing script
    does when run from raw joints -- we don't have the raw joints file on this
    machine, but new_joint_vecs (263-dim) round-trips back to the same joint
    positions via the same skeleton math, so this is faithful, not approximated.
    """
    example_path = os.path.join(H3D_ROOT, "new_joint_vecs", "000021.npy")
    data = np.load(example_path)
    data_t = torch.from_numpy(data).unsqueeze(0).float()
    joints = _mp.recover_from_ric(data_t, NUM_JOINTS)[0]  # (T, 22, 3), Y-up
    n_raw_offsets = torch.from_numpy(_paramUtil.t2m_raw_offsets)
    kinematic_chain = _paramUtil.t2m_kinematic_chain
    skel = _Skeleton(n_raw_offsets, kinematic_chain, "cpu")
    tgt_offsets = skel.get_offsets_joints(joints[0])
    return tgt_offsets


def _ensure_globals():
    """process_file()/uniform_skeleton() in the vendored motion_process.py
    read several free variables as module globals (this is how the original
    EricGuo5513 script works when executed top-to-bottom). We set them once,
    explicitly, instead of relying on import-time side effects.
    """
    global _initialized
    if _initialized:
        return
    _mp.n_raw_offsets = torch.from_numpy(_paramUtil.t2m_raw_offsets)
    _mp.kinematic_chain = _paramUtil.t2m_kinematic_chain
    _mp.l_idx1, _mp.l_idx2 = _L_IDX1, _L_IDX2
    _mp.fid_r, _mp.fid_l = _FID_R, _FID_L
    _mp.face_joint_indx = _FACE_JOINT_INDX
    _mp.tgt_offsets = _compute_tgt_offsets()
    _initialized = True


def extract_263(positions_yup: np.ndarray, feet_thre: float = 0.002):
    """Run the UNMODIFIED HumanML3D process_file on (T, 22, 3) Y-up joint
    positions. Returns (data_263, global_positions, local_positions, l_velocity)
    exactly as process_file does.
    """
    _ensure_globals()
    positions_yup = positions_yup.copy().astype(np.float32)
    return _mp.process_file(positions_yup, feet_thre)


def humanise_positions_to_263(positions_zup_22: np.ndarray, feet_thre: float = 0.002):
    """Full adapter: HUMANISE (T, 22, 3) joint positions in HUMANISE's native
    Z-up frame -> HumanML3D 263-dim feature vector, via the unmodified
    process_file extractor. Joint order requires no reindexing (confirmed
    match, see module docstring); only the up-axis relabel is applied.
    """
    positions_yup = zup_to_yup_hml(positions_zup_22)
    return extract_263(positions_yup, feet_thre=feet_thre)


def local_joint_positions(data_263: np.ndarray, joints_num: int = NUM_JOINTS) -> np.ndarray:
    """Per-frame LOCAL (root-relative, heading-canonicalized) joint positions,
    read directly out of the 263-dim vector's root-height + ric_data columns
    -- no cumulative integration involved (unlike recover_from_ric, which
    reintroduces world heading/position via a cumulative sum over the whole
    clip and is therefore unsuitable for a per-frame reconstruction-fidelity
    metric on long clips: a tiny per-frame heading-velocity error compounds
    into meters of drift by frame 100, swamping any signal about local pose
    quality). This is the same "local pose" process_file's get_rifke()
    already produces internally, just re-read from the packed feature vector.
    Column layout: [0]=rot_vel [1:3]=lin_vel_xz [3]=root_y [4:4+63]=ric_data.
    """
    root_y = data_263[:, 3]
    ric = data_263[:, 4:4 + (joints_num - 1) * 3].reshape(-1, joints_num - 1, 3)
    root = np.zeros((data_263.shape[0], 1, 3), dtype=data_263.dtype)
    root[:, 0, 1] = root_y
    return np.concatenate([root, ric], axis=1)  # (T, joints_num, 3)


def recover_positions(data_263: np.ndarray) -> np.ndarray:
    """Inverse: 263-dim -> (T, 22, 3) joint positions, Y-up (HML convention).
    Thin wrapper around the vendored recover_from_ric (also available,
    identically, in T2M-GPT/utils/motion_process.py)."""
    data_t = torch.from_numpy(data_263).unsqueeze(0).float()
    joints = _mp.recover_from_ric(data_t, NUM_JOINTS)[0]
    return joints.numpy()
