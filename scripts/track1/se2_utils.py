"""
SE(2) placement helpers for the grounding probe eval (docs/track_1/
001_grounding_probe.md, "Probe"): take a generated, canonicalized 263-dim
clip and place its implied root trajectory into the world frame at a given
start pose, so it can be compared against the fed goal coordinate.

recover_positions (motion_features.py, validated/frozen -- reused, not
reimplemented) already reintroduces the LOCAL root trajectory via cumulative
integration of the 263-dim's per-frame velocities, with frame 0 at local
origin by construction of that integration (same property CLAUDE.md 2a
describes for the canonicalized representation in general). What's added
here is only the small glue this repo doesn't already have: undoing the
Y-up relabel (motion_features.zup_to_yup_hml's inverse) and composing the
resulting local (x, y) trajectory onto an arbitrary world start pose via a
rigid SE(2) transform (rotate by the start yaw, then translate).
"""
import numpy as np

from humanise_join import J_PELVIS


def yup_to_zup(positions_yup: np.ndarray) -> np.ndarray:
    """Inverse of motion_features.zup_to_yup_hml (x_hml=x_zup, y_hml=z_zup,
    z_hml=-y_zup) -- recovers HUMANISE/ScanNet's native Z-up frame from
    HumanML3D's Y-up convention."""
    out = np.empty_like(positions_yup)
    out[..., 0] = positions_yup[..., 0]
    out[..., 1] = -positions_yup[..., 2]
    out[..., 2] = positions_yup[..., 1]
    return out


# Constant offset between the two yaw conventions in play. HumanML3D's
# process_file canonicalizes frame 0 to face +Z in ITS Y-up frame ("All
# initially face Z+"); under zup_to_yup_hml (y_hml = z_zup, z_hml = -y_zup)
# that direction is -Y in the Z-up world frame, i.e. world yaw -pi/2. But
# humanise_join.compute_track2 defines yaw = 0 as facing +X. So composing a
# recovered local trajectory onto a world start pose must rotate by
# (yaw0 + pi/2), not yaw0.
#
# Getting this wrong rotates every placed trajectory 90 degrees about the fed
# start. It is invisible to a start-error check (frame 0 sits at the local
# origin either way, so start-error is exactly 0.0 regardless) -- the only
# control that catches it is a ground-truth-token round trip. Measured on 200
# held-out clips: 0.862m mean end-point error at offset 0, 0.124m at +pi/2.
CANONICAL_YAW_OFFSET = np.pi / 2


def rot_z2d(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def start_pose_rotation(start_pose) -> np.ndarray:
    """SE(2) rotation taking the canonicalized LOCAL frame to the world frame,
    for a clip placed at `start_pose` = (x0, y0, sin_yaw0, cos_yaw0)."""
    _, _, s0, c0 = start_pose
    return rot_z2d(np.arctan2(s0, c0) + CANONICAL_YAW_OFFSET)


def world_to_local_xy(world_xy: np.ndarray, start_pose) -> np.ndarray:
    """Inverse of se2_place: express a world (..., 2) point in the frame the
    model actually generates in. Used to build start-relative, heading-aligned
    goal conditioning (see train_probe.py --cond-mode rel)."""
    x0, y0, _, _ = start_pose
    R = start_pose_rotation(start_pose)
    return (np.asarray(world_xy, dtype=np.float64) - np.array([x0, y0])) @ R


def local_xy_trajectory(data263: np.ndarray, mf_module) -> np.ndarray:
    """canonicalized (T, 263) -> local (T, 2) pelvis xy trajectory, Z-up,
    frame 0 at local origin (structural property of the representation, see
    module docstring)."""
    joints_yup = mf_module.recover_positions(data263)  # (T, 22, 3)
    joints_zup = yup_to_zup(joints_yup)
    return joints_zup[:, J_PELVIS, :2].astype(np.float64)


def se2_place(data263: np.ndarray, start_pose, mf_module) -> np.ndarray:
    """Compose the clip's local xy trajectory onto a world start pose.
    start_pose: (x0, y0, sin_yaw0, cos_yaw0). Returns world_xy (T, 2).
    """
    local_xy = local_xy_trajectory(data263, mf_module)
    x0, y0, _, _ = start_pose
    R = start_pose_rotation(start_pose)
    return local_xy @ R.T + np.array([x0, y0])


def se2_place_full_body(data263: np.ndarray, start_pose, mf_module) -> np.ndarray:
    """Same composition as se2_place, but applied to the FULL (T, 22, 3)
    joint set (all limbs), not just the pelvis xy -- for rendering an
    animated body in world coordinates rather than just a root trajectory.
    Z (height) is untouched by an SE(2) transform. Returns (T, 22, 3), Z-up.
    """
    joints_yup = mf_module.recover_positions(data263)
    joints_zup = yup_to_zup(joints_yup)
    x0, y0, _, _ = start_pose
    R = start_pose_rotation(start_pose)
    xy = joints_zup[..., :2] @ R.T + np.array([x0, y0])
    out = joints_zup.copy()
    out[..., :2] = xy
    return out
