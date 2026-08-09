#!/usr/bin/env python3
"""
Converter-only round-trip fidelity, NO VQ-VAE in the loop:

    raw (T,22,3) SMPL-X-derived joints (Z-up)
        -> motion_features.zup_to_yup_hml()          [axis relabel]
        -> motion_features.extract_263()              [[Path 1]] this converter
        -> motion_features.recover_positions()         [inverse]
        -> rigid align to raw input's frame-0 root pose (see below)
        -> compare against the Y-up-relabeled input

NOTE on alignment: the 263-dim representation is deliberately translation/
rotation-invariant (CLAUDE.md 2a), so recover_from_ric always reconstructs
starting at world-origin XZ with zero initial heading, independent of where
the raw clip actually was in the room. Comparing raw vs. recovered GLOBAL
positions directly conflates "did the shape/motion round-trip faithfully"
with "the representation doesn't retain absolute placement" (which is by
design, not a bug). This script rigid-aligns (yaw rotation + XZ translation
only, frame-0 root pose) before computing MPJPE, isolating the former.

This isolates the converter's own round-trip fidelity from the VQ-VAE's
lossiness (all of check7/13/15/16/17 measure REAL-vs-RECON, i.e. converter
output through the VQ-VAE -- this script never touches the VQ-VAE).

IMPORTANT CAVEAT: process_file's first step (uniform_skeleton, in the
vendored motion_process.py) retargets the input skeleton's bone LENGTHS to
a canonical reference skeleton (H3D's "000021" clip), before doing anything
else. This is a deliberate, documented part of the HumanML3D representation
-- not a bug -- so MPJPE here reflects retargeting-to-canonical-proportions
error, not just numerical round-trip error. It will not be exactly zero even
for a perfect implementation, and will be larger for a person whose bone
lengths differ more from the canonical reference. Do not treat a small
nonzero number as a defect without checking whether it's dominated by
retargeting vs. an actual bug.
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
warnings.filterwarnings("ignore")

import motion_features as mf  # noqa: E402
from humanise_join import build_flat_join  # noqa: E402

HUMANISE_MOTIONS = os.environ.get("WANDER_HUMANISE_ROOT", "/media/user/2tb/motion_data/HUMANISE") + "/contact_motion/motions"
N_PER_ACTION = 3


def mpjpe_per_frame(pos_a, pos_b):
    T = min(pos_a.shape[0], pos_b.shape[0])
    d = np.linalg.norm(pos_a[:T] - pos_b[:T], axis=-1)  # (T, 22)
    return d.mean(axis=1), d  # per-frame mean, full (T,22) error


def heading_angle_frame0(positions_yup):
    """Same convention as motion_process.recover_root_rot_heading_ang, but
    numpy, frame 0 only. face_joint_idx = [2,1,17,16] = r_hip, l_hip, r_sdr,
    l_sdr (Y-up joint order)."""
    p0 = positions_yup[0]
    r_hip, l_hip, sdr_r, sdr_l = 2, 1, 17, 16
    across = (p0[r_hip] - p0[l_hip]) + (p0[sdr_r] - p0[sdr_l])
    across = across / (np.linalg.norm(across) + 1e-8)
    forward = np.cross(np.array([0.0, 1.0, 0.0]), across)
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    return np.arctan2(forward[0], forward[2])


def align_recovered_to_raw(raw_yup, recovered_yup):
    """recover_from_ric always reconstructs starting at world-origin XZ with
    zero initial heading (by construction of recover_root_rot_pos's cumsum
    starting at 0) -- this is the DESIGN of the canonicalized representation
    (CLAUDE.md 2a: translation/rotation invariant), not a converter defect.
    Rotate+translate the recovered trajectory (rigid, preserves shape/MPJPE
    meaning) so its frame-0 root pose matches the raw input's frame-0 root
    pose, making a subsequent MPJPE comparison actually meaningful."""
    theta = heading_angle_frame0(raw_yup) - heading_angle_frame0(recovered_yup)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    rotated = recovered_yup @ rot.T
    offset = raw_yup[0, 0] - rotated[0, 0]  # align root (joint 0), frame 0
    return rotated + offset


def main():
    flat = build_flat_join()
    by_action = {}
    for i, p in enumerate(flat):
        by_action.setdefault(p["action"], []).append(i)

    rng = np.random.RandomState(99)
    for action in ["walk", "lie"]:
        candidates = list(by_action[action])
        rng.shuffle(candidates)
        chosen = []
        for i in candidates:
            cm = np.load(f"{HUMANISE_MOTIONS}/{i:05d}.npy")
            if cm.shape[0] >= 40:
                chosen.append(i)
            if len(chosen) == N_PER_ACTION:
                break

        print("=" * 70)
        print(f"ACTION: {action}  (converter-only round-trip, NO VQ-VAE)")
        all_frame_errs = []
        for idx in chosen:
            cm = np.load(f"{HUMANISE_MOTIONS}/{idx:05d}.npy").astype(np.float32)
            positions_yup = mf.zup_to_yup_hml(cm)  # exactly what process_file consumes
            data263, _, _, _ = mf.extract_263(positions_yup)
            recovered = mf.recover_positions(data263.astype(np.float32))  # Y-up out
            recovered_aligned = align_recovered_to_raw(positions_yup, recovered)

            per_frame, full = mpjpe_per_frame(positions_yup, recovered_aligned)
            all_frame_errs.append(per_frame)
            print(f"  clip {idx:05d}  T={cm.shape[0]:<4}  "
                  f"MPJPE mean={per_frame.mean()*1000:7.2f}mm  "
                  f"median={np.median(per_frame)*1000:7.2f}mm  "
                  f"max={per_frame.max()*1000:7.2f}mm  "
                  f"(frame0={per_frame[0]*1000:.2f}mm, "
                  f"frame{len(per_frame)-1}={per_frame[-1]*1000:.2f}mm)")

        cat = np.concatenate(all_frame_errs)
        print(f"  -- {action} aggregate over {len(chosen)} clips, {len(cat)} frames: "
              f"mean={cat.mean()*1000:.2f}mm  median={np.median(cat)*1000:.2f}mm  "
              f"max={cat.max()*1000:.2f}mm")


if __name__ == "__main__":
    main()
