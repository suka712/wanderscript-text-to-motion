#!/usr/bin/env python3
"""
Converter exact-match / correctness check against H3D's OWN shipped
precomputed 263-dim files.

INTENDED test (per request): raw HumanML3D mocap joints -> motion_features.py
-> compare to H3D's shipped new_joint_vecs/*.npy. This machine does NOT have
H3D's raw joint positions -- confirmed by exhaustive search: H3D/,
HumanML3D/, T2M-GPT/dataset/HumanML3D, motion-diffusion-model/dataset/
HumanML3D all ship ONLY new_joint_vecs (precomputed 263-dim), texts, and
contact annotations -- no raw joints/pose_data directory anywhere. (There IS
a /media/user/2tb/motion_data/HumanML3D/contact_motion/motions/ directory,
but its joint data is natively Z-up like HUMANISE's, meaning it was produced
by an independent SMPL-X refit of HumanML3D motions, NOT the original mocap
joints that produced new_joint_vecs -- using it would confound "does our
converter have a bug" with "how much does the SMPL-X refit differ from the
original mocap," which defeats the purpose. Not used here.)

SUBSTITUTE actually run: since we lack raw joints, take H3D's shipped
263-dim as ground truth, recover joint positions from it (recover_from_ric,
the vendored INVERSE), then re-encode those recovered joints through OUR
motion_features.extract_263 (the SAME process_file wrapper used for
HUMANISE), and compare the re-encoded 263-dim against the ORIGINAL shipped
263-dim, per feature block. This still satisfies "same reference skeleton so
retargeting cancels, expect near-zero" (the recovered joints are already at
the canonical target skeleton's bone lengths, since that's what the shipped
file encodes) and still exercises every line of process_file (fps
assumption, foot-contact threshold, rotation continuity/IK, floor/origin/
heading normalization) on real, official H3D data -- it just tests
"joints->263->joints->263 self-consistency" rather than "raw mocap->263",
because raw mocap isn't available here. This is a WEAKER claim than the
originally requested test and that gap is intentional to disclose, not
paper over: a bug shared identically by process_file and recover_from_ric
(e.g. a systematic rotation-convention error present in both) would NOT be
caught by this test, only bugs where our specific invocation (globals,
tgt_offsets derivation, feet_thre) diverges from whatever produced the
shipped files.

Feature block layout (263-dim, joints_num=22), per motion_process.py:
  [0]      rot_vel          (root angular velocity, y-axis)
  [1:3]    lin_vel_xz       (root linear velocity, xz-plane)
  [3]      root_y           (root height)
  [4:67]   ric_data         (63 = (22-1)*3, rotation-invariant local joint pos)
  [67:193] rot_data         (126 = (22-1)*6, continuous 6D joint rotations)
  [193:259] local_vel       (66 = 22*3, per-joint velocity)
  [259:263] feet_contact    (4 binary foot-contact flags)
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, "/home/user/Khiem-ssh/wander/src")
warnings.filterwarnings("ignore")

import motion_features as mf  # noqa: E402

H3D_ROOT = "/media/user/2tb/motion_data/H3D"
N_CLIPS = 15

BLOCKS = [
    ("rot_vel", 0, 1),
    ("lin_vel_xz", 1, 3),
    ("root_y", 3, 4),
    ("ric_data", 4, 67),
    ("rot_data", 67, 193),
    ("local_vel", 193, 259),
    ("feet_contact", 259, 263),
]


def per_block_mae(a, b):
    T = min(a.shape[0], b.shape[0])
    a, b = a[:T], b[:T]
    out = {}
    for name, lo, hi in BLOCKS:
        out[name] = np.abs(a[:, lo:hi] - b[:, lo:hi]).mean()
    out["_all"] = np.abs(a - b).mean()
    out["_T_a"] = a.shape[0]
    out["_T_b"] = b.shape[0]
    return out


def main():
    files = sorted(os.listdir(f"{H3D_ROOT}/new_joint_vecs"))
    files = [f for f in files if not f.startswith("M")]  # skip mirrored-augmentation copies
    rng = np.random.RandomState(0)
    sample = list(rng.choice(files, size=N_CLIPS, replace=False))
    if "000021.npy" not in sample:  # the reference-skeleton clip itself -- special case
        sample = ["000021.npy"] + sample

    print(f"{'clip':<14}{'T':<6}" + "".join(f"{n:<14}" for n, _, _ in BLOCKS) + "all(mm-ish)")
    agg = {name: [] for name, _, _ in BLOCKS}
    agg["_all"] = []
    for fn in sample:
        shipped = np.load(f"{H3D_ROOT}/new_joint_vecs/{fn}").astype(np.float32)
        if shipped.shape[0] < 6:
            continue
        joints = mf.recover_positions(shipped)  # (T,22,3) Y-up, our own inverse
        reencoded, _, _, _ = mf.extract_263(joints)  # our converter, same code path as HUMANISE

        stats = per_block_mae(shipped, reencoded)
        row = f"{fn:<14}{stats['_T_a']:<6}"
        for name, _, _ in BLOCKS:
            row += f"{stats[name]:<14.6f}"
            agg[name].append(stats[name])
        row += f"{stats['_all']:.6f}"
        agg["_all"].append(stats["_all"])
        print(row)

        # mm-scale MPJPE (local, root-relative -- consistent with check7/13/16's
        # metric convention) for direct comparability with earlier numbers.
        shipped_local = mf.local_joint_positions(shipped)
        reencoded_local = mf.local_joint_positions(reencoded.astype(np.float32))
        T = min(shipped_local.shape[0], reencoded_local.shape[0])
        mpjpe_mm = np.linalg.norm(shipped_local[:T] - reencoded_local[:T], axis=-1).mean() * 1000
        agg.setdefault("_mpjpe_mm", []).append(mpjpe_mm)

    print("=" * 60)
    print("AGGREGATE (mean absolute error per block, across all clips):")
    for name, _, _ in BLOCKS:
        vals = np.array(agg[name])
        print(f"  {name:<14} mean={vals.mean():.6f}  max={vals.max():.6f}")
    all_vals = np.array(agg["_all"])
    print(f"  {'ALL (263-d)':<14} mean={all_vals.mean():.6f}  max={all_vals.max():.6f}")
    mpjpe_vals = np.array(agg["_mpjpe_mm"])
    print(f"  {'MPJPE (mm)':<14} mean={mpjpe_vals.mean():.4f}  max={mpjpe_vals.max():.4f}")

    print()
    worst = max(BLOCKS, key=lambda nb: np.mean(agg[nb[0]]))
    print(f"Worst-diverging block: {worst[0]} "
          f"(mean MAE {np.mean(agg[worst[0]]):.6f} vs. overall {all_vals.mean():.6f})")


if __name__ == "__main__":
    main()
