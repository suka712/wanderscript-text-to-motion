#!/usr/bin/env python3
"""Diagnostic: does the model sit facing the RIGHT way, and can we steer it? (Step 11)

A viewer noticed the model sits in the wrong direction -- it performs a sit without knowing
which way the furniture faces. Root cause: the scene conditioning is an occupancy FOOTPRINT
(orientation-agnostic), so nothing tells the model the furniture's facing; the seated
orientation just follows the approach / a learned prior.

Unlike the walk heading (redundant with the goal, so the model ignores it -- RESULTS §11), a
SIT facing is NOT derivable from the goal position, so the full_action_head heading input could
carry it -- IF the model responds to it. This measures both:
  baseline    generate the sit with the normal inference command; |gen facing - GT facing|.
  cmd=GT      command the target heading = GT seated facing; error should collapse if steerable.
  cmd=GT+180  command the opposite; gen should flip ~180deg from cmd=GT if steerable.
The gap between cmd=GT and cmd=GT+180 is the real test: large => the model obeys the sit-facing
command (fixable at inference by supplying furniture orientation); ~0 => it ignores it.
"""
import argparse
import os
import sys

import clip
import numpy as np
import torch

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
for p in ["src", "scripts/track1", "scripts/chaining"]:
    sys.path.insert(0, os.path.join(REPO_ROOT, p))

import motion_features as mf  # noqa: E402
from humanise_join import build_flat_join, get_record, compute_track2, J_PELVIS  # noqa: E402
from se2_utils import yup_to_zup  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402
from rollout import rollout, load_model, yaw_from_joints  # noqa: E402

T2M = os.environ.get("WANDER_T2M_GPT_ROOT")
HUMANISE = os.environ.get("WANDER_HUMANISE_ROOT")
BEV = os.path.expanduser("~/wander_data/bev_cache")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SIT_Z = 0.70


def wrapdeg(a):
    return abs((np.degrees(a) + 180) % 360 - 180)


def gt_end_z(cm):
    d263, *_ = mf.humanise_positions_to_263(cm)
    j = yup_to_zup(mf.recover_positions(d263.astype(np.float32)))
    return j[-1, J_PELVIS, 2]


def gen_sit(trans, net, cmodel, mean, std, ns, occ, extent, rec, start_pose, goal, prefix,
            head_target):
    segs = rollout(trans, net, cmodel, clip, mean, std, ns, [rec.utterance], [goal],
                   start_pose, prefix, occ, extent, actions=["sit"],
                   head_targets=None if head_target is None else [head_target])
    if not segs:
        return None
    w = segs[0]["world"]
    return yaw_from_joints(w[-1]), float(w[-1, J_PELVIS, 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vqvae-ckpt", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    net = load_vqvae(ckpt_path=args.vqvae_ckpt, device=DEV); net.eval()
    global mean, std
    mean = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy").astype(np.float32)
    std = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy").astype(np.float32)
    cmodel, _ = clip.load("ViT-B/32", device=DEV, jit=False); cmodel.eval()
    trans, ns = load_model(args.ckpt)
    print(f"model cond_mode={ns['cond_mode']}")
    if ns["cond_mode"] != "full_action_head":
        print("NOTE: model has no heading input; only the baseline column is meaningful.")

    flat = build_flat_join()
    idxs = [i for i, p in enumerate(flat) if p["action"] == "sit"]
    rng = np.random.RandomState(args.seed); rng.shuffle(idxs)
    base, cgt, copp, flip = [], [], [], []
    used = 0
    scenes = {}
    for idx in idxs:
        if used >= args.n:
            break
        rec = get_record(int(idx))
        f = os.path.join(BEV, f"{rec.scene}.npz")
        if not os.path.exists(f):
            continue
        cm = np.load(os.path.join(HUMANISE, "contact_motion", "motions", f"{idx:05d}.npy"))
        if cm.shape[0] < 8 or gt_end_z(cm) >= SIT_Z:
            continue
        try:
            d0, *_ = mf.humanise_positions_to_263(cm)
        except Exception:
            continue
        _, xy, _, sincos = compute_track2(rec)
        gt_yaw = float(np.arctan2(sincos[-1, 0], sincos[-1, 1]))     # GT seated facing
        start_pose = np.array([xy[0, 0], xy[0, 1], sincos[0, 0], sincos[0, 1]], np.float32)
        goal = xy[-1].astype(np.float32)
        prefix = mf.local_joint_positions(d0.astype(np.float32))[0].ravel()
        if rec.scene not in scenes:
            if len(scenes) > 40:
                scenes.clear()
            z = np.load(f); scenes[rec.scene] = (z["occ"].astype(np.float32), z["extent"])
        occ, extent = scenes[rec.scene]

        r_auto = gen_sit(trans, net, cmodel, mean, std, ns, occ, extent, rec, start_pose, goal, prefix, None)
        if r_auto is None:
            continue
        base.append(wrapdeg(r_auto[0] - gt_yaw))
        if ns["cond_mode"] == "full_action_head":
            r_gt = gen_sit(trans, net, cmodel, mean, std, ns, occ, extent, rec, start_pose, goal, prefix, gt_yaw)
            r_op = gen_sit(trans, net, cmodel, mean, std, ns, occ, extent, rec, start_pose, goal, prefix, gt_yaw + np.pi)
            if r_gt and r_op:
                cgt.append(wrapdeg(r_gt[0] - gt_yaw))
                copp.append(wrapdeg(r_op[0] - (gt_yaw + np.pi)))
                flip.append(wrapdeg(r_gt[0] - r_op[0]))
        used += 1

    print(f"\n=== sit facing, {used} clips (GT interacts) ===")
    print(f"baseline  |gen - GT seated facing|      : {np.mean(base):.0f} deg  "
          f"(90 = random-ish, 0 = perfect)")
    if cgt:
        print(f"cmd=GT    |gen - GT|                    : {np.mean(cgt):.0f} deg")
        print(f"cmd=GT+180 |gen - commanded|            : {np.mean(copp):.0f} deg")
        print(f"|gen(cmd=GT) - gen(cmd=GT+180)|         : {np.mean(flip):.0f} deg  "
              f"(~180 => the model OBEYS the sit-facing command; ~0 => ignores it)")


if __name__ == "__main__":
    main()
