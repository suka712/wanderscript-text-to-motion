#!/usr/bin/env python3
"""Multi-segment rollout — chaining (build-order step 9).

Generates a chain of segments, each conditioned on the previous segment's
DECODED ending pose, placing each into the world with SE(2). This is the first
code in the project that produces motion longer than one segment.

THE THREE RULES IT OBEYS, all established by earlier stages:
 1. Goal is fed start-relative and heading-aligned (RESULTS §4). Absolute world
    coordinates do not work.
 2. The prefix is the ending BODY CONFIGURATION -- root-relative,
    heading-canonicalized joint positions, which are frame-independent -- not
    the previous segment's tokens, which describe a different canonical frame
    (RESULTS §7).
 3. The pose fed forward is the DECODED one, never a blended or smoothed one
    (RESULTS §7). An off-manifold prefix collapses the model: iid noise at 25mm
    costs 331mm of seam, while a VQ-VAE-reconstructed prefix at ~70mm costs 15.
    So when a seam blend is applied for display, it is applied to the OUTPUT
    only and never fed back as conditioning.

Next-segment heading is recovered from the generated body itself, using the
same hip/shoulder face-direction formula as humanise_join.compute_track2, so
the chain's own geometry decides where it is pointing rather than an assumption.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
for p in ["src", "scripts/track1", "scripts/scene_probe"]:
    sys.path.insert(0, os.path.join(REPO_ROOT, p))

import motion_features as mf  # noqa: E402
from humanise_join import J_LHIP, J_RHIP, J_LSHOULDER, J_RSHOULDER  # noqa: E402
from se2_utils import se2_place_full_body, world_to_local_xy  # noqa: E402
from train_probe import build_transformer, COND_EXTRA_DIMS  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OCC_N = 28


def yaw_from_joints(p22):
    """World yaw from a single (22,3) Z-up pose, matching compute_track2's
    convention (yaw 0 = facing +X) so chained poses stay in one frame."""
    across = (p22[J_RHIP, :2] - p22[J_LHIP, :2]) + (p22[J_RSHOULDER, :2] - p22[J_LSHOULDER, :2])
    n = np.linalg.norm(across)
    across = across / (n if n > 1e-8 else 1e-8)
    fwd = np.array([-across[1], across[0]])
    return float(np.arctan2(fwd[1], fwd[0]))


def occ_crop(occ, extent, xy, yaw):
    from scene_probe import crop_agent_frame
    c = crop_agent_frame(occ, extent, xy, yaw)
    k = c.shape[0] // OCC_N
    return c.reshape(OCC_N, k, OCC_N, k).mean((1, 3)).ravel().astype(np.float32)


def build_cond(cond_mode, goal_world, start_pose, prefix_pose, occ, extent, cmean, cstd):
    parts = [world_to_local_xy(goal_world, start_pose).ravel()]
    if cond_mode in ("rel_prefix", "full"):
        parts.append(prefix_pose)
    if cond_mode == "full":
        yaw = float(np.arctan2(start_pose[2], start_pose[3]))
        parts.append(occ_crop(occ, extent, start_pose[:2], yaw))
    raw = np.concatenate(parts).astype(np.float32)
    return ((raw - cmean) / cstd).astype(np.float32)


def rollout(trans, net, clip_model, clip_mod, mean, std, ns, texts, goals,
            start_pose, prefix_pose, occ=None, extent=None, max_seg=None):
    """Chain len(goals) segments. Returns list of per-segment dicts."""
    cmean = np.array(ns["cond_mean"], np.float32)
    cstd = np.array(ns["cond_std"], np.float32)
    pose = np.asarray(start_pose, dtype=np.float32).copy()
    prefix = np.asarray(prefix_pose, dtype=np.float32).copy()
    segs = []
    with torch.no_grad():
        for k, (txt, goal) in enumerate(zip(texts, goals)):
            if max_seg and k >= max_seg:
                break
            feat = clip_model.encode_text(
                clip_mod.tokenize([txt], truncate=True).to(DEV)).float()
            extra = build_cond(ns["cond_mode"], np.asarray(goal, float), pose,
                               prefix, occ, extent, cmean, cstd)
            cond = torch.cat([feat, torch.from_numpy(extra).unsqueeze(0).to(DEV)], -1)
            tok = trans.sample(cond, if_categorial=False)
            if tok.numel() == 0:
                break
            motion = net.forward_decoder(tok)[0].cpu().numpy() * std + mean
            world = se2_place_full_body(motion.astype(np.float32), pose, mf)  # (T,22,3) Z-up

            local = mf.local_joint_positions(motion.astype(np.float32))
            segs.append({
                "world": world,
                "goal": np.asarray(goal, float),
                "start_pose": pose.copy(),
                "seam_err": float(np.linalg.norm(local[0] - prefix.reshape(22, 3), axis=-1).mean()),
                "goal_err": float(np.linalg.norm(world[-1, 0, :2] - np.asarray(goal, float))),
                "text": txt,
            })
            # hand off: DECODED ending pose, and heading from the generated body
            end_xy = world[-1, 0, :2]
            end_yaw = yaw_from_joints(world[-1])
            pose = np.array([end_xy[0], end_xy[1], np.sin(end_yaw), np.cos(end_yaw)], np.float32)
            prefix = local[-1].ravel().astype(np.float32)
    return segs


def blend_seam(a_world, b_world, n=4):
    """Cosmetic 4-frame crossfade for DISPLAY ONLY (CLAUDE.md 2d). Never feed
    the blended pose back as conditioning -- it is off-manifold."""
    if a_world is None or len(b_world) < n or len(a_world) < n:
        return b_world
    out = b_world.copy()
    w = np.linspace(0, 1, n + 2)[1:-1][:, None, None]
    out[:n] = (1 - w) * a_world[-1][None] + w * b_world[:n]
    return out


def load_model(ckpt_dir):
    ns = json.load(open(os.path.join(ckpt_dir, "norm_stats.json")))
    tr = build_transformer(ns["clip_dim"])
    tr.load_state_dict(torch.load(os.path.join(ckpt_dir, "net_final.pth"),
                                  map_location="cpu")["trans"], strict=True)
    return tr.eval().to(DEV), ns
