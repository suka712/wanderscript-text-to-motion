#!/usr/bin/env python3
"""
Grounding probe eval (docs/track_1/001_grounding_probe.md, "Probe" +
"Decision gate"). Held-out generation: text (+ start + goal, if the model
was trained conditioned) -> frozen decoder -> canonicalized motion -> SE(2)
place at the FED start pose -> world-frame motion. Reports, per clip:
  goal-error   = distance(generated end xy, fed goal xy)
  start-error  = distance(generated start xy, fed start xy)  -- expected to
    be ~0 for ANY model (conditioned or not): the SE(2) placement composes
    the local trajectory's frame-0 (which sits at local origin by
    construction, see se2_utils.py) directly onto the fed start pose, so
    this is a sanity check on the placement code, not a learned behaviour.
    The real signal is goal-error, compared between the conditioned and
    unconditioned runs per the REQUIRED-baseline decision gate.
"""
import argparse
import json
import os
import pickle
import sys

import clip
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))

import motion_features as mf  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402
from se2_utils import se2_place  # noqa: E402
from train_probe import build_transformer, CLIP_DIM_BASE  # noqa: E402

MOTION_DATA_ROOT = os.environ.get("WANDER_MOTION_DATA_ROOT", "/media/user/2tb/motion_data")
OUT_DIR = os.environ.get("WANDER_TRACK1_PROBE_ROOT", os.path.join(os.path.dirname(MOTION_DATA_ROOT), "track1_probe"))
TOKENS_DIR = os.path.join(OUT_DIR, "tokens")
T2M_GPT_ROOT = os.environ.get("WANDER_T2M_GPT_ROOT", "/home/user/Khiem-ssh/T2M-GPT")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_cond(conditioned, feat_clip_text, start, goal, xy_mean, xy_std):
    if not conditioned:
        return feat_clip_text
    start_n = start.copy()
    start_n[:2] = (start_n[:2] - xy_mean) / xy_std
    goal_n = (goal - xy_mean) / xy_std
    extra = torch.from_numpy(np.concatenate([start_n, goal_n]).astype(np.float32)).unsqueeze(0).to(DEVICE)
    return torch.cat([feat_clip_text, extra], dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-name", type=str, required=True, help="dir name under checkpoints/")
    ap.add_argument("--n-clips", type=int, default=200)
    ap.add_argument("--n-renders", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ckpt_dir = os.path.join(OUT_DIR, "checkpoints", args.ckpt_name)
    with open(os.path.join(ckpt_dir, "norm_stats.json")) as f:
        norm_stats = json.load(f)
    conditioned = norm_stats["conditioned"]
    xy_mean = np.array(norm_stats["xy_mean"], dtype=np.float32)
    xy_std = np.array(norm_stats["xy_std"], dtype=np.float32)

    with open(os.path.join(TOKENS_DIR, "test.pkl"), "rb") as f:
        test_manifest = pickle.load(f)
    rng = np.random.RandomState(args.seed)
    idxs = rng.choice(len(test_manifest), size=min(args.n_clips, len(test_manifest)), replace=False)

    net = load_vqvae(device=DEVICE)
    mean = np.load(f"{T2M_GPT_ROOT}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy").astype(np.float32)
    std = np.load(f"{T2M_GPT_ROOT}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy").astype(np.float32)
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE, jit=False)
    clip_model.eval()

    trans_encoder = build_transformer(norm_stats["clip_dim"])
    ckpt = torch.load(os.path.join(ckpt_dir, "net_final.pth"), map_location="cpu")
    trans_encoder.load_state_dict(ckpt["trans"], strict=True)
    trans_encoder.eval()
    trans_encoder.to(DEVICE)

    goal_errs, start_errs = [], []
    renders = []
    with torch.no_grad():
        for n, i in enumerate(idxs):
            d = test_manifest[i]
            text_tok = clip.tokenize([d["text"]], truncate=True).to(DEVICE)
            feat_clip_text = clip_model.encode_text(text_tok).float()
            cond = build_cond(conditioned, feat_clip_text, d["start"], d["goal"], xy_mean, xy_std)

            tokens = trans_encoder.sample(cond, if_categorial=False)
            if tokens.numel() == 0:
                continue
            motion_norm = net.forward_decoder(tokens)[0].cpu().numpy()  # (T', 263), normalized space
            motion = motion_norm * std + mean  # denormalize -- see prepare_probe_data.py's note

            world_xy = se2_place(motion, d["start"], mf)
            gen_start, gen_end = world_xy[0], world_xy[-1]
            fed_start_xy = d["start"][:2]
            fed_goal_xy = d["goal"]

            goal_err = float(np.linalg.norm(gen_end - fed_goal_xy))
            start_err = float(np.linalg.norm(gen_start - fed_start_xy))
            goal_errs.append(goal_err)
            start_errs.append(start_err)

            if len(renders) < args.n_renders:
                renders.append((d, world_xy, fed_start_xy, fed_goal_xy, goal_err))

            if (n + 1) % 50 == 0:
                print(f"  {n + 1}/{len(idxs)}", flush=True)

    goal_errs = np.array(goal_errs)
    start_errs = np.array(start_errs)
    summary = {
        "ckpt_name": args.ckpt_name,
        "conditioned": conditioned,
        "n": len(goal_errs),
        "goal_err_mean": float(goal_errs.mean()),
        "goal_err_median": float(np.median(goal_errs)),
        "goal_err_std": float(goal_errs.std()),
        "start_err_mean": float(start_errs.mean()),
        "start_err_median": float(np.median(start_errs)),
    }
    print(json.dumps(summary, indent=2))

    render_dir = os.path.join(OUT_DIR, "renders", args.ckpt_name)
    os.makedirs(render_dir, exist_ok=True)
    for k, (d, world_xy, fed_start_xy, fed_goal_xy, goal_err) in enumerate(renders):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(world_xy[:, 0], world_xy[:, 1], "-o", color="tab:blue", markersize=2, label="generated")
        ax.plot(*fed_start_xy, "o", color="tab:green", markersize=10, label="fed start")
        ax.plot(*fed_goal_xy, "*", color="tab:red", markersize=15, label="fed goal")
        ax.set_title(f"{d['text'][:40]}\ngoal_err={goal_err:.2f}m", fontsize=9)
        ax.set_aspect("equal")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(render_dir, f"example_{k:02d}.png"), dpi=100)
        plt.close(fig)

    with open(os.path.join(ckpt_dir, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"renders written to {render_dir}")


if __name__ == "__main__":
    main()
