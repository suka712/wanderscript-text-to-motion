#!/usr/bin/env python3
"""
Presentation-friendly mp4s of the grounding probe: REAL (ground-truth) vs
GENERATED (from a trained probe checkpoint) stick figures side by side, both
placed in the same world frame via SE(2) composition (se2_utils.py), with
the fed goal marked as a static pole so a viewer can see whether the
generated motion actually reaches it. Reuses check17_recon_video.py's
validated matplotlib-3D + ffmpeg pattern (kinematic chain, fixed camera box,
FFMpegWriter) -- not modified, just re-applied to world-placed clips instead
of local reconstruction pairs.
"""
import argparse
import os
import pickle
import sys

import clip
import numpy as np
import torch

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))

import motion_features as mf  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402
from se2_utils import se2_place_full_body  # noqa: E402
from train_probe import build_transformer  # noqa: E402

HUMANISE_ROOT = os.environ.get("WANDER_HUMANISE_ROOT", "/media/user/2tb/motion_data/HUMANISE")
T2M_GPT_ROOT = os.environ.get("WANDER_T2M_GPT_ROOT", "/home/user/Khiem-ssh/T2M-GPT")
MOTION_DATA_ROOT = os.environ.get("WANDER_MOTION_DATA_ROOT", "/media/user/2tb/motion_data")
OUT_DIR = os.environ.get("WANDER_TRACK1_PROBE_ROOT", os.path.join(os.path.dirname(MOTION_DATA_ROOT), "track1_probe"))
TOKENS_DIR = os.path.join(OUT_DIR, "tokens")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FPS = 20
MAX_T = 196

KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]


def crop_to_multiple(T, factor=4, max_t=MAX_T):
    T = min(T, max_t)
    return (T // factor) * factor


def load_real_world_joints(idx, start_pose, mean, std):
    cm = np.load(os.path.join(HUMANISE_ROOT, "contact_motion", "motions", f"{idx:05d}.npy"))
    data263_full, *_ = mf.humanise_positions_to_263(cm)
    T = crop_to_multiple(data263_full.shape[0])
    data263 = data263_full[:T].astype(np.float32)
    return se2_place_full_body(data263, start_pose, mf)


def generate_world_joints(d, trans_encoder, net, clip_model, conditioned, xy_mean, xy_std, mean, std):
    text_tok = clip.tokenize([d["text"]], truncate=True).to(DEVICE)
    with torch.no_grad():
        feat_clip_text = clip_model.encode_text(text_tok).float()
        if conditioned:
            start_n = d["start"].copy()
            start_n[:2] = (start_n[:2] - xy_mean) / xy_std
            goal_n = (d["goal"] - xy_mean) / xy_std
            extra = torch.from_numpy(np.concatenate([start_n, goal_n]).astype(np.float32)).unsqueeze(0).to(DEVICE)
            cond = torch.cat([feat_clip_text, extra], dim=-1)
        else:
            cond = feat_clip_text
        tokens = trans_encoder.sample(cond, if_categorial=False)
        motion_norm = net.forward_decoder(tokens)[0].cpu().numpy()
    motion = motion_norm * std + mean
    return se2_place_full_body(motion, d["start"], mf)


def render(idx, real_joints, gen_joints, goal_xy, text, goal_err, out_path):
    import shutil
    import matplotlib
    matplotlib.use("Agg")
    ffmpeg_path = shutil.which("ffmpeg") or os.path.join(os.path.dirname(sys.executable), "ffmpeg")
    matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_path
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    T = min(real_joints.shape[0], gen_joints.shape[0])
    real_joints, gen_joints = real_joints[:T], gen_joints[:T]

    goal_pt = np.array([[goal_xy[0], goal_xy[1], 0.0]])
    both = np.concatenate([real_joints.reshape(-1, 3), gen_joints.reshape(-1, 3), goal_pt], axis=0)
    xs, ys, zs = both[:, 0], both[:, 1], both[:, 2]
    ctr = np.array([xs.mean(), ys.mean(), zs.mean()])
    r = max(xs.ptp(), ys.ptp(), zs.ptp(), 1e-6) / 2 * 1.15

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), subplot_kw={"projection": "3d"})
    lines = [[], []]
    for ax, label, joints in zip(axes, ["REAL", "GENERATED"], [real_joints, gen_joints]):
        ax.set_xlim(ctr[0] - r, ctr[0] + r)
        ax.set_ylim(ctr[1] - r, ctr[1] + r)
        ax.set_zlim(ctr[2] - r, ctr[2] + r)
        ax.set_box_aspect([1, 1, 1])
        ax.set_title(label)
        # fed-goal marker: a vertical pole + floor dot, static every frame
        ax.plot([goal_xy[0], goal_xy[0]], [goal_xy[1], goal_xy[1]], [0, 1.8],
                color="red", linestyle="--", linewidth=1.5)
        ax.scatter([goal_xy[0]], [goal_xy[1]], [0], color="red", s=60, marker="*")
        for chain in KINEMATIC_CHAIN:
            (ln,) = ax.plot([], [], [], marker="o", markersize=2, color="tab:blue")
            lines[0 if label == "REAL" else 1].append((ln, chain))
    fig.suptitle(f"clip {idx:05d}: \"{text}\"\ngoal_err={goal_err:.2f}m (red = fed goal)", fontsize=10)
    fig.tight_layout()

    def update(fi):
        for pos, ln_group in zip([real_joints, gen_joints], lines):
            p = pos[fi]
            for ln, chain in ln_group:
                ln.set_data(p[chain, 0], p[chain, 1])
                ln.set_3d_properties(p[chain, 2])
        return [ln for grp in lines for ln, _ in grp]

    writer = FFMpegWriter(fps=FPS)
    with writer.saving(fig, out_path, dpi=110):
        for fi in range(T):
            update(fi)
            writer.grab_frame()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-name", type=str, default="conditioned")
    ap.add_argument("--n-clips", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import json
    ckpt_dir = os.path.join(OUT_DIR, "checkpoints", args.ckpt_name)
    with open(os.path.join(ckpt_dir, "norm_stats.json")) as f:
        norm_stats = json.load(f)
    conditioned = norm_stats["conditioned"]
    xy_mean = np.array(norm_stats["xy_mean"], dtype=np.float32)
    xy_std = np.array(norm_stats["xy_std"], dtype=np.float32)

    mean = np.load(f"{T2M_GPT_ROOT}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy").astype(np.float32)
    std = np.load(f"{T2M_GPT_ROOT}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy").astype(np.float32)

    with open(os.path.join(TOKENS_DIR, "test.pkl"), "rb") as f:
        test_manifest = pickle.load(f)
    rng = np.random.RandomState(args.seed)
    idxs = rng.choice(len(test_manifest), size=args.n_clips, replace=False)

    net = load_vqvae(device=DEVICE)
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE, jit=False)
    clip_model.eval()

    trans_encoder = build_transformer(norm_stats["clip_dim"])
    ckpt = torch.load(os.path.join(ckpt_dir, "net_final.pth"), map_location="cpu")
    trans_encoder.load_state_dict(ckpt["trans"], strict=True)
    trans_encoder.eval()
    trans_encoder.to(DEVICE)

    out_dir = os.path.join(OUT_DIR, "renders", "videos")
    os.makedirs(out_dir, exist_ok=True)

    for i in idxs:
        d = test_manifest[i]
        real_joints = load_real_world_joints(d["index"], d["start"], mean, std)
        gen_joints = generate_world_joints(d, trans_encoder, net, clip_model, conditioned, xy_mean, xy_std, mean, std)
        gen_end = gen_joints[-1, 0, :2]
        goal_err = float(np.linalg.norm(gen_end - d["goal"]))
        out_path = os.path.join(out_dir, f"{args.ckpt_name}_{d['index']:05d}.mp4")
        render(d["index"], real_joints, gen_joints, d["goal"], d["text"], goal_err, out_path)
        print(f"wrote {out_path} (goal_err={goal_err:.2f}m)")


if __name__ == "__main__":
    main()
