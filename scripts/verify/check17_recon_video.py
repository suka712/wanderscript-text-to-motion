#!/usr/bin/env python3
"""
Full-length animated mp4 of REAL vs VQ-VAE-RECON stick figures, side by
side, for one clip per HUMANISE action category (default: all 4). Unlike
check13/16 (5 sampled frames as static PNGs), this plays every frame, which
is a much more direct check of whether the reconstruction looks right in
motion (e.g. limb jitter, foot sliding, timing drift) rather than just at
a few snapshots. Same evaluator-consistent mean/std as check12/13/14/15/16.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import motion_features as mf  # noqa: E402
from humanise_join import build_flat_join  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402

T2M_GPT_ROOT = os.environ.get("WANDER_T2M_GPT_ROOT", "/home/user/Khiem-ssh/T2M-GPT")
HUMANISE_MOTIONS = os.environ.get("WANDER_HUMANISE_ROOT", "/media/user/2tb/motion_data/HUMANISE") + "/contact_motion/motions"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "scratch_outputs", "recon_video")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_T = 196
FPS = 20  # HumanML3D's native frame rate convention

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


def mpjpe(pos_a, pos_b):
    T = min(pos_a.shape[0], pos_b.shape[0])
    return np.linalg.norm(pos_a[:T] - pos_b[:T], axis=-1).mean()


def pick_clips(by_action, action, rng, n, min_frames=40):
    candidates = list(by_action[action])
    rng.shuffle(candidates)
    chosen = []
    for i in candidates:
        cm = np.load(f"{HUMANISE_MOTIONS}/{i:05d}.npy")
        if cm.shape[0] >= min_frames:
            chosen.append(i)
        if len(chosen) == n:
            break
    if len(chosen) < n:
        chosen = candidates[:n]
    return chosen


def make_video(action, chosen, net, mean, std, out_dir=OUT_DIR):
    import shutil

    import matplotlib

    matplotlib.use("Agg")
    ffmpeg_path = shutil.which("ffmpeg") or os.path.join(
        os.path.dirname(sys.executable), "ffmpeg"
    )
    matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_path
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    cm = np.load(f"{HUMANISE_MOTIONS}/{chosen:05d}.npy")
    data263, _, _, _ = mf.humanise_positions_to_263(cm)
    T = crop_to_multiple(data263.shape[0])
    data263 = data263[:T].astype(np.float32)
    norm = (data263 - mean) / std
    x = torch.from_numpy(norm).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        x_out, _, _ = net(x)
    recon263 = (x_out[0].cpu().numpy() * std + mean).astype(np.float32)

    orig_local = mf.local_joint_positions(data263)
    recon_local = mf.local_joint_positions(recon263)
    e = mpjpe(orig_local, recon_local)

    orig_pos = mf.recover_positions(data263)
    recon_pos = mf.recover_positions(recon263)

    # fixed camera box for the whole clip (union over ALL frames), so the
    # figure doesn't jitter/rescale frame to frame -- same equal-aspect
    # principle as the static renders (check7/13/16), just clip-wide instead
    # of per-frame, which is what a video needs to look stable.
    both = np.concatenate([orig_pos, recon_pos], axis=0).reshape(-1, 3)
    xs, ys, zs = both[:, 0], both[:, 2], both[:, 1]
    ctr = np.array([xs.mean(), ys.mean(), zs.mean()])
    r = max(xs.ptp(), ys.ptp(), zs.ptp(), 1e-6) / 2 * 1.1

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), subplot_kw={"projection": "3d"})
    lines = [[], []]
    for ax, label in zip(axes, ["REAL", "RECON"]):
        ax.set_xlim(ctr[0] - r, ctr[0] + r)
        ax.set_ylim(ctr[1] - r, ctr[1] + r)
        ax.set_zlim(ctr[2] - r, ctr[2] + r)
        ax.set_box_aspect([1, 1, 1])
        ax.set_title(label)
        for chain in KINEMATIC_CHAIN:
            (ln,) = ax.plot([], [], [], marker="o", markersize=2)
            lines[0 if label == "REAL" else 1].append((ln, chain))
    fig.suptitle(f"HUMANISE {action} clip {chosen:05d} (T={T}, MPJPE={e*1000:.1f}mm)")
    fig.tight_layout()

    def update(fi):
        for pos, ln_group in zip([orig_pos, recon_pos], lines):
            p = pos[fi]
            for ln, chain in ln_group:
                ln.set_data(p[chain, 0], p[chain, 2])
                ln.set_3d_properties(p[chain, 1])
        return [ln for grp in lines for ln, _ in grp]

    writer = FFMpegWriter(fps=FPS)
    out_path = f"{out_dir}/{action.replace(' ', '_')}_{chosen:05d}.mp4"
    with writer.saving(fig, out_path, dpi=110):
        for fi in range(T):
            update(fi)
            writer.grab_frame()
    plt.close(fig)
    return out_path, e, T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", nargs="+", default=["walk", "stand up", "sit", "lie"])
    parser.add_argument("--n", type=int, default=1, help="videos per category")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    net = load_vqvae(device=DEVICE)
    mean = np.load(f"{T2M_GPT_ROOT}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy").astype(
        np.float32
    )
    std = np.load(f"{T2M_GPT_ROOT}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy").astype(
        np.float32
    )

    flat = build_flat_join()
    by_action = {}
    for i, p in enumerate(flat):
        by_action.setdefault(p["action"], []).append(i)

    rng = np.random.RandomState(args.seed)
    for action in args.actions:
        chosen_ids = pick_clips(by_action, action, rng, args.n)
        for chosen in chosen_ids:
            out_path, e, T = make_video(action, chosen, net, mean, std)
            print(f"{action:<10} clip {chosen:05d}  T={T:<4} MPJPE={e*1000:6.1f}mm -> {out_path}")


if __name__ == "__main__":
    main()
