#!/usr/bin/env python3
"""
Side-by-side real vs VQ-VAE-reconstructed stick figures across ALL FOUR
HUMANISE action categories (walk, stand up, sit, lie), 2 clips each --
extends check13 (which only covered walk + lie) so there's visual evidence
for the interaction categories check13 didn't render. Same
evaluator-consistent mean/std as check12/13/14/15 (checkpoint's own meta
mean/std, not H3D_ROOT's -- see check14 for why that distinction matters).
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

T2M_GPT_ROOT = "/home/user/Khiem-ssh/T2M-GPT"
HUMANISE_MOTIONS = "/media/user/2tb/motion_data/HUMANISE/contact_motion/motions"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "scratch_outputs", "category_examples")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_T = 196
ACTIONS = ["walk", "stand up", "sit", "lie"]
N_PER_ACTION = 2


def crop_to_multiple(T, factor=4, max_t=MAX_T):
    T = min(T, max_t)
    return (T // factor) * factor


def mpjpe(pos_a, pos_b):
    T = min(pos_a.shape[0], pos_b.shape[0])
    return np.linalg.norm(pos_a[:T] - pos_b[:T], axis=-1).mean()


def render_pair(orig_pos, recon_pos, out_path, title, n_poses=5):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinematic_chain = [
        [0, 2, 5, 8, 11],
        [0, 1, 4, 7, 10],
        [0, 3, 6, 9, 12, 15],
        [9, 14, 17, 19, 21],
        [9, 13, 16, 18, 20],
    ]
    T = min(orig_pos.shape[0], recon_pos.shape[0])
    idxs = np.linspace(0, T - 1, n_poses).astype(int)

    fig, axes = plt.subplots(2, n_poses, figsize=(3 * n_poses, 6), subplot_kw={"projection": "3d"})
    for col, fi in enumerate(idxs):
        # shared equal-aspect limits across REAL/RECON at this frame, computed
        # from their union (see check7/check13 for why box_aspect alone isn't
        # sufficient -- unequal per-axis autoscaled ranges get stretched into
        # a cube and the figure looks anatomically distorted).
        both = np.concatenate([orig_pos[fi], recon_pos[fi]], axis=0)
        xs, ys, zs = both[:, 0], both[:, 2], both[:, 1]
        ctr = np.array([xs.mean(), ys.mean(), zs.mean()])
        r = max(xs.ptp(), ys.ptp(), zs.ptp(), 1e-6) / 2 * 1.1
        for row, pos, label in [(0, orig_pos, "REAL"), (1, recon_pos, "RECON")]:
            ax = axes[row, col]
            p = pos[fi]
            for chain in kinematic_chain:
                ax.plot(p[chain, 0], p[chain, 2], p[chain, 1], marker="o", markersize=2)
            ax.set_title(f"{label} frame {fi}")
            ax.set_xlim(ctr[0] - r, ctr[0] + r)
            ax.set_ylim(ctr[1] - r, ctr[1] + r)
            ax.set_zlim(ctr[2] - r, ctr[2] + r)
            ax.set_box_aspect([1, 1, 1])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_PER_ACTION, help="clips per category")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    n_per_action = args.n

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
    summary = []
    for action in ACTIONS:
        candidates = list(by_action[action])
        rng.shuffle(candidates)
        chosen_ids = []
        for i in candidates:
            cm = np.load(f"{HUMANISE_MOTIONS}/{i:05d}.npy")
            if cm.shape[0] >= 40:
                chosen_ids.append(i)
            if len(chosen_ids) == n_per_action:
                break
        if len(chosen_ids) < n_per_action:
            chosen_ids = candidates[:n_per_action]

        for chosen in chosen_ids:
            cm = np.load(f"{HUMANISE_MOTIONS}/{chosen:05d}.npy")
            data263, _, _, _ = mf.humanise_positions_to_263(cm)
            T = crop_to_multiple(data263.shape[0])
            data263 = data263[:T].astype(np.float32)
            norm = (data263 - mean) / std
            x = torch.from_numpy(norm).float().unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                x_out, _, _ = net(x)
            recon263 = x_out[0].cpu().numpy() * std + mean
            recon263 = recon263.astype(np.float32)

            orig_local = mf.local_joint_positions(data263)
            recon_local = mf.local_joint_positions(recon263)
            e = mpjpe(orig_local, recon_local)

            orig_pos = mf.recover_positions(data263)
            recon_pos = mf.recover_positions(recon263)

            slug = action.replace(" ", "_")
            out_path = f"{OUT_DIR}/{slug}_{chosen:05d}_compare.png"
            render_pair(orig_pos, recon_pos, out_path,
                        f"HUMANISE {action} clip {chosen:05d} (T={T})  MPJPE={e*1000:.1f}mm")
            print(f"{action:<10} clip {chosen:05d}  T={T:<4} MPJPE={e*1000:6.1f}mm -> {out_path}")
            summary.append((action, chosen, T, e))

    print("=" * 60)
    print("SUMMARY (evaluator-consistent norm)")
    for action in ACTIONS:
        errs = [e for a, _, _, e in summary if a == action]
        print(f"{action:<10} n={len(errs)}  mean MPJPE={np.mean(errs)*1000:.1f}mm")


if __name__ == "__main__":
    main()
