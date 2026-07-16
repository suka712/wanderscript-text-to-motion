#!/usr/bin/env python3
"""
Side-by-side real vs VQ-VAE-reconstructed stick figures for one HUMANISE
'walk' clip and one 'lie' clip -- the visual arbiter for the walk/lie
recon-FID discrepancy in check12 (FID says lie reconstructs better than
walk; Step 1's MPJPE/visual canary said the opposite). Uses the same
evaluator-consistent mean/std as check12 (not H3D_ROOT's), so the VQ-VAE
input here matches exactly what check12's FID numbers were computed on.
"""
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
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "scratch_outputs", "walk_lie_compare")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_T = 196


def crop_to_multiple(T, factor=4, max_t=MAX_T):
    T = min(T, max_t)
    return (T // factor) * factor


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
    for row, pos, label in [(0, orig_pos, "REAL"), (1, recon_pos, "RECON")]:
        for col, fi in enumerate(idxs):
            ax = axes[row, col]
            p = pos[fi]
            for chain in kinematic_chain:
                ax.plot(p[chain, 0], p[chain, 2], p[chain, 1], marker="o", markersize=2)
            ax.set_title(f"{label} frame {fi}")
            ax.set_box_aspect([1, 1, 1])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
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

    rng = np.random.RandomState(42)
    for action in ["walk", "lie"]:
        # pick a clip with a reasonable number of frames (not a near-static one-frame clip)
        candidates = list(by_action[action])
        rng.shuffle(candidates)
        chosen = None
        for i in candidates[:50]:
            cm = np.load(f"{HUMANISE_MOTIONS}/{i:05d}.npy")
            if cm.shape[0] >= 40:
                chosen = i
                break
        if chosen is None:
            chosen = candidates[0]

        cm = np.load(f"{HUMANISE_MOTIONS}/{chosen:05d}.npy")
        data263, _, _, _ = mf.humanise_positions_to_263(cm)
        T = crop_to_multiple(data263.shape[0])
        data263 = data263[:T].astype(np.float32)
        norm = (data263 - mean) / std
        x = torch.from_numpy(norm).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            x_out, _, _ = net(x)
        recon_norm = x_out[0].cpu().numpy()
        recon263 = recon_norm * std + mean

        orig_pos = mf.recover_positions(data263)
        recon_pos = mf.recover_positions(recon263.astype(np.float32))

        out_path = f"{OUT_DIR}/{action.replace(' ', '_')}_{chosen:05d}_compare.png"
        render_pair(orig_pos, recon_pos, out_path, f"HUMANISE {action} clip {chosen:05d} (T={T})")
        print(f"{action}: clip {chosen:05d}, T={T} frames -> {out_path}")


if __name__ == "__main__":
    main()
