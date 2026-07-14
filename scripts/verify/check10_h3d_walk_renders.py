#!/usr/bin/env python3
"""
STEP2 Task 2: render 2-3 H3D "walk" reconstructions (original vs. frozen
VQ-VAE recon) for visual inspection, alongside the recon-FID number from
VQ_eval.py. Reuses the same roundtrip/render helpers as
scripts/verify/check7_vqvae_canary.py (Step1b's canary), just narrowed to
clips whose text description contains "walk" and n=3 instead of a random
H3D sample.
"""
import os
import sys
import warnings

import numpy as np
import torch

sys.path.insert(0, "/home/user/Khiem-ssh/wander/src")
warnings.filterwarnings("ignore")

import motion_features as mf  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402

H3D_ROOT = "/media/user/2tb/motion_data/H3D"
OUT_DIR = "/home/user/Khiem-ssh/wander/scratch_outputs/step2_recon"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_T = 196
N_WALK = 3


def crop_to_multiple(T, factor=4, max_t=MAX_T):
    T = min(T, max_t)
    return (T // factor) * factor


def mpjpe(pos_a, pos_b):
    T = min(pos_a.shape[0], pos_b.shape[0])
    d = np.linalg.norm(pos_a[:T] - pos_b[:T], axis=-1)
    return d.mean()


def run_vqvae_roundtrip(net, data263, mean, std):
    T = crop_to_multiple(data263.shape[0])
    if T < 4:
        return None
    data263 = data263[:T]
    norm = (data263 - mean) / std
    x = torch.from_numpy(norm).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        x_out, _, _ = net(x)
    recon_norm = x_out[0].cpu().numpy()
    recon263 = recon_norm * std + mean

    orig_local = mf.local_joint_positions(data263.astype(np.float32))
    recon_local = mf.local_joint_positions(recon263.astype(np.float32))
    orig_global = mf.recover_positions(data263.astype(np.float32))
    recon_global = mf.recover_positions(recon263.astype(np.float32))
    return orig_local, recon_local, orig_global, recon_global


def render_stick_figure(pos, out_path, title, n_poses=5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15],
                        [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
    T = pos.shape[0]
    idxs = np.linspace(0, T - 1, n_poses).astype(int)
    fig, axes = plt.subplots(1, n_poses, figsize=(3 * n_poses, 3), subplot_kw={"projection": "3d"})
    for ax, fi in zip(axes, idxs):
        p = pos[fi]
        for chain in kinematic_chain:
            ax.plot(p[chain, 0], p[chain, 2], p[chain, 1], marker="o", markersize=2)
        ax.set_title(f"frame {fi}")
        ax.set_box_aspect([1, 1, 1])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    mean = np.load(f"{H3D_ROOT}/Mean.npy").astype(np.float32)
    std = np.load(f"{H3D_ROOT}/Std.npy").astype(np.float32)
    net = load_vqvae(device=DEVICE)
    print(f"VQ-VAE loaded on {DEVICE}")

    with open(f"{H3D_ROOT}/test.txt") as f:
        test_ids = [l.strip() for l in f if l.strip()]

    walk_ids = []
    for name in test_ids:
        if name.startswith("M"):
            continue  # skip mirrored-augmentation copies for a clean visual set
        text_path = f"{H3D_ROOT}/texts/{name}.txt"
        if not os.path.exists(text_path):
            continue
        with open(text_path) as tf:
            text = tf.read().lower()
        if "walk" in text:
            walk_ids.append(name)

    print(f"Found {len(walk_ids)} H3D test clips with 'walk' in the text description")
    rng = np.random.RandomState(2)
    chosen = rng.choice(walk_ids, size=min(N_WALK, len(walk_ids)), replace=False)

    for fn in chosen:
        data263 = np.load(f"{H3D_ROOT}/new_joint_vecs/{fn}.npy").astype(np.float32)
        result = run_vqvae_roundtrip(net, data263, mean, std)
        if result is None:
            print(f"  {fn}: too short, skipped")
            continue
        orig_local, recon_local, orig_global, recon_global = result
        err = mpjpe(orig_local, recon_local)
        with open(f"{H3D_ROOT}/texts/{fn}.txt") as tf:
            caption = tf.readline().split("#")[0]
        print(f"  {fn}: MPJPE={err*1000:.1f}mm  text=\"{caption.strip()}\"")
        render_stick_figure(orig_global, f"{OUT_DIR}/{fn}_orig.png", f"H3D {fn} ORIGINAL (walk)")
        render_stick_figure(recon_global, f"{OUT_DIR}/{fn}_recon.png", f"H3D {fn} VQ-VAE RECON (walk)")

    print(f"\nRendered {len(chosen)} walk reconstructions to {OUT_DIR}/ (gitignored)")
