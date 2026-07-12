#!/usr/bin/env python3
"""
Task 2 -- Frozen VQ-VAE reconstruction canary, two stages (order matters):
  1. H3D (original HumanML3D 263-dim, native format) -- baseline sanity.
  2. Converted HUMANISE (via src/motion_features.py adapter) -- the real test.

Methodology (same for both stages, for a fair comparison):
  - Take a batch of clips' 263-dim features (H3D: read directly from
    new_joint_vecs; HUMANISE: extracted via the unmodified HumanML3D
    process_file, see src/motion_features.py).
  - Normalize with H3D's own Mean.npy / Std.npy (the frozen VQ-VAE was
    trained on H3D-normalized data -- this is the correct normalization for
    BOTH stages, since it defines what "in-distribution" means for the frozen
    codebook).
  - Crop each clip's T to a multiple of 4 (down_t=2 => x4 temporal downsample)
    and clip to a max window (avoids one huge outlier clip dominating runtime).
  - Encode -> quantize -> decode through the frozen VQ-VAE.
  - Un-normalize, recover_from_ric() (the vendored INVERSE function) on both
    the original and reconstructed 263-dim vectors to get joint positions.
  - MPJPE = mean per-joint L2 position error (meters) between original and
    reconstructed joint positions, averaged over all frames/joints/clips.

One rendered sample (simple multi-pose stick-figure PNG) is saved per stage.
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
HUMANISE_MOTIONS = "/media/user/2tb/motion_data/HUMANISE/contact_motion/motions"
OUT_DIR = "/home/user/Khiem-ssh/wander/scratch_outputs/canary"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_T = 196  # HumanML3D's own max clip length convention
N_CLIPS = 200


def crop_to_multiple(T, factor=4, max_t=MAX_T):
    T = min(T, max_t)
    return (T // factor) * factor


def mpjpe(pos_a, pos_b):
    # pos_*: (T, 22, 3)
    T = min(pos_a.shape[0], pos_b.shape[0])
    d = np.linalg.norm(pos_a[:T] - pos_b[:T], axis=-1)  # (T, 22)
    return d.mean(), d


def run_vqvae_roundtrip(net, data263: np.ndarray, mean, std):
    """data263: (T, 263) raw (unnormalized).
    Returns (orig_local_pos, recon_local_pos, orig_global_pos, recon_global_pos).
    *_local_pos (used for the MPJPE metric) are root-relative, heading-
    canonicalized per-frame positions (src/motion_features.local_joint_positions)
    -- no cross-frame integration, so they measure per-frame pose fidelity
    without confounding it with the representation's cumulative root-drift.
    *_global_pos (used only for rendering) come from recover_from_ric, which
    DOES integrate root motion over the whole clip -- fine for a qualitative
    picture, not for a numeric fidelity metric on long clips.
    """
    T = crop_to_multiple(data263.shape[0])
    if T < 4:
        return None
    data263 = data263[:T]
    norm = (data263 - mean) / std
    x = torch.from_numpy(norm).float().unsqueeze(0).to(DEVICE)  # (1, T, 263)
    with torch.no_grad():
        x_out, _, _ = net(x)
    recon_norm = x_out[0].cpu().numpy()
    recon263 = recon_norm * std + mean

    orig_local = mf.local_joint_positions(data263.astype(np.float32))
    recon_local = mf.local_joint_positions(recon263.astype(np.float32))
    orig_global = mf.recover_positions(data263.astype(np.float32))
    recon_global = mf.recover_positions(recon263.astype(np.float32))
    return orig_local, recon_local, orig_global, recon_global


def render_stick_figure(pos, out_path, title, n_poses=5, kinematic_chain=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if kinematic_chain is None:
        kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15],
                            [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
    T = pos.shape[0]
    idxs = np.linspace(0, T - 1, n_poses).astype(int)
    fig, axes = plt.subplots(1, n_poses, figsize=(3 * n_poses, 3), subplot_kw={"projection": "3d"})
    if n_poses == 1:
        axes = [axes]
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


def stage_h3d(net, mean, std):
    print("=" * 60)
    print("STAGE 1: H3D (original HumanML3D) baseline")
    files = sorted(os.listdir(f"{H3D_ROOT}/new_joint_vecs"))
    files = [f for f in files if not f.startswith("M")]  # skip mirrored-augmentation copies
    rng = np.random.RandomState(0)
    sample = rng.choice(files, size=min(N_CLIPS, len(files)), replace=False)

    all_errs = []
    example = None
    for fn in sample:
        data263 = np.load(f"{H3D_ROOT}/new_joint_vecs/{fn}").astype(np.float32)
        result = run_vqvae_roundtrip(net, data263, mean, std)
        if result is None:
            continue
        orig_local, recon_local, orig_global, recon_global = result
        m, _ = mpjpe(orig_local, recon_local)
        all_errs.append(m)
        if example is None:
            example = (fn, orig_global, recon_global)

    all_errs = np.array(all_errs)
    print(f"Clips evaluated: {len(all_errs)}")
    print(f"MPJPE mean: {all_errs.mean()*1000:.2f} mm   median: {np.median(all_errs)*1000:.2f} mm   "
          f"max: {all_errs.max()*1000:.2f} mm")

    fn, orig_pos, recon_pos = example
    render_stick_figure(orig_pos, f"{OUT_DIR}/h3d_{fn}_orig.png", f"H3D {fn} ORIGINAL")
    render_stick_figure(recon_pos, f"{OUT_DIR}/h3d_{fn}_recon.png", f"H3D {fn} VQ-VAE RECON")
    print(f"Rendered sample: {fn} -> {OUT_DIR}/h3d_{fn}_{{orig,recon}}.png")
    return all_errs


def stage_humanise(net, mean, std):
    print("=" * 60)
    print("STAGE 2: HUMANISE (converted via motion_features adapter)")
    rng = np.random.RandomState(1)
    idxs = rng.choice(19648, size=N_CLIPS, replace=False)

    all_errs = []
    example = None
    n_skipped = 0
    for i in idxs:
        cm = np.load(f"{HUMANISE_MOTIONS}/{i:05d}.npy")
        if cm.shape[0] < 6:
            n_skipped += 1
            continue
        data263, _, _, _ = mf.humanise_positions_to_263(cm)
        result = run_vqvae_roundtrip(net, data263.astype(np.float32), mean, std)
        if result is None:
            n_skipped += 1
            continue
        orig_local, recon_local, orig_global, recon_global = result
        m, _ = mpjpe(orig_local, recon_local)
        all_errs.append(m)
        if example is None:
            example = (i, orig_global, recon_global)

    all_errs = np.array(all_errs)
    print(f"Clips evaluated: {len(all_errs)}  (skipped too-short: {n_skipped})")
    print(f"MPJPE mean: {all_errs.mean()*1000:.2f} mm   median: {np.median(all_errs)*1000:.2f} mm   "
          f"max: {all_errs.max()*1000:.2f} mm")

    i, orig_pos, recon_pos = example
    render_stick_figure(orig_pos, f"{OUT_DIR}/humanise_{i:05d}_orig.png", f"HUMANISE {i:05d} ORIGINAL")
    render_stick_figure(recon_pos, f"{OUT_DIR}/humanise_{i:05d}_recon.png", f"HUMANISE {i:05d} VQ-VAE RECON")
    print(f"Rendered sample: {i:05d} -> {OUT_DIR}/humanise_{i:05d}_{{orig,recon}}.png")
    return all_errs


if __name__ == "__main__":
    mean = np.load(f"{H3D_ROOT}/Mean.npy").astype(np.float32)
    std = np.load(f"{H3D_ROOT}/Std.npy").astype(np.float32)

    net = load_vqvae(device=DEVICE)
    print(f"VQ-VAE loaded on {DEVICE}")

    h3d_errs = stage_h3d(net, mean, std)
    humanise_errs = stage_humanise(net, mean, std)

    print("=" * 60)
    print("SUMMARY")
    print(f"H3D       MPJPE mean: {h3d_errs.mean()*1000:.2f} mm  (n={len(h3d_errs)})")
    print(f"HUMANISE  MPJPE mean: {humanise_errs.mean()*1000:.2f} mm  (n={len(humanise_errs)})")
    ratio = humanise_errs.mean() / h3d_errs.mean()
    print(f"Ratio HUMANISE/H3D: {ratio:.2f}x")
