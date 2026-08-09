#!/usr/bin/env python3
"""
Re-run Step 1's MPJPE reconstruction canary (check7_vqvae_canary.py) with the
evaluator-consistent normalization stats (checkpoints/t2m/
VQVAEV3_CB1024_CMT_H1024_NRES3/meta/{mean,std}.npy) instead of H3D_ROOT's
Mean.npy/Std.npy, and broken out per HUMANISE action category this time (the
original canary used one flat random HUMANISE sample, not a per-category
split). Purpose: determine whether the original 703mm 'lie' MPJPE number was
inflated by a normalization mismatch (same class of bug just found and fixed
in check12's FID pipeline), or whether it's real and the mean/std choice was
a red herring for MPJPE specifically (unlike for FID, since MPJPE is a
self-consistent round-trip metric -- see caveat in check12's fix commit).
"""
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import motion_features as mf  # noqa: E402
from humanise_join import build_flat_join  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402

T2M_GPT_ROOT = os.environ.get("WANDER_T2M_GPT_ROOT", "/home/user/Khiem-ssh/T2M-GPT")
H3D_ROOT = os.environ.get("WANDER_H3D_ROOT", "/media/user/2tb/motion_data/H3D")
HUMANISE_MOTIONS = os.environ.get("WANDER_HUMANISE_ROOT", "/media/user/2tb/motion_data/HUMANISE") + "/contact_motion/motions"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_T = 196
N_CLIPS = 200


def crop_to_multiple(T, factor=4, max_t=MAX_T):
    T = min(T, max_t)
    return (T // factor) * factor


def mpjpe(pos_a, pos_b):
    T = min(pos_a.shape[0], pos_b.shape[0])
    d = np.linalg.norm(pos_a[:T] - pos_b[:T], axis=-1)
    return d.mean()


def roundtrip_mpjpe(net, data263, mean, std):
    T = crop_to_multiple(data263.shape[0])
    if T < 4:
        return None
    data263 = data263[:T].astype(np.float32)
    norm = (data263 - mean) / std
    x = torch.from_numpy(norm).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        x_out, _, _ = net(x)
    recon263 = x_out[0].cpu().numpy() * std + mean
    orig_local = mf.local_joint_positions(data263)
    recon_local = mf.local_joint_positions(recon263.astype(np.float32))
    return mpjpe(orig_local, recon_local)


def category_clip_indices():
    flat = build_flat_join()
    by_action = {}
    for i, p in enumerate(flat):
        by_action.setdefault(p["action"], []).append(i)
    return by_action


def main():
    net = load_vqvae(device=DEVICE)
    mean = np.load(f"{T2M_GPT_ROOT}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy").astype(
        np.float32
    )
    std = np.load(f"{T2M_GPT_ROOT}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy").astype(
        np.float32
    )
    print(f"VQ-VAE loaded on {DEVICE}, using evaluator-consistent mean/std")

    # H3D baseline
    rng = np.random.RandomState(0)
    files = sorted(os.listdir(f"{H3D_ROOT}/new_joint_vecs"))
    files = [f for f in files if not f.startswith("M")]
    sample = rng.choice(files, size=min(N_CLIPS, len(files)), replace=False)
    h3d_errs = []
    for fn in sample:
        data263 = np.load(f"{H3D_ROOT}/new_joint_vecs/{fn}").astype(np.float32)
        e = roundtrip_mpjpe(net, data263, mean, std)
        if e is not None:
            h3d_errs.append(e)
    h3d_errs = np.array(h3d_errs)
    print(f"H3D baseline: MPJPE mean {h3d_errs.mean()*1000:.2f}mm  median {np.median(h3d_errs)*1000:.2f}mm  n={len(h3d_errs)}")

    # HUMANISE, per category
    by_action = category_clip_indices()
    results = {}
    for action in ["walk", "stand up", "sit", "lie"]:
        indices = by_action.get(action, [])
        rng = np.random.RandomState(1)
        chosen = rng.choice(indices, size=min(N_CLIPS, len(indices)), replace=False)
        errs = []
        for i in chosen:
            cm = np.load(f"{HUMANISE_MOTIONS}/{i:05d}.npy")
            if cm.shape[0] < 6:
                continue
            data263, _, _, _ = mf.humanise_positions_to_263(cm)
            e = roundtrip_mpjpe(net, data263.astype(np.float32), mean, std)
            if e is not None:
                errs.append(e)
        errs = np.array(errs)
        results[action] = errs
        ratio = errs.mean() / h3d_errs.mean()
        print(
            f"{action:<10} MPJPE mean {errs.mean()*1000:7.2f}mm  median {np.median(errs)*1000:7.2f}mm  "
            f"max {errs.max()*1000:7.2f}mm  n={len(errs):<4}  ratio-to-H3D {ratio:.2f}x"
        )

    print("\nComparison to Step 1's original (H3D_ROOT-normalized) canary numbers:")
    print("  H3D baseline was 137.3mm; HUMANISE walk ~112mm, stand-up ~192mm (1.4x), "
          "sit ~293mm (2.1x), lie ~703mm (5.1x, 'structurally broken').")


if __name__ == "__main__":
    main()
