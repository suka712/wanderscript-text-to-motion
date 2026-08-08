#!/usr/bin/env python3
"""
Render 10 HUMANISE 'lie' clips (real vs recon, one representative frame
each), with the evaluator-consistent normalization (same as check14), to
determine whether the 3.09x-vs-H3D MPJPE gap is a broken subset or a
uniform moderate degradation across all lie clips.
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

T2M_GPT_ROOT = "/home/user/Khiem-ssh/T2M-GPT"
HUMANISE_MOTIONS = "/media/user/2tb/motion_data/HUMANISE/contact_motion/motions"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "scratch_outputs", "lie_ten_samples")
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_T = 196
N_SAMPLES = 10


def crop_to_multiple(T, factor=4, max_t=MAX_T):
    T = min(T, max_t)
    return (T // factor) * factor


def mpjpe(pos_a, pos_b):
    T = min(pos_a.shape[0], pos_b.shape[0])
    d = np.linalg.norm(pos_a[:T] - pos_b[:T], axis=-1)
    return d.mean()


def main():
    net = load_vqvae(device=DEVICE)
    mean = np.load(f"{T2M_GPT_ROOT}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy").astype(
        np.float32
    )
    std = np.load(f"{T2M_GPT_ROOT}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy").astype(
        np.float32
    )

    flat = build_flat_join()
    lie_indices = [i for i, p in enumerate(flat) if p["action"] == "lie"]
    rng = np.random.RandomState(7)
    chosen = rng.choice(lie_indices, size=N_SAMPLES, replace=False)

    kinematic_chain = [
        [0, 2, 5, 8, 11],
        [0, 1, 4, 7, 10],
        [0, 3, 6, 9, 12, 15],
        [9, 14, 17, 19, 21],
        [9, 13, 16, 18, 20],
    ]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(N_SAMPLES, 2, figsize=(6, 3 * N_SAMPLES), subplot_kw={"projection": "3d"})

    per_clip_mpjpe = []
    for row, i in enumerate(chosen):
        cm = np.load(f"{HUMANISE_MOTIONS}/{i:05d}.npy")
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
        per_clip_mpjpe.append((i, e, T))

        orig_pos = mf.recover_positions(data263)
        recon_pos = mf.recover_positions(recon263)
        mid = orig_pos.shape[0] // 2

        # shared equal-aspect limits across REAL/RECON, computed from their
        # union, so set_box_aspect doesn't distort either relative to the
        # other (see check7 for why box_aspect alone is not sufficient).
        both = np.concatenate([orig_pos[mid], recon_pos[mid]], axis=0)
        xs, ys, zs = both[:, 0], both[:, 2], both[:, 1]
        ctr = np.array([xs.mean(), ys.mean(), zs.mean()])
        r = max(xs.ptp(), ys.ptp(), zs.ptp(), 1e-6) / 2 * 1.1

        for col, (pos, label) in enumerate([(orig_pos, "REAL"), (recon_pos, "RECON")]):
            ax = axes[row, col]
            p = pos[mid]
            for chain in kinematic_chain:
                ax.plot(p[chain, 0], p[chain, 2], p[chain, 1], marker="o", markersize=2)
            ax.set_title(f"clip {i:05d} {label}\nMPJPE={e*1000:.0f}mm" if col == 0 else f"{label}")
            ax.set_xlim(ctr[0] - r, ctr[0] + r)
            ax.set_ylim(ctr[1] - r, ctr[1] + r)
            ax.set_zlim(ctr[2] - r, ctr[2] + r)
            ax.set_box_aspect([1, 1, 1])

    fig.suptitle("HUMANISE 'lie': 10 random clips, real vs recon (evaluator-consistent norm)")
    fig.tight_layout()
    out_path = f"{OUT_DIR}/lie_10_samples.png"
    fig.savefig(out_path, dpi=100)
    plt.close(fig)

    errs = np.array([e for _, e, _ in per_clip_mpjpe])
    print(f"Saved {out_path}")
    print(f"{'clip':<8}{'T':<6}{'MPJPE(mm)'}")
    for i, e, T in per_clip_mpjpe:
        print(f"{i:05d}   {T:<6}{e*1000:.1f}")
    print(f"\nmean={errs.mean()*1000:.1f}mm  median={np.median(errs)*1000:.1f}mm  "
          f"min={errs.min()*1000:.1f}mm  max={errs.max()*1000:.1f}mm  std={errs.std()*1000:.1f}mm")


if __name__ == "__main__":
    main()
