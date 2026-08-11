#!/usr/bin/env python3
"""
Presentation asset: full-length animated mp4, GT / FROZEN recon / FINETUNED
recon side by side (3 panels), for held-out HUMANISE-lie clips. Every frame
played (not 5 static snapshots like render_lie_comparison.py's PNGs), so the
finetune's improvement is visible in motion, not just at a few poses --
same rationale as scripts/verify/check17_recon_video.py, extended to a 3-way
comparison and to lie clips specifically (Track 2's focus category).
"""
import argparse
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import motion_features as mf  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402
import eval_per_category_mpjpe as evalcat  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "scratch_outputs", "track2_lie_video")
os.makedirs(OUT_DIR, exist_ok=True)

FROZEN_CKPT = evalcat.DEFAULT_CKPT
FINETUNED_CKPT = "/media/user/2tb/motion_data/track2_checkpoints/track2_joint_finetune_run1/net_iter020000.pth"
FPS = 20  # HumanML3D native frame rate convention

KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]


def roundtrip_positions(net, data263, mean, std):
    T = evalcat.crop_to_multiple(data263.shape[0])
    data263c = data263[:T].astype(np.float32)
    norm = (data263c - mean) / std
    x = torch.from_numpy(norm).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        x_out, _, _ = net(x)
    recon263 = (x_out[0].cpu().numpy() * std + mean).astype(np.float32)
    orig_local = mf.local_joint_positions(data263c)
    recon_local = mf.local_joint_positions(recon263)
    e = np.linalg.norm(orig_local - recon_local, axis=-1).mean() * 1000
    return mf.recover_positions(recon263), e, T


def make_video(idx, gt263, frozen_net, finetuned_net, mean, std, out_dir=OUT_DIR):
    import matplotlib

    matplotlib.use("Agg")
    ffmpeg_path = shutil.which("ffmpeg") or os.path.join(os.path.dirname(sys.executable), "ffmpeg")
    matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_path
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    T = evalcat.crop_to_multiple(gt263.shape[0])
    gt263c = gt263[:T].astype(np.float32)
    gt_pos = mf.recover_positions(gt263c)

    frozen_pos, frozen_mpjpe, _ = roundtrip_positions(frozen_net, gt263, mean, std)
    finetuned_pos, finetuned_mpjpe, _ = roundtrip_positions(finetuned_net, gt263, mean, std)

    all_pos = [gt_pos, frozen_pos, finetuned_pos]
    labels = [
        "GT",
        f"FROZEN recon ({frozen_mpjpe:.1f}mm)",
        f"FINETUNED recon ({finetuned_mpjpe:.1f}mm)",
    ]

    both = np.concatenate(all_pos, axis=0).reshape(-1, 3)
    xs, ys, zs = both[:, 0], both[:, 2], both[:, 1]
    ctr = np.array([xs.mean(), ys.mean(), zs.mean()])
    r = max(xs.ptp(), ys.ptp(), zs.ptp(), 1e-6) / 2 * 1.1

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), subplot_kw={"projection": "3d"})
    lines = [[] for _ in range(3)]
    for ax, label, ln_group in zip(axes, labels, lines):
        ax.set_xlim(ctr[0] - r, ctr[0] + r)
        ax.set_ylim(ctr[1] - r, ctr[1] + r)
        ax.set_zlim(ctr[2] - r, ctr[2] + r)
        ax.set_box_aspect([1, 1, 1])
        ax.set_title(label, fontsize=10)
        for chain in KINEMATIC_CHAIN:
            (ln,) = ax.plot([], [], [], marker="o", markersize=2)
            ln_group.append((ln, chain))
    fig.suptitle(f"HUMANISE lie clip {idx:05d} (T={T}, held-out) -- frozen vs finetuned VQ-VAE")
    fig.tight_layout()

    def update(fi):
        for pos, ln_group in zip(all_pos, lines):
            p = pos[fi]
            for ln, chain in ln_group:
                ln.set_data(p[chain, 0], p[chain, 2])
                ln.set_3d_properties(p[chain, 1])
        return [ln for grp in lines for ln, _ in grp]

    writer = FFMpegWriter(fps=FPS)
    out_path = f"{out_dir}/lie_{idx:05d}_frozen_vs_finetuned.mp4"
    with writer.saving(fig, out_path, dpi=120):
        for fi in range(T):
            update(fi)
            writer.grab_frame()
    plt.close(fig)
    return out_path, frozen_mpjpe, finetuned_mpjpe, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="number of held-out lie clips to render")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-frames", type=int, default=60)
    args = ap.parse_args()

    mean = np.load(evalcat.EVAL_MEAN_PATH).astype(np.float32)
    std = np.load(evalcat.EVAL_STD_PATH).astype(np.float32)

    frozen = load_vqvae(ckpt_path=FROZEN_CKPT, device=DEVICE)
    frozen.eval()
    finetuned = load_vqvae(ckpt_path=FINETUNED_CKPT, device=DEVICE)
    finetuned.eval()
    print(f"Loaded frozen ({FROZEN_CKPT}) and finetuned ({FINETUNED_CKPT})")

    by_action = evalcat.humanise_category_test_indices()
    lie_indices = by_action.get("lie", [])
    print(f"Held-out HUMANISE-lie clips available: {len(lie_indices)}")

    rng = np.random.RandomState(args.seed)
    candidates = list(lie_indices)
    rng.shuffle(candidates)

    chosen = []
    for i in candidates:
        cm = np.load(f"{evalcat.HUMANISE_MOTIONS}/{i:05d}.npy")
        if cm.shape[0] >= args.min_frames:
            chosen.append(i)
        if len(chosen) == args.n:
            break

    for idx in chosen:
        cm = np.load(f"{evalcat.HUMANISE_MOTIONS}/{idx:05d}.npy")
        data263, _, _, _ = mf.humanise_positions_to_263(cm)
        out_path, f_mpjpe, ft_mpjpe, T = make_video(idx, data263, frozen, finetuned, mean, std)
        print(f"clip {idx:05d}  T={T:<4}  frozen={f_mpjpe:6.1f}mm  finetuned={ft_mpjpe:6.1f}mm  "
              f"delta={ft_mpjpe - f_mpjpe:+7.1f}mm  -> {out_path}")


if __name__ == "__main__":
    main()
