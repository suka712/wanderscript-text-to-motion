#!/usr/bin/env python3
"""
Track 2 deliverable: qualitative lie reconstruction renders, frozen vs
finetuned, on held-out HUMANISE-lie clips -- per 001_tokenizer_finetune.md's
"a few lie reconstruction renders (moderate vs broken)".

Picks a few held-out lie clips spanning the per-clip MPJPE range under the
FROZEN model (best/median/worst of a held-out sample), then renders GT vs
frozen-recon vs finetuned-recon stick figures side by side for each, plus
prints the per-clip MPJPE table (frozen vs finetuned) so the numeric and
visual evidence line up.
"""
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import motion_features as mf  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402
import eval_per_category_mpjpe as evalcat  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "scratch_outputs", "track2_lie_renders")
os.makedirs(OUT_DIR, exist_ok=True)

FROZEN_CKPT = evalcat.DEFAULT_CKPT
FINETUNED_CKPT = "/media/user/2tb/motion_data/track2_checkpoints/track2_joint_finetune_run1/net_iter020000.pth"


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
        xs, ys, zs = p[:, 0], p[:, 2], p[:, 1]
        ctr = np.array([xs.mean(), ys.mean(), zs.mean()])
        r = max(xs.ptp(), ys.ptp(), zs.ptp(), 1e-6) / 2 * 1.1
        ax.set_xlim(ctr[0] - r, ctr[0] + r)
        ax.set_ylim(ctr[1] - r, ctr[1] + r)
        ax.set_zlim(ctr[2] - r, ctr[2] + r)
        ax.set_box_aspect([1, 1, 1])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def roundtrip(net, data263, mean, std):
    T = evalcat.crop_to_multiple(data263.shape[0])
    data263 = data263[:T].astype(np.float32)
    norm = (data263 - mean) / std
    x = torch.from_numpy(norm).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        x_out, _, _ = net(x)
    recon263 = x_out[0].cpu().numpy() * std + mean
    orig_local = mf.local_joint_positions(data263)
    recon_local = mf.local_joint_positions(recon263.astype(np.float32))
    mpjpe = np.linalg.norm(orig_local - recon_local, axis=-1).mean() * 1000
    orig_global = mf.recover_positions(data263)
    recon_global = mf.recover_positions(recon263.astype(np.float32))
    return mpjpe, orig_global, recon_global


def main():
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

    rng = np.random.RandomState(0)
    sample = rng.choice(lie_indices, size=min(60, len(lie_indices)), replace=False)

    frozen_errs = {}
    for i in sample:
        cm = np.load(f"{evalcat.HUMANISE_MOTIONS}/{i:05d}.npy")
        if cm.shape[0] < 6:
            continue
        data263, _, _, _ = mf.humanise_positions_to_263(cm)
        mpjpe, _, _ = roundtrip(frozen, data263, mean, std)
        frozen_errs[i] = mpjpe

    ranked = sorted(frozen_errs.items(), key=lambda kv: kv[1])
    picks = {
        "best_under_frozen": ranked[0],
        "median_under_frozen": ranked[len(ranked) // 2],
        "worst_under_frozen": ranked[-1],
    }

    print("\nPer-clip comparison (frozen vs finetuned MPJPE, mm):")
    print(f"{'clip':<8}{'role':<22}{'frozen':>10}{'finetuned':>12}{'delta':>10}")
    for role, (idx, frozen_mpjpe) in picks.items():
        cm = np.load(f"{evalcat.HUMANISE_MOTIONS}/{idx:05d}.npy")
        data263, _, _, _ = mf.humanise_positions_to_263(cm)

        f_mpjpe, f_orig, f_recon = roundtrip(frozen, data263, mean, std)
        ft_mpjpe, ft_orig, ft_recon = roundtrip(finetuned, data263, mean, std)
        delta = ft_mpjpe - f_mpjpe
        print(f"{idx:05d}   {role:<22}{f_mpjpe:>10.1f}{ft_mpjpe:>12.1f}{delta:>10.1f}")

        render_stick_figure(f_orig, f"{OUT_DIR}/{idx:05d}_{role}_GT.png",
                             f"HUMANISE lie {idx:05d} GT ({role})")
        render_stick_figure(f_recon, f"{OUT_DIR}/{idx:05d}_{role}_frozen_recon.png",
                             f"HUMANISE lie {idx:05d} FROZEN recon ({f_mpjpe:.1f}mm)")
        render_stick_figure(ft_recon, f"{OUT_DIR}/{idx:05d}_{role}_finetuned_recon.png",
                             f"HUMANISE lie {idx:05d} FINETUNED recon ({ft_mpjpe:.1f}mm)")

    print(f"\nRenders saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
