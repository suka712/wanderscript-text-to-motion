#!/usr/bin/env python3
"""Render a report gallery covering every completed stage.

Produces many variants per figure so there is something to pick from, into
subdirectories named by build-order stage. Everything is regenerable; nothing
here is committed (renders are gitignored).

Deliberately includes UNFLATTERING cases -- worst-case reconstructions, failure
examples, the no-prefix seam -- because a gallery of only good samples is not
evidence. Several figures are the first visual inspection of results that have
so far only been reported as scalars.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
for p in ["src", "scripts/track1", "scripts/scene_probe", "scripts/track2"]:
    sys.path.insert(0, os.path.join(REPO_ROOT, p))

import motion_features as mf  # noqa: E402
from humanise_join import build_flat_join, get_record, compute_track2  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402
from se2_utils import se2_place_full_body, CANONICAL_YAW_OFFSET  # noqa: E402

T2M = os.environ.get("WANDER_T2M_GPT_ROOT")
HUMANISE = os.environ.get("WANDER_HUMANISE_ROOT")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CHAIN = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15],
         [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def skeleton(ax, p, color=None, alpha=1.0, lw=1.8, zup=False):
    """p: (22,3). zup=True means data is already Z-up world frame."""
    for i, ch in enumerate(CHAIN):
        c = color or COLORS[i]
        if zup:
            ax.plot(p[ch, 0], p[ch, 1], p[ch, 2], color=c, alpha=alpha, lw=lw, marker="o", ms=2)
        else:
            ax.plot(p[ch, 0], p[ch, 2], p[ch, 1], color=c, alpha=alpha, lw=lw, marker="o", ms=2)


def equalize(ax, pts, zup=False):
    a, b, c = (pts[..., 0], pts[..., 1], pts[..., 2]) if zup else (pts[..., 0], pts[..., 2], pts[..., 1])
    ctr = np.array([a.mean(), b.mean(), c.mean()])
    r = max(a.ptp(), b.ptp(), c.ptp(), 1e-6) / 2 * 1.15
    ax.set_xlim(ctr[0] - r, ctr[0] + r)
    ax.set_ylim(ctr[1] - r, ctr[1] + r)
    ax.set_zlim(ctr[2] - r, ctr[2] + r)
    ax.set_box_aspect([1, 1, 1])


def poses_strip(pos, out, title, n=6, zup=False):
    idx = np.linspace(0, pos.shape[0] - 1, n).astype(int)
    fig, axes = plt.subplots(1, n, figsize=(2.7 * n, 3.0), subplot_kw={"projection": "3d"})
    for ax, fi in zip(axes, idx):
        skeleton(ax, pos[fi], zup=zup)
        equalize(ax, pos, zup=zup)
        ax.set_title(f"f{fi}", fontsize=8)
        ax.tick_params(labelsize=5)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def load263(idx):
    cm = np.load(os.path.join(HUMANISE, "contact_motion", "motions", f"{idx:05d}.npy"))
    d, *_ = mf.humanise_positions_to_263(cm)
    T = (min(d.shape[0], 196) // 4) * 4
    return d[:T].astype(np.float32) if T >= 4 else None


# ---------------------------------------------------------------- stage 03
def stage_tokenizer(out_dir, n=8):
    """GT vs frozen vs finetuned reconstruction, per category, including the
    worst cases -- the 03 numbers are means and hide the tail."""
    os.makedirs(out_dir, exist_ok=True)
    frozen = load_vqvae(device=DEV); frozen.eval()
    ft = load_vqvae(ckpt_path=os.path.expanduser(
        "~/wander_data/motion_data/track2_checkpoints/net_iter020000.pth"), device=DEV); ft.eval()
    mean = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy").astype(np.float32)
    std = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy").astype(np.float32)

    flat = build_flat_join()
    by = {}
    for i, p in enumerate(flat):
        by.setdefault(p["action"], []).append(i)

    def recon(net, d):
        x = torch.from_numpy((d - mean) / std).unsqueeze(0).to(DEV)
        with torch.no_grad():
            y, _, _ = net(x)
        return y[0].cpu().numpy() * std + mean

    for action in ["lie", "sit", "stand up", "walk"]:
        picks = np.random.RandomState(0).choice(by.get(action, []), size=min(n, len(by.get(action, []))), replace=False)
        for k, idx in enumerate(picks):
            d = load263(int(idx))
            if d is None:
                continue
            g = mf.local_joint_positions(d)
            rf = mf.local_joint_positions(recon(frozen, d).astype(np.float32))
            rt = mf.local_joint_positions(recon(ft, d).astype(np.float32))
            T = min(len(g), len(rf), len(rt))
            ef = np.linalg.norm(g[:T] - rf[:T], axis=-1).mean() * 1000
            et = np.linalg.norm(g[:T] - rt[:T], axis=-1).mean() * 1000
            sel = np.linspace(0, T - 1, 5).astype(int)
            fig, axes = plt.subplots(3, 5, figsize=(15, 9), subplot_kw={"projection": "3d"})
            for row, (pos, lab) in enumerate([(g, "ground truth"), (rf, f"frozen  {ef:.0f}mm"),
                                              (rt, f"finetuned  {et:.0f}mm")]):
                for col, fi in enumerate(sel):
                    ax = axes[row, col]
                    skeleton(ax, pos[fi])
                    equalize(ax, pos)
                    ax.tick_params(labelsize=5)
                    if col == 0:
                        ax.set_ylabel(lab)
                    ax.set_title(f"{lab} · f{fi}" if col == 0 else f"f{fi}", fontsize=7)
            fig.suptitle(f"[03] {action} · clip {idx} · frozen {ef:.0f}mm -> finetuned {et:.0f}mm", fontsize=11)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"03_tokenizer_{action.replace(' ','')}_{k:02d}_clip{idx}.png"), dpi=100)
            plt.close(fig)
    print(f"  stage 03 -> {out_dir}")


# ---------------------------------------------------------------- stage 06
def stage_scene(out_dir, n=10):
    """What each scene representation actually sees: RGB render vs occupancy,
    cropped in the agent's frame. This is the figure behind the DINOv2 result."""
    os.makedirs(out_dir, exist_ok=True)
    from scene_probe import crop_agent_frame
    bev = os.path.expanduser("~/wander_data/bev_cache")
    flat = build_flat_join()
    by = {}
    for i, p in enumerate(flat):
        by.setdefault(p["action"], []).append(i)

    for action in ["lie", "sit", "walk", "stand up"]:
        picks = np.random.RandomState(1).choice(by.get(action, []), size=min(n, len(by.get(action, []))), replace=False)
        shown = 0
        for idx in picks:
            rec = get_record(int(idx))
            f = os.path.join(bev, f"{rec.scene}.npz")
            if not os.path.exists(f):
                continue
            z = np.load(f)
            rgb, occ, ext = z["rgb"], z["occ"].astype(np.float32), z["extent"]
            _, xy, _, sincos = compute_track2(rec)
            yaw = float(np.arctan2(sincos[0, 0], sincos[0, 1]))
            cr = crop_agent_frame(rgb, ext, xy[-1], yaw)
            co = crop_agent_frame(occ, ext, xy[-1], yaw)
            fig, axes = plt.subplots(1, 4, figsize=(15, 4))
            axes[0].imshow(rgb); axes[0].set_title(f"scene BEV rgb\n{rec.scene}", fontsize=8)
            axes[1].imshow(occ, cmap="gray_r"); axes[1].set_title("scene occupancy", fontsize=8)
            axes[2].imshow(cr); axes[2].set_title("rgb crop @ goal (agent frame)\nwhat DINOv2 sees", fontsize=8)
            axes[3].imshow(co, cmap="gray_r"); axes[3].set_title("occupancy crop @ goal\nwhat won the probe", fontsize=8)
            for a in axes:
                a.axis("off")
            fig.suptitle(f"[06] {action} · '{rec.utterance[:60]}' · goal object: {flat[int(idx)]['object_label']}", fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"06_scene_{action.replace(' ','')}_{shown:02d}_clip{idx}.png"), dpi=100)
            plt.close(fig)
            shown += 1
            if shown >= n:
                break
    print(f"  stage 06 -> {out_dir}")


# ---------------------------------------------------------------- stage 01
def stage_data(out_dir, n=10):
    """World-frame trajectory on the scene occupancy -- the two-track
    representation, and the placement that everything downstream relies on."""
    os.makedirs(out_dir, exist_ok=True)
    bev = os.path.expanduser("~/wander_data/bev_cache")
    flat = build_flat_join()
    picks = np.random.RandomState(2).choice(len(flat), size=n * 3, replace=False)
    shown = 0
    for idx in picks:
        rec = get_record(int(idx))
        f = os.path.join(bev, f"{rec.scene}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        occ, ext = z["occ"].astype(np.float32), z["extent"]
        rgb = z["rgb"]
        jw, xy, _, _ = compute_track2(rec)
        xmin, xmax, ymin, ymax = ext
        H, W = occ.shape
        col = (xy[:, 0] - xmin) / (xmax - xmin) * W
        row = (ymax - xy[:, 1]) / (ymax - ymin) * H
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        for ax, img, cmap, lab in [(axes[0], rgb, None, "rgb"), (axes[1], occ, "gray_r", "occupancy")]:
            ax.imshow(img, cmap=cmap)
            ax.plot(col, row, "-", color="#d62728", lw=2, label="world-frame root track")
            ax.plot(col[0], row[0], "o", color="#2ca02c", ms=9, label="start")
            ax.plot(col[-1], row[-1], "*", color="#ff7f0e", ms=15, label="goal")
            ax.set_title(lab, fontsize=9); ax.axis("off"); ax.legend(fontsize=7)
        fig.suptitle(f"[01] {rec.scene} · {flat[int(idx)]['action']} · '{rec.utterance[:70]}'", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"01_worldtrack_{shown:02d}_clip{idx}.png"), dpi=100)
        plt.close(fig)
        shown += 1
        if shown >= n:
            break
    print(f"  stage 01 -> {out_dir}")


# ---------------------------------------------------------------- charts
def stage_charts(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # 03 per-category MPJPE
    cats = ["H3D base", "walk", "stand up", "sit", "lie"]
    fro = [56.11, 50.20, 66.98, 69.55, 136.91]
    fin = [56.2, 34.1, 53.0, 47.6, 96.3]
    x = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.2, fro, 0.4, label="frozen", color="#9aa5b1")
    ax.bar(x + 0.2, fin, 0.4, label="finetuned", color="#1f77b4")
    for i, (a, b) in enumerate(zip(fro, fin)):
        ax.text(i + 0.2, b + 2, f"{(b-a)/a*100:+.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(cats); ax.set_ylabel("MPJPE (mm)")
    ax.set_title("[03] Tokenizer joint finetune — held-out reconstruction")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "chart_03_tokenizer_mpjpe.png"), dpi=130); plt.close(fig)

    # 04/05 goal error
    labels = ["stay at\nstart", "uncond", "abs\nframe", "rel frame\n(frozen tok)", "rel frame\n(finetuned)", "oracle"]
    vals = [0.627, 0.490, 0.515, 0.164, 0.132, 0.107]
    cols = ["#9aa5b1", "#9aa5b1", "#9aa5b1", "#1f77b4", "#1f77b4", "#2ca02c"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(labels, vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("goal error (m)")
    ax.set_title("[04/05] Grounding — goal error vs controls")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "chart_04_goal_error.png"), dpi=130); plt.close(fig)

    # 04 by displacement
    bins = ["0-.25", ".25-.5", ".5-1", "1-2", ">2"]
    model = [0.071, 0.098, 0.181, 0.286, 0.508]
    oracle = [0.049, 0.077, 0.159, 0.228, 0.385]
    null = [0.068, 0.344, 0.729, 1.471, 2.399]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(bins, null, "o--", color="#9aa5b1", label="null (stay)")
    ax.plot(bins, model, "o-", color="#1f77b4", label="model")
    ax.plot(bins, oracle, "o-", color="#2ca02c", label="oracle (tokenizer floor)")
    ax.set_xlabel("commanded displacement (m)"); ax.set_ylabel("goal error (m)")
    ax.set_title("[04] Goal error tracks the tokenizer floor at every range")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "chart_04_by_displacement.png"), dpi=130); plt.close(fig)

    # 06 scene probe
    arms = ["prior", "DINOv2\nViT-B/14", "rgb crop\n(no encoder)", "DINOv2\nViT-S/14", "occupancy\ncrop"]
    acc = [26.4, 34.3, 36.4, 37.6, 63.0]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(arms, acc, color=["#9aa5b1", "#d62728", "#9aa5b1", "#d62728", "#2ca02c"])
    for i, v in enumerate(acc):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("action classification accuracy (%)")
    ax.set_title("[06] Scene representation — DINOv2 does not beat its own unencoded input")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "chart_06_scene_probe.png"), dpi=130); plt.close(fig)

    # 07 seam
    labels = ["no-prefix\n(naive)", "continuation\n(recon prefix)", "continuation\n(exact prefix)",
              "oracle", "canonicalization\nfloor"]
    vals = [155.8, 86.0, 71.2, 70.9, 21.6]
    cols = ["#d62728", "#1f77b4", "#1f77b4", "#2ca02c", "#9aa5b1"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(labels, vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + 3, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_ylabel("seam error (mm)")
    ax.set_title("[07] Conditional continuation — seam error vs controls")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "chart_07_seam.png"), dpi=130); plt.close(fig)
    print(f"  charts -> {out_dir}")


# ---------------------------------------------------------------- stage 04/05
def stage_grounding(out_dir, n=12):
    """Same clip, same fed goal, generated by each model. This is the figure
    that shows grounding working -- and the one that would have exposed the
    90-degree placement bug immediately had it been made earlier."""
    import clip, json, pickle
    from train_probe import build_transformer, cond_extra
    from se2_utils import se2_place
    os.makedirs(out_dir, exist_ok=True)
    root = os.path.expanduser("~/wander_data/track1_probe")
    test = pickle.load(open(os.path.join(root, "tokens_finetuned", "test.pkl"), "rb"))
    net = load_vqvae(ckpt_path=os.path.expanduser(
        "~/wander_data/motion_data/track2_checkpoints/net_iter020000.pth"), device=DEV); net.eval()
    mean = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy").astype(np.float32)
    std = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy").astype(np.float32)
    cm_, _ = clip.load("ViT-B/32", device=DEV, jit=False); cm_.eval()

    models = {}
    for nm in ["unconditioned-ft", "conditioned-rel-ft"]:
        cd = os.path.join(root, "checkpoints", nm)
        ns = json.load(open(os.path.join(cd, "norm_stats.json")))
        tr = build_transformer(ns["clip_dim"])
        tr.load_state_dict(torch.load(os.path.join(cd, "net_final.pth"), map_location="cpu")["trans"], strict=True)
        models[nm] = (tr.eval().to(DEV), ns)

    rng = np.random.RandomState(3)
    disp = np.array([np.linalg.norm(d["goal"] - d["start"][:2]) for d in test])
    picks = list(rng.choice(np.where(disp > 1.0)[0], size=min(n, (disp > 1.0).sum()), replace=False))
    for k, i in enumerate(picks):
        d = test[int(i)]
        fig, ax = plt.subplots(figsize=(8.6, 6))
        with torch.no_grad():
            feat = cm_.encode_text(clip.tokenize([d["text"]], truncate=True).to(DEV)).float()
            gt = torch.from_numpy(d["tokens"]).long().unsqueeze(0).to(DEV)
            w = se2_place(net.forward_decoder(gt)[0].cpu().numpy() * std + mean, d["start"], mf)
            ax.plot(w[:, 0], w[:, 1], "-", color="#2ca02c", lw=2.5, label="ground truth")
            for nm, col in [("unconditioned-ft", "#9aa5b1"), ("conditioned-rel-ft", "#1f77b4")]:
                tr, ns = models[nm]
                if ns["conditioned"]:
                    ex = torch.from_numpy(cond_extra(d, ns["cond_mode"],
                        np.array(ns["cond_mean"], np.float32), np.array(ns["cond_std"], np.float32))).unsqueeze(0).to(DEV)
                    cond = torch.cat([feat, ex], -1)
                else:
                    cond = feat
                tk = tr.sample(cond, if_categorial=False)
                if tk.numel() == 0:
                    continue
                wp = se2_place(net.forward_decoder(tk)[0].cpu().numpy() * std + mean, d["start"], mf)
                e = np.linalg.norm(wp[-1] - d["goal"])
                ax.plot(wp[:, 0], wp[:, 1], "-o", color=col, ms=2.5, lw=2,
                        label=f"{nm}  err={e:.2f}m")
        ax.plot(*d["start"][:2], "o", color="#2ca02c", ms=13, label="fed start")
        ax.plot(*d["goal"], "*", color="#d62728", ms=20, label="fed GOAL")
        ax.set_aspect("equal"); ax.grid(alpha=.3)
        ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        ax.set_title(f"[04/05] '{d['text'][:62]}'\ncommanded displacement {disp[i]:.2f} m", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"04_goal_{k:02d}_clip{d['index']}.png"), dpi=110)
        plt.close(fig)
    print(f"  stage 04/05 -> {out_dir}")


# ---------------------------------------------------------------- stage 07
def stage_seam(out_dir, n=12):
    """The seam itself: previous segment's ending pose (green) overlaid with the
    generated first pose, no-prefix vs continuation. This is what 'the body
    teleports' looks like."""
    import clip, json, pickle
    from train_probe import build_transformer, cond_extra
    os.makedirs(out_dir, exist_ok=True)
    root = os.path.expanduser("~/wander_data/continuation")
    test = pickle.load(open(os.path.join(root, "tokens", "test.pkl"), "rb"))
    net = load_vqvae(ckpt_path=os.path.expanduser(
        "~/wander_data/motion_data/track2_checkpoints/net_iter020000.pth"), device=DEV); net.eval()
    mean = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy").astype(np.float32)
    std = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy").astype(np.float32)
    cm_, _ = clip.load("ViT-B/32", device=DEV, jit=False); cm_.eval()

    models = {}
    for nm in ["noprefix", "continuation"]:
        cd = os.path.join(root, "checkpoints", nm)
        ns = json.load(open(os.path.join(cd, "norm_stats.json")))
        tr = build_transformer(ns["clip_dim"])
        tr.load_state_dict(torch.load(os.path.join(cd, "net_final.pth"), map_location="cpu")["trans"], strict=True)
        models[nm] = (tr.eval().to(DEV), ns)

    picks = np.random.RandomState(4).choice(len(test), size=n, replace=False)
    for k, i in enumerate(picks):
        d = test[int(i)]
        prev = d["prefix_pose"].reshape(22, 3)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), subplot_kw={"projection": "3d"})
        axes[0].set_title("previous segment's ending pose\n(the target, shown black elsewhere)", fontsize=9)
        skeleton(axes[0], prev); equalize(axes[0], prev[None])
        with torch.no_grad():
            feat = cm_.encode_text(clip.tokenize([d["text"]], truncate=True).to(DEV)).float()
            for ax, nm in zip(axes[1:], ["noprefix", "continuation"]):
                tr, ns = models[nm]
                ex = torch.from_numpy(cond_extra(d, ns["cond_mode"],
                    np.array(ns["cond_mean"], np.float32), np.array(ns["cond_std"], np.float32))).unsqueeze(0).to(DEV)
                tk = tr.sample(torch.cat([feat, ex], -1), if_categorial=False)
                if tk.numel() == 0:
                    continue
                m = net.forward_decoder(tk)[0].cpu().numpy() * std + mean
                first = mf.local_joint_positions(m.astype(np.float32))[0]
                err = np.linalg.norm(first - prev, axis=-1).mean() * 1000
                skeleton(ax, prev, color="#000000", alpha=.30, lw=4)
                skeleton(ax, first)
                equalize(ax, np.stack([prev, first]))
                ax.set_title(f"{nm}\nseam {err:.0f}mm   (black ghost = target)", fontsize=9)
        for a in axes:
            a.tick_params(labelsize=5)
        fig.suptitle(f"[07] seam continuity · '{d['text'][:60]}'", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"07_seam_{k:02d}_clip{d['index']}.png"), dpi=110)
        plt.close(fig)
    print(f"  stage 07 -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", nargs="*", default=["charts", "data", "tokenizer", "scene"])
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if "charts" in args.stages:
        stage_charts(os.path.join(args.out, "00_charts"))
    if "data" in args.stages:
        stage_data(os.path.join(args.out, "01_data_pipeline"), args.n)
    if "tokenizer" in args.stages:
        stage_tokenizer(os.path.join(args.out, "03_tokenizer"), args.n)
    if "scene" in args.stages:
        stage_scene(os.path.join(args.out, "06_scene"), args.n)
    if "grounding" in args.stages:
        stage_grounding(os.path.join(args.out, "04_grounding"), args.n)
    if "seam" in args.stages:
        stage_seam(os.path.join(args.out, "07_continuation"), args.n)
    print("done")


if __name__ == "__main__":
    main()
