#!/usr/bin/env python3
"""Render report videos for every completed stage.

Three modes, each producing a batch of mp4s:
  grounding   world-frame motion over the scene, fed goal marked, conditioned
              vs unconditioned vs ground truth. The headline result.
  seam        segment A (real) then segment B (generated), animated through the
              join, no-prefix vs continuation. Shows the teleport, or its absence.
  tokenizer   ground truth / frozen / finetuned reconstruction, side by side.

All motion is placed with the CORRECTED SE(2) convention (yaw0 + pi/2). Any mp4
rendered before that fix shows trajectories rotated 90 degrees and should be
deleted rather than re-used.
"""
import argparse
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
import torch

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
for p in ["src", "scripts/track1", "scripts/scene_probe"]:
    sys.path.insert(0, os.path.join(REPO_ROOT, p))

import motion_features as mf  # noqa: E402
from humanise_join import build_flat_join, get_record, compute_track2  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402
from se2_utils import se2_place, se2_place_full_body  # noqa: E402

T2M = os.environ.get("WANDER_T2M_GPT_ROOT")
HUMANISE = os.environ.get("WANDER_HUMANISE_ROOT")
FT_CKPT = os.path.expanduser("~/wander_data/motion_data/track2_checkpoints/net_iter020000.pth")
BEV = os.path.expanduser("~/wander_data/bev_cache")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CHAIN = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15],
         [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
FPS = 20


def _norm():
    m = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy").astype(np.float32)
    s = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy").astype(np.float32)
    return m, s


def _lines(ax, zup=True):
    return [ax.plot([], [], [], color=c, lw=2.2, marker="o", ms=3)[0] for c in COLORS]


def _set(lines, p, zup=True):
    for ln, ch in zip(lines, CHAIN):
        if zup:
            ln.set_data(p[ch, 0], p[ch, 1]); ln.set_3d_properties(p[ch, 2])
        else:
            ln.set_data(p[ch, 0], p[ch, 2]); ln.set_3d_properties(p[ch, 1])


def _bounds(ax, pts, zup=True, pad=0.25):
    a, b, c = (pts[..., 0], pts[..., 1], pts[..., 2]) if zup else (pts[..., 0], pts[..., 2], pts[..., 1])
    cx, cy = (a.min() + a.max()) / 2, (b.min() + b.max()) / 2
    r = max(a.ptp(), b.ptp(), 1e-6) / 2 + pad
    ax.set_xlim(cx - r, cx + r); ax.set_ylim(cy - r, cy + r)
    ax.set_zlim(min(c.min(), 0) - 0.05, max(c.max(), 1.8))
    ax.set_box_aspect([1, 1, 0.85])


def save(fig, update, n_frames, out):
    ani = FuncAnimation(fig, update, frames=n_frames, interval=1000 / FPS, blit=False)
    ani.save(out, writer=FFMpegWriter(fps=FPS, bitrate=2400))
    plt.close(fig)


# ------------------------------------------------------------------ grounding
def do_grounding(out_dir, n):
    import clip
    from train_probe import build_transformer, cond_extra
    os.makedirs(out_dir, exist_ok=True)
    root = os.path.expanduser("~/wander_data/track1_probe")
    test = pickle.load(open(os.path.join(root, "tokens_finetuned", "test.pkl"), "rb"))
    net = load_vqvae(ckpt_path=FT_CKPT, device=DEV); net.eval()
    mean, std = _norm()
    cm_, _ = clip.load("ViT-B/32", device=DEV, jit=False); cm_.eval()

    models = {}
    for nm in ["unconditioned-ft", "conditioned-rel-ft"]:
        cd = os.path.join(root, "checkpoints", nm)
        ns = json.load(open(os.path.join(cd, "norm_stats.json")))
        tr = build_transformer(ns["clip_dim"])
        tr.load_state_dict(torch.load(os.path.join(cd, "net_final.pth"), map_location="cpu")["trans"], strict=True)
        models[nm] = (tr.eval().to(DEV), ns)

    disp = np.array([np.linalg.norm(d["goal"] - d["start"][:2]) for d in test])
    cand = np.where(disp > 1.2)[0]
    picks = np.random.RandomState(3).choice(cand, size=min(n, len(cand)), replace=False)

    for k, i in enumerate(picks):
        d = test[int(i)]
        panels = []
        with torch.no_grad():
            gt = torch.from_numpy(d["tokens"]).long().unsqueeze(0).to(DEV)
            panels.append(("ground truth",
                           se2_place_full_body(net.forward_decoder(gt)[0].cpu().numpy() * std + mean, d["start"], mf), None))
            feat = cm_.encode_text(clip.tokenize([d["text"]], truncate=True).to(DEV)).float()
            for nm in ["unconditioned-ft", "conditioned-rel-ft"]:
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
                m = net.forward_decoder(tk)[0].cpu().numpy() * std + mean
                j = se2_place_full_body(m, d["start"], mf)
                err = np.linalg.norm(se2_place(m, d["start"], mf)[-1] - d["goal"])
                panels.append((f"{'goal-conditioned' if ns['conditioned'] else 'unconditioned'}  err={err:.2f}m", j, err))
        if len(panels) < 3:
            continue
        allp = np.concatenate([p[1].reshape(-1, 3) for p in panels])
        nf = max(p[1].shape[0] for p in panels)
        fig = plt.figure(figsize=(15, 5.2))
        axes, lines, trails = [], [], []
        for c, (lab, j, _) in enumerate(panels):
            ax = fig.add_subplot(1, 3, c + 1, projection="3d")
            _bounds(ax, allp)
            ax.plot([d["goal"][0]], [d["goal"][1]], [0], "*", color="#d62728", ms=22)
            ax.plot([d["start"][0]], [d["start"][1]], [0], "o", color="#2ca02c", ms=10)
            tr_, = ax.plot([], [], [], "-", color="#444", lw=1.2, alpha=.8)
            ax.set_title(lab, fontsize=10)
            ax.tick_params(labelsize=5)
            axes.append(ax); lines.append(_lines(ax)); trails.append(tr_)
        fig.suptitle(f"'{d['text'][:70]}'   ★ = fed goal   ● = fed start", fontsize=11)
        fig.tight_layout()

        def update(f):
            for (lab, j, _), ls, tr_ in zip(panels, lines, trails):
                fi = min(f, j.shape[0] - 1)
                _set(ls, j[fi])
                tr_.set_data(j[:fi + 1, 0, 0], j[:fi + 1, 0, 1])
                tr_.set_3d_properties(np.zeros(fi + 1))
            return []
        save(fig, update, nf, os.path.join(out_dir, f"grounding_{k:02d}_clip{d['index']}.mp4"))
        print(f"  grounding {k+1}/{len(picks)}", flush=True)


# ----------------------------------------------------------------------- seam
def do_seam(out_dir, n):
    import clip
    from train_probe import build_transformer, cond_extra
    os.makedirs(out_dir, exist_ok=True)
    root = os.path.expanduser("~/wander_data/continuation")
    test = pickle.load(open(os.path.join(root, "tokens", "test.pkl"), "rb"))
    net = load_vqvae(ckpt_path=FT_CKPT, device=DEV); net.eval()
    mean, std = _norm()
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
        # real segment A, in world frame, ending at the seam
        cm = np.load(os.path.join(HUMANISE, "contact_motion", "motions", f"{d['index']:05d}.npy"))
        T_raw = cm.shape[0]
        t = (T_raw // 2 // 4) * 4
        rec = get_record(int(d["index"]))
        jw, xy, _, sincos = compute_track2(rec)
        A = jw[:t + 1]

        outs = []
        with torch.no_grad():
            feat = cm_.encode_text(clip.tokenize([d["text"]], truncate=True).to(DEV)).float()
            for nm in ["noprefix", "continuation"]:
                tr, ns = models[nm]
                ex = torch.from_numpy(cond_extra(d, ns["cond_mode"],
                    np.array(ns["cond_mean"], np.float32), np.array(ns["cond_std"], np.float32))).unsqueeze(0).to(DEV)
                tk = tr.sample(torch.cat([feat, ex], -1), if_categorial=False)
                if tk.numel() == 0:
                    continue
                m = net.forward_decoder(tk)[0].cpu().numpy() * std + mean
                B = se2_place_full_body(m, d["start"], mf)
                err = np.linalg.norm(mf.local_joint_positions(m.astype(np.float32))[0]
                                     - d["prefix_pose"].reshape(22, 3), axis=-1).mean() * 1000
                # lift B onto A's floor height so the two segments share a ground plane
                B = B.copy(); B[..., 2] += A[-1, :, 2].min() - B[0, :, 2].min()
                outs.append((nm, B, err))
        if len(outs) < 2:
            continue

        allp = np.concatenate([A.reshape(-1, 3)] + [o[1].reshape(-1, 3) for o in outs])
        nB = max(o[1].shape[0] for o in outs)
        nf = A.shape[0] + nB
        fig = plt.figure(figsize=(11, 5.4))
        axes, lines, trails = [], [], []
        for c, (nm, B, err) in enumerate(outs):
            ax = fig.add_subplot(1, 2, c + 1, projection="3d")
            _bounds(ax, allp)
            tr_, = ax.plot([], [], [], "-", color="#444", lw=1.2, alpha=.8)
            ax.set_title(f"{'no-prefix (naive)' if nm=='noprefix' else 'continuation'}\nseam {err:.0f}mm", fontsize=10)
            ax.tick_params(labelsize=5)
            axes.append(ax); lines.append(_lines(ax)); trails.append(tr_)
        fig.suptitle(f"segment A (real, faded) then segment B (GENERATED, solid) · '{d['text'][:55]}'", fontsize=10)
        fig.tight_layout()

        def update(f):
            for (nm, B, err), ls, tr_, ax in zip(outs, lines, trails, axes):
                if f < A.shape[0]:
                    p = A[f]
                    path = A[:f + 1, 0, :2]
                    for ln in ls:
                        ln.set_alpha(0.45)
                else:
                    fi = min(f - A.shape[0], B.shape[0] - 1)
                    p = B[fi]
                    path = np.concatenate([A[:, 0, :2], B[:fi + 1, 0, :2]])
                    for ln in ls:
                        ln.set_alpha(1.0)
                _set(ls, p)
                tr_.set_data(path[:, 0], path[:, 1])
                tr_.set_3d_properties(np.zeros(len(path)))
            return []
        save(fig, update, nf, os.path.join(out_dir, f"seam_{k:02d}_clip{d['index']}.mp4"))
        print(f"  seam {k+1}/{len(picks)}", flush=True)


# ------------------------------------------------------------------ tokenizer
def do_tokenizer(out_dir, n):
    os.makedirs(out_dir, exist_ok=True)
    frozen = load_vqvae(device=DEV); frozen.eval()
    ft = load_vqvae(ckpt_path=FT_CKPT, device=DEV); ft.eval()
    mean, std = _norm()
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
            cm = np.load(os.path.join(HUMANISE, "contact_motion", "motions", f"{int(idx):05d}.npy"))
            d, *_ = mf.humanise_positions_to_263(cm)
            T = (min(d.shape[0], 196) // 4) * 4
            if T < 8:
                continue
            d = d[:T].astype(np.float32)
            g = mf.local_joint_positions(d)
            rf = mf.local_joint_positions(recon(frozen, d).astype(np.float32))
            rt = mf.local_joint_positions(recon(ft, d).astype(np.float32))
            Tm = min(len(g), len(rf), len(rt))
            ef = np.linalg.norm(g[:Tm] - rf[:Tm], axis=-1).mean() * 1000
            et = np.linalg.norm(g[:Tm] - rt[:Tm], axis=-1).mean() * 1000
            panels = [("ground truth", g), (f"frozen  {ef:.0f}mm", rf), (f"finetuned  {et:.0f}mm", rt)]
            allp = np.concatenate([p[1].reshape(-1, 3) for p in panels])
            fig = plt.figure(figsize=(13, 4.6))
            lines = []
            for c, (lab, pos) in enumerate(panels):
                ax = fig.add_subplot(1, 3, c + 1, projection="3d")
                _bounds(ax, allp, zup=False)
                ax.set_title(lab, fontsize=10); ax.tick_params(labelsize=5)
                lines.append(_lines(ax))
            fig.suptitle(f"[03] {action} · clip {idx} · frozen {ef:.0f}mm -> finetuned {et:.0f}mm", fontsize=11)
            fig.tight_layout()

            def update(f):
                for (lab, pos), ls in zip(panels, lines):
                    _set(ls, pos[min(f, len(pos) - 1)], zup=False)
                return []
            save(fig, update, Tm, os.path.join(out_dir, f"tokenizer_{action.replace(' ','')}_{k:02d}_clip{idx}.mp4"))
        print(f"  tokenizer {action} done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", nargs="*", default=["grounding", "seam", "tokenizer"])
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    if "grounding" in args.modes:
        do_grounding(os.path.join(args.out, "04_grounding"), args.n)
    if "seam" in args.modes:
        do_seam(os.path.join(args.out, "07_continuation"), args.n)
    if "tokenizer" in args.modes:
        do_tokenizer(os.path.join(args.out, "03_tokenizer"), max(3, args.n // 2))
    print("done")


if __name__ == "__main__":
    main()
