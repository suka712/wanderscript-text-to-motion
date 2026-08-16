#!/usr/bin/env python3
"""Render a chained multi-segment rollout — the demo video (done-criteria 4, 5).

Two synchronized views:
  left   top-down over the scene's occupancy, showing the whole path, the
         waypoints, and which segment is currently generating
  right  the 3D body, world-frame, walking that path

A 4-frame crossfade is applied at each seam FOR DISPLAY ONLY. Per RESULTS §7
the blended pose is never fed back as conditioning -- an off-manifold prefix
collapses the model -- so blending happens here, after the rollout, and the
rollout itself always hands the DECODED pose forward.

Segment boundaries are colour-coded so the seams are visible rather than
hidden: the point of the video is to show that the body does not teleport.
"""
import argparse
import os
import sys

import clip
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
import torch

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
for p in ["src", "scripts/track1", "scripts/chaining"]:
    sys.path.insert(0, os.path.join(REPO_ROOT, p))

import motion_features as mf  # noqa: E402
from humanise_join import build_flat_join, get_record, compute_track2  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402
from rollout import rollout, load_model, blend_seam  # noqa: E402
from demo_rollout import sample_waypoints  # noqa: E402

T2M = os.environ.get("WANDER_T2M_GPT_ROOT")
HUMANISE = os.environ.get("WANDER_HUMANISE_ROOT")
BEV = os.path.expanduser("~/wander_data/bev_cache")
TALL = os.path.expanduser("~/wander_data/bev_tall_cache")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CHAIN = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15],
         [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
SEGC = plt.get_cmap("viridis")
FPS = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vqvae-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-videos", type=int, default=6)
    ap.add_argument("--n-segments", type=int, default=10)
    ap.add_argument("--min-step", type=float, default=0.6)
    ap.add_argument("--max-step", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    net = load_vqvae(ckpt_path=args.vqvae_ckpt, device=DEV); net.eval()
    mean = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy").astype(np.float32)
    std = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy").astype(np.float32)
    cmodel, _ = clip.load("ViT-B/32", device=DEV, jit=False); cmodel.eval()
    trans, ns = load_model(args.ckpt)

    flat = build_flat_join()
    walk_idx = [i for i, p in enumerate(flat) if p["action"] == "walk"]
    rng = np.random.RandomState(args.seed)
    rng.shuffle(walk_idx)

    made = 0
    for idx in walk_idx:
        if made >= args.n_videos:
            break
        rec = get_record(int(idx))
        fb, ft = os.path.join(BEV, f"{rec.scene}.npz"), os.path.join(TALL, f"{rec.scene}.npz")
        if not (os.path.exists(fb) and os.path.exists(ft)):
            continue
        zb, zt = np.load(fb), np.load(ft)
        occ, extent = zb["occ"].astype(np.float32), zb["extent"]
        tall = zt["occ"].astype(np.float32)
        cm = np.load(os.path.join(HUMANISE, "contact_motion", "motions", f"{idx:05d}.npy"))
        try:
            d0, *_ = mf.humanise_positions_to_263(cm)
        except Exception:
            continue
        if d0.shape[0] < 8:
            continue
        _, xy, _, sincos = compute_track2(rec)
        start_pose = np.array([xy[0, 0], xy[0, 1], sincos[0, 0], sincos[0, 1]], np.float32)
        prefix = mf.local_joint_positions(d0.astype(np.float32))[0].ravel()
        wps = sample_waypoints(occ, extent, start_pose[:2], args.n_segments,
                               args.min_step, rng, max_step=args.max_step)
        if wps is None:
            continue
        segs = rollout(trans, net, cmodel, clip, mean, std, ns,
                       ["walk to the target"] * args.n_segments, wps,
                       start_pose, prefix, occ, extent)
        if len(segs) < args.n_segments:
            continue

        # display-only seam blend; conditioning already used the decoded pose
        worlds, prev = [], None
        for s in segs:
            w = blend_seam(prev, s["world"], n=4)
            worlds.append(w); prev = w
        allw = np.concatenate(worlds)
        seg_of = np.concatenate([[i] * len(w) for i, w in enumerate(worlds)])
        path = allw[:, 0, :2]
        xmin, xmax, ymin, ymax = extent
        H, W = tall.shape
        pc = np.clip(((path[:, 0] - xmin) / (xmax - xmin) * W).astype(int), 0, W - 1)
        pr = np.clip(((ymax - path[:, 1]) / (ymax - ymin) * H).astype(int), 0, H - 1)
        coll = float((tall[pr, pc] > .5).mean())
        dist = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))

        fig = plt.figure(figsize=(14, 6.2))
        axm = fig.add_subplot(1, 2, 1)
        axm.imshow(occ, cmap="gray_r")
        for i in range(len(worlds)):
            m = seg_of == i
            axm.plot(pc[m], pr[m], "-", lw=2.2, color=SEGC(i / max(1, len(worlds) - 1)))
        wr = [np.clip(int((ymax - w[1]) / (ymax - ymin) * H), 0, H - 1) for w in wps]
        wc = [np.clip(int((w[0] - xmin) / (xmax - xmin) * W), 0, W - 1) for w in wps]
        axm.plot(wc, wr, "*", color="#d62728", ms=13, ls="none", label="waypoints")
        axm.plot(pc[0], pr[0], "o", color="#2ca02c", ms=11, label="start")
        dot, = axm.plot([], [], "o", color="#111", ms=9)
        axm.set_title(f"{rec.scene} · {len(worlds)} chained segments · path {dist:.1f} m · "
                      f"collision {coll*100:.1f}%", fontsize=9)
        axm.axis("off"); axm.legend(fontsize=7, loc="lower right")

        ax3 = fig.add_subplot(1, 2, 2, projection="3d")
        a, b, c3 = allw[..., 0], allw[..., 1], allw[..., 2]
        cx, cy = (a.min() + a.max()) / 2, (b.min() + b.max()) / 2
        r = max(a.ptp(), b.ptp(), 1e-6) / 2 + 0.4
        ax3.set_xlim(cx - r, cx + r); ax3.set_ylim(cy - r, cy + r)
        ax3.set_zlim(min(c3.min(), 0) - .05, max(c3.max(), 1.8))
        ax3.set_box_aspect([1, 1, .8]); ax3.tick_params(labelsize=5)
        ax3.plot(path[:, 0], path[:, 1], np.zeros(len(path)), "-", color="#888", lw=1)
        lines = [ax3.plot([], [], [], lw=2.4, marker="o", ms=3)[0] for _ in CHAIN]
        fig.suptitle("chained rollout — each segment conditioned on the previous segment's "
                     "decoded ending pose", fontsize=11)
        fig.tight_layout()

        def update(f):
            p = allw[f]
            col = SEGC(seg_of[f] / max(1, len(worlds) - 1))
            for ln, ch in zip(lines, CHAIN):
                ln.set_data(p[ch, 0], p[ch, 1]); ln.set_3d_properties(p[ch, 2])
                ln.set_color(col)
            dot.set_data([pc[f]], [pr[f]])
            ax3.set_title(f"segment {seg_of[f]+1}/{len(worlds)}", fontsize=10)
            return []

        out = os.path.join(args.out, f"chain_{made:02d}_{rec.scene}.mp4")
        FuncAnimation(fig, update, frames=len(allw), interval=1000 / FPS).save(
            out, writer=FFMpegWriter(fps=FPS, bitrate=2600))
        plt.close(fig)
        made += 1
        print(f"  {made}/{args.n_videos}  {rec.scene}  {dist:.1f}m  coll {coll*100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
