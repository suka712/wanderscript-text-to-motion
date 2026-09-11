#!/usr/bin/env python3
"""Path B / Stage 1 figure: the LEARNED scene-aware model steers around an obstacle that the
scene-BLIND baseline walks through -- both GREEDY, no A* planner, no guided_seg selector.

This is the visual counterpart to the occ-ablation number (memory path-b-scene-aware-plan): the
step-10 goalaug model (scene-blind, RESULTS §9/§13: greedy collides >= a straight line) vs the
full-build synth model sf2k (trained on scene-CAUSED curved walks). Both decode greedily on the
SAME held-out scene / waypoints / start, so the only difference is what the network learned. Left
= pre (clips the salmon tall-obstacle), right = sf2k (routes around it). Scans held-out scenes with
obstacle-biased waypoints (something to avoid) and renders the largest real pre->post improvement.

Distinct from scripts/chaining/render_cg_compare.py, which contrasts one model's greedy vs
guided_seg (an INFERENCE-time selector). Here BOTH arms are greedy: the avoidance is in the WEIGHTS.
"""
import argparse
import json
import os
import sys

import clip
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
for p in ["src", "scripts/track1", "scripts/chaining"]:
    sys.path.insert(0, os.path.join(REPO_ROOT, p))

import motion_features as mf  # noqa: E402
from humanise_join import build_flat_join, get_record, compute_track2  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402
from rollout import load_model  # noqa: E402
from collision_guided import run_chain, path_collision, straight_line_collision, T2M, HUMANISE, BEV, TALL, DEV  # noqa: E402
from distill_guided import sample_hard_waypoints  # noqa: E402
from render_cg_compare import draw_panel  # noqa: E402


def chain_coll(segs, tall, extent):
    return path_collision(np.concatenate([s["world"][:, 0, :2] for s in segs]), tall, extent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True, help="scene-blind baseline (step10/goalaug)")
    ap.add_argument("--post", required=True, help="scene-aware full-build model (sf_2000)")
    ap.add_argument("--vqvae-ckpt", required=True)
    ap.add_argument("--eval-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-scan", type=int, default=40)
    ap.add_argument("--n-segments", type=int, default=6)
    ap.add_argument("--min-step", type=float, default=0.6)
    ap.add_argument("--max-step", type=float, default=1.2)
    ap.add_argument("--hard-frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--scene", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    net = load_vqvae(ckpt_path=args.vqvae_ckpt, device=DEV); net.eval()
    meta = f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta"
    mean = np.load(f"{meta}/mean.npy").astype(np.float32)
    std = np.load(f"{meta}/std.npy").astype(np.float32)
    cmodel, _ = clip.load("ViT-B/32", device=DEV, jit=False); cmodel.eval()
    pre, ns_pre = load_model(args.pre)
    post, ns_post = load_model(args.post)

    eval_scenes = set(json.load(open(args.eval_json))["eval_scenes"])
    flat = build_flat_join()
    walk_idx = [i for i, p in enumerate(flat) if p["action"] == "walk"]
    rng = np.random.RandomState(args.seed); rng.shuffle(walk_idx)

    best, scanned = None, 0
    for idx in walk_idx:
        if scanned >= args.n_scan and best is not None:
            break
        rec = get_record(int(idx))
        if rec.scene not in eval_scenes or (args.scene and rec.scene != args.scene):
            continue
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
        wps = sample_hard_waypoints(occ, tall, extent, start_pose[:2], args.n_segments,
                                    args.min_step, args.max_step, rng, hard_frac=args.hard_frac)
        if wps is None:
            continue
        texts = ["walk to the target"] * args.n_segments
        gp = run_chain(pre, net, cmodel, mean, std, ns_pre, texts, wps, start_pose, prefix,
                       occ, extent, tall, True, "greedy", 1, 0.0, rng)
        gq = run_chain(post, net, cmodel, mean, std, ns_post, texts, wps, start_pose, prefix,
                       occ, extent, tall, True, "greedy", 1, 0.0, rng)
        if len(gp) < args.n_segments or len(gq) < args.n_segments:
            continue
        pc, qc = chain_coll(gp, tall, extent), chain_coll(gq, tall, extent)
        scanned += 1
        print(f"  scan {scanned} {rec.scene} pre={pc*100:.1f}% post={qc*100:.1f}% "
              f"(improve {(pc-qc)*100:.1f}pp)", flush=True)
        if best is None or (pc - qc) > best["impr"]:
            best = dict(impr=pc - qc, rec=rec, occ=occ, tall=tall, extent=extent,
                        gp=gp, gq=gq, pc=pc, qc=qc, wps=wps, start=start_pose[:2].copy())
        if args.scene:
            break

    if best is None:
        print("nothing rendered"); return
    line = straight_line_collision(best["start"], best["wps"], best["tall"], best["extent"])
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.4))
    draw_panel(ax[0], best["occ"], best["tall"], best["extent"], best["gp"], best["wps"],
               best["start"], f"scene-BLIND baseline (goalaug) — collision {best['pc']*100:.1f}%")
    draw_panel(ax[1], best["occ"], best["tall"], best["extent"], best["gq"], best["wps"],
               best["start"], f"scene-AWARE (learned, sf2k) — collision {best['qc']*100:.1f}%")
    fig.suptitle(f"{best['rec'].scene}: the LEARNED model steers around tall obstacles (salmon) — "
                 f"both greedy, no planner/selector  ·  straight-line {line*100:.1f}%", fontsize=12)
    fig.tight_layout()
    p = os.path.join(args.out, f"scene_aware_{best['rec'].scene}.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"\nrendered {p}\n  pre {best['pc']*100:.1f}%  post {best['qc']*100:.1f}%  line {line*100:.1f}%")


if __name__ == "__main__":
    main()
