#!/usr/bin/env python3
"""Non-collision eval — the metric the scene-conditioning ablation actually needs.

WHY THIS EXISTS. eval_continuation.py measures seam and goal error, and on the
step-8 data BOTH are saturated: the models sit within ~1mm (seam) and ~4mm
(goal) of the ground-truth-token oracle, so there is no headroom left for an
ablation to show up in. More importantly those metrics are the wrong question.
Occupancy conditioning tells the model where the furniture is; goal error only
asks whether it reached a coordinate, and is perfectly happy with a path that
goes through a sofa. Collision is what a scene arm is FOR.

METRIC. Place the generated motion in the world with SE(2), map the root
trajectory to occupancy pixels with the renderer's own validated world->pixel
mapping, and report:
  collision rate  = fraction of frames whose root lands in occupied space
  collided clips  = fraction of clips with >=1 colliding frame

CONTROLS, on identical clips:
  ORACLE  the clip's own ground-truth tokens. Real human motion in a real
          scene, so this is NOT zero -- the occupancy raster is a height-sliced
          approximation and a real person's root passes near furniture. It is
          the floor, and a model at the floor is as clean as this metric can
          see. Reporting a raw collision rate without it would be meaningless.
  NULL    stay at the start pose for the same number of frames. Catches the
          degenerate win: a model that never moves never collides.
"""
import argparse
import json
import os
import pickle
import sys

import clip
import numpy as np
import torch

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "track1"))

import motion_features as mf  # noqa: E402
from humanise_join import get_record  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402
from se2_utils import se2_place  # noqa: E402
from train_probe import build_transformer, cond_extra  # noqa: E402

T2M = os.environ.get("WANDER_T2M_GPT_ROOT")
BEV = os.path.expanduser("~/wander_data/bev_cache")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def collide(world_xy, occ, ext):
    """Fraction of frames whose root lands on an occupied pixel."""
    xmin, xmax, ymin, ymax = ext
    H, W = occ.shape
    col = np.clip(((world_xy[:, 0] - xmin) / (xmax - xmin) * W).astype(int), 0, W - 1)
    row = np.clip(((ymax - world_xy[:, 1]) / (ymax - ymin) * H).astype(int), 0, H - 1)
    hits = occ[row, col] > 0.5
    return float(hits.mean()), bool(hits.any())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-name", nargs="+", required=True)
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--tokens-dir", required=True)
    ap.add_argument("--vqvae-ckpt", default=None)
    ap.add_argument("--n-clips", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    test = pickle.load(open(os.path.join(args.tokens_dir, "test.pkl"), "rb"))
    idxs = np.random.RandomState(args.seed).choice(
        len(test), size=min(args.n_clips, len(test)), replace=False)

    net = load_vqvae(ckpt_path=args.vqvae_ckpt, device=DEV) if args.vqvae_ckpt else load_vqvae(device=DEV)
    net.eval()
    mean = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy").astype(np.float32)
    std = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy").astype(np.float32)
    cm_, _ = clip.load("ViT-B/32", device=DEV, jit=False)
    cm_.eval()

    scenes = {}
    def scene_of(i):
        sid = get_record(int(test[i]["index"])).scene
        if sid not in scenes:
            if len(scenes) > 40:
                scenes.clear()
            z = np.load(os.path.join(BEV, f"{sid}.npz"))
            scenes[sid] = (z["occ"].astype(np.float32), z["extent"])
        return scenes[sid]

    def decode(tok):
        return net.forward_decoder(tok)[0].cpu().numpy() * std + mean

    report = {}
    for name in args.ckpt_name:
        cd = os.path.join(args.ckpt_root, name)
        ns = json.load(open(os.path.join(cd, "norm_stats.json")))
        cmean = np.array(ns["cond_mean"], np.float32)
        cstd = np.array(ns["cond_std"], np.float32)
        tr = build_transformer(ns["clip_dim"])
        tr.load_state_dict(torch.load(os.path.join(cd, "net_final.pth"), map_location="cpu")["trans"], strict=True)
        tr.eval().to(DEV)

        rates, any_hit = [], []
        with torch.no_grad():
            for n, i in enumerate(idxs):
                d = test[int(i)]
                occ, ext = scene_of(int(i))
                feat = cm_.encode_text(clip.tokenize([d["text"]], truncate=True).to(DEV)).float()
                ex = torch.from_numpy(cond_extra(d, ns["cond_mode"], cmean, cstd)).unsqueeze(0).to(DEV)
                tok = tr.sample(torch.cat([feat, ex], -1), if_categorial=False)
                if tok.numel() == 0:
                    continue
                w = se2_place(decode(tok), d["start"], mf)
                r, h = collide(w, occ, ext)
                rates.append(r); any_hit.append(h)
                if (n + 1) % 100 == 0:
                    print(f"  [{name}] {n+1}/{len(idxs)}", flush=True)
        report[f"{name} [{ns['cond_mode']}]"] = (np.mean(rates), np.mean(any_hit), len(rates))
        del tr
        torch.cuda.empty_cache()

    o_r, o_h, n_r, n_h = [], [], [], []
    with torch.no_grad():
        for i in idxs:
            d = test[int(i)]
            occ, ext = scene_of(int(i))
            tok = torch.from_numpy(d["tokens"]).long().unsqueeze(0).to(DEV)
            w = se2_place(decode(tok), d["start"], mf)
            r, h = collide(w, occ, ext)
            o_r.append(r); o_h.append(h)
            stay = np.repeat(d["start"][None, :2], len(w), axis=0)
            r2, h2 = collide(stay, occ, ext)
            n_r.append(r2); n_h.append(h2)
    report["ORACLE (GT tokens)"] = (np.mean(o_r), np.mean(o_h), len(o_r))
    report["NULL (stay at start)"] = (np.mean(n_r), np.mean(n_h), len(n_r))

    print(f"\n{'':<26}{'collision rate':>16}{'clips w/ any':>14}{'n':>6}")
    for k, (r, h, n) in report.items():
        print(f"{k:<26}{r*100:>15.2f}%{h*100:>13.1f}%{n:>6}")
    print("\ncollision rate = fraction of frames whose root is on an occupied pixel.\n"
          "ORACLE is real human motion, so it is NOT zero -- occupancy is a height-sliced\n"
          "approximation. A model at the ORACLE level is as clean as this metric can resolve.")


if __name__ == "__main__":
    main()
