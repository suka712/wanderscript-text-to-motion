#!/usr/bin/env python3
"""Continuation probe eval (docs/07_continuation_probe.md).

THE QUESTION (CLAUDE.md 2d): segment k ends in some body configuration; segment
k+1 is generated from a canonical neutral pose. Gluing them makes the body
teleport at the seam. Does conditioning on the previous segment's ending pose
actually fix that?

PRIMARY METRIC -- seam error: mean per-joint distance between the previous
segment's ENDING local pose and the generated segment's FIRST local pose. Both
are root-relative and heading-canonicalized, so this is frame-independent and
measures exactly one thing: does the body configuration match across the seam.
It deliberately ignores root position and heading, which SE(2) placement matches
by construction and which therefore carry no information about teleporting.

CONTROLS, on identical clips:
  ORACLE   the clip's own ground-truth B tokens, decoded. The floor -- VQ-VAE
           reconstruction error alone. Nothing can beat this.
  NO-PREFIX a model trained with goal conditioning but NO prefix pose. This is
           the naive approach 2d says fails, and it is what the number has to
           be read against. Without it, a good-looking seam error means nothing.
Also reports goal error, to confirm continuation conditioning did not come at
the cost of the grounding that already works.
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
from vqvae_loader import load_vqvae  # noqa: E402
from se2_utils import se2_place  # noqa: E402
from train_probe import build_transformer, cond_extra  # noqa: E402

T2M_GPT_ROOT = os.environ.get("WANDER_T2M_GPT_ROOT", "/home/user/Khiem-ssh/T2M-GPT")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def summarize(v):
    v = np.asarray(v)
    return {"mean": float(v.mean()), "median": float(np.median(v)),
            "sem": float(v.std() / np.sqrt(len(v))), "n": int(len(v))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-name", nargs="+", required=True)
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--tokens-dir", required=True)
    ap.add_argument("--vqvae-ckpt", default=None)
    ap.add_argument("--n-clips", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefix-source", choices=["gt", "recon"], default="gt",
                    help="gt: the exact seam pose. recon: the seam pose after a VQ-VAE "
                         "round trip, which is what real chaining actually supplies -- "
                         "segment k's ending pose reaches k+1 through the decoder. Unlike "
                         "iid noise this error is STRUCTURED and anatomically plausible, "
                         "so it is the honest robustness test.")
    ap.add_argument("--prefix-noise-mm", type=float, nargs="*", default=[0.0],
                    help="perturb the conditioning seam pose by iid Gaussian noise of this "
                         "per-joint sigma (mm) before generating. In real chaining the seam "
                         "pose comes from GENERATED segment k, so it is not exact -- this "
                         "measures whether the mechanism survives that.")
    args = ap.parse_args()

    test = pickle.load(open(os.path.join(args.tokens_dir, "test.pkl"), "rb"))
    idxs = np.random.RandomState(args.seed).choice(
        len(test), size=min(args.n_clips, len(test)), replace=False)

    net = load_vqvae(ckpt_path=args.vqvae_ckpt, device=DEVICE) if args.vqvae_ckpt else load_vqvae(device=DEVICE)
    net.eval()
    mean = np.load(f"{T2M_GPT_ROOT}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy").astype(np.float32)
    std = np.load(f"{T2M_GPT_ROOT}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy").astype(np.float32)
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE, jit=False)
    clip_model.eval()

    def decode(tokens):
        return net.forward_decoder(tokens)[0].cpu().numpy() * std + mean

    def seam_and_goal(motion, d):
        first_local = mf.local_joint_positions(motion.astype(np.float32))[0]
        prev_end = d["prefix_pose"].reshape(22, 3)
        seam = float(np.linalg.norm(first_local - prev_end, axis=-1).mean())
        goal = float(np.linalg.norm(se2_place(motion, d["start"], mf)[-1] - d["goal"]))
        return seam, goal

    report = {}
    jobs = [(n, sg) for n in args.ckpt_name for sg in args.prefix_noise_mm]
    for name, noise_mm in jobs:
        cdir = os.path.join(args.ckpt_root, name)
        ns = json.load(open(os.path.join(cdir, "norm_stats.json")))
        cm_, cs_ = np.array(ns["cond_mean"], np.float32), np.array(ns["cond_std"], np.float32)
        tr = build_transformer(ns["clip_dim"])
        tr.load_state_dict(torch.load(os.path.join(cdir, "net_final.pth"), map_location="cpu")["trans"], strict=True)
        tr.eval().to(DEVICE)

        seams, goals = [], []
        rng = np.random.RandomState(args.seed + 1)
        with torch.no_grad():
            for n, i in enumerate(idxs):
                d = test[i]
                if args.prefix_source == "recon" and ns["cond_mode"] == "rel_prefix":
                    d = dict(d)
                    gt_tok = torch.from_numpy(d["tokens"]).long().unsqueeze(0).to(DEVICE)
                    d["prefix_pose"] = mf.local_joint_positions(
                        decode(gt_tok).astype(np.float32))[0].ravel().astype(np.float32)
                if noise_mm > 0 and ns["cond_mode"] == "rel_prefix":
                    d = dict(d)
                    d["prefix_pose"] = (d["prefix_pose"]
                                        + rng.randn(*d["prefix_pose"].shape).astype(np.float32)
                                        * (noise_mm / 1000.0))
                feat = clip_model.encode_text(clip.tokenize([d["text"]], truncate=True).to(DEVICE)).float()
                extra = torch.from_numpy(cond_extra(d, ns["cond_mode"], cm_, cs_)).unsqueeze(0).to(DEVICE)
                tok = tr.sample(torch.cat([feat, extra], -1), if_categorial=False)
                if tok.numel() == 0:
                    continue
                # seam is always scored against the TRUE seam pose, never the
                # perturbed one -- the question is whether the model still lands
                # on the real pose when told a noisy version of it.
                s, g = seam_and_goal(decode(tok), test[i])
                seams.append(s)
                goals.append(g)
        label = name + ("" if args.prefix_source == "gt" else " [recon prefix]")
        label = label if noise_mm == 0 else f"{label} +{noise_mm:g}mm noise"
        report[label] = (summarize(seams), summarize(goals), ns["cond_mode"])
        del tr
        torch.cuda.empty_cache()

    # controls
    o_seam, o_goal, gt_seam = [], [], []
    with torch.no_grad():
        for i in idxs:
            d = test[i]
            tok = torch.from_numpy(d["tokens"]).long().unsqueeze(0).to(DEVICE)
            s, g = seam_and_goal(decode(tok), d)
            o_seam.append(s)
            o_goal.append(g)
            # pure data check, no VQ-VAE: the same physical seam frame as seen
            # from segment A's canonicalization vs segment B's. Nonzero because
            # process_file normalizes per segment (see prepare_continuation_data).
            gt_seam.append(float(np.linalg.norm(
                d["a_end_pose"].reshape(22, 3) - d["prefix_pose"].reshape(22, 3), axis=-1).mean()))
    report["ORACLE (GT tokens)"] = (summarize(o_seam), summarize(o_goal), None)
    report["CANON. FLOOR (no VQ-VAE)"] = (summarize(gt_seam), None, None)

    print(f"\n{'':<28}{'seam err (mm)':>16}{'goal err (m)':>16}")
    for k, (s, g, mode) in report.items():
        gs = "—" if g is None else f"{g['mean']:.4f}"
        tag = f" [{mode}]" if mode else ""
        print(f"{k+tag:<28}{s['mean']*1000:>15.1f} {gs:>15}")
    print(f"\n(seam = mean per-joint |prev-segment end pose - generated first pose|, "
          f"n={report['ORACLE (GT tokens)'][0]['n']})")


if __name__ == "__main__":
    main()
