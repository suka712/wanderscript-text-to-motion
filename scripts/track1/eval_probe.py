#!/usr/bin/env python3
"""
Grounding probe eval (docs/track_1/001_grounding_probe.md, "Probe" +
"Decision gate"). Held-out generation: text (+ goal conditioning, if the model
was trained conditioned) -> frozen decoder -> canonicalized motion -> SE(2)
place at the FED start pose -> world-frame motion -> goal-error.

Two CONTROLS are reported alongside every model, on exactly the same clips.
Neither was present in the first version of this script, and their absence is
what allowed a 90-degree SE(2) placement bug to be read as a model result (see
se2_utils.CANONICAL_YAW_OFFSET):

  ORACLE  -- decode the clip's OWN ground-truth tokens and place them. No
    transformer involved, so this is the measurement's noise floor: VQ-VAE
    reconstruction error plus any residual placement error. If ORACLE is not
    small compared to the model numbers, the experiment cannot detect
    grounding at all and nothing below it means anything.
  NULL    -- distance from the fed start to the fed goal, i.e. what a trivial
    "never move" policy scores. A model must beat this to be doing anything.

start-error is NOT reported as a result: SE(2) placement composes frame 0
(which sits at the local origin by construction) onto the fed start pose, so
it is exactly 0.0 for any model. It is asserted instead.

Also reports a directional diagnostic -- correlation between the COMMANDED
local goal displacement and the ACHIEVED local displacement, per axis. A model
that is genuinely using the goal shows positive correlation; goal-error alone
can be dragged around by clip-duration effects and hides this.
"""
import argparse
import json
import os
import pickle
import sys

import clip
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import motion_features as mf  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402
from se2_utils import se2_place, world_to_local_xy  # noqa: E402
from train_probe import build_transformer, cond_extra  # noqa: E402

MOTION_DATA_ROOT = os.environ.get("WANDER_MOTION_DATA_ROOT", "/media/user/2tb/motion_data")
OUT_DIR = os.environ.get("WANDER_TRACK1_PROBE_ROOT", os.path.join(os.path.dirname(MOTION_DATA_ROOT), "track1_probe"))
TOKENS_DIR = os.path.join(OUT_DIR, "tokens")
T2M_GPT_ROOT = os.environ.get("WANDER_T2M_GPT_ROOT", "/home/user/Khiem-ssh/T2M-GPT")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def summarize(errs):
    e = np.asarray(errs)
    return {"mean": float(e.mean()), "median": float(np.median(e)),
            "std": float(e.std()), "sem": float(e.std() / np.sqrt(len(e))), "n": int(len(e))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-name", type=str, required=True, nargs="+",
                    help="one or more dir names under checkpoints/, evaluated on identical clips")
    ap.add_argument("--n-clips", type=int, default=200)
    ap.add_argument("--n-renders", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokens-dir", type=str, default=TOKENS_DIR)
    ap.add_argument("--vqvae-ckpt", type=str, default=None,
                    help="VQ-VAE to decode with. MUST be the same one --tokens-dir was "
                         "extracted with, or every number here is meaningless.")
    args = ap.parse_args()

    with open(os.path.join(args.tokens_dir, "test.pkl"), "rb") as f:
        test_manifest = pickle.load(f)
    rng = np.random.RandomState(args.seed)
    idxs = rng.choice(len(test_manifest), size=min(args.n_clips, len(test_manifest)), replace=False)

    net = load_vqvae(ckpt_path=args.vqvae_ckpt, device=DEVICE) if args.vqvae_ckpt else load_vqvae(device=DEVICE)
    net.eval()
    mean = np.load(f"{T2M_GPT_ROOT}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy").astype(np.float32)
    std = np.load(f"{T2M_GPT_ROOT}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy").astype(np.float32)
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE, jit=False)
    clip_model.eval()

    def decode_place(tokens, d):
        motion = net.forward_decoder(tokens)[0].cpu().numpy() * std + mean
        return se2_place(motion, d["start"], mf)

    report, renders = {}, {}

    for ckpt_name in args.ckpt_name:
        ckpt_dir = os.path.join(OUT_DIR, "checkpoints", ckpt_name)
        with open(os.path.join(ckpt_dir, "norm_stats.json")) as f:
            ns = json.load(f)
        conditioned = ns["conditioned"]
        cond_mode = ns.get("cond_mode", "abs")
        if "cond_mean" in ns:
            cond_mean = np.array(ns["cond_mean"], dtype=np.float32)
            cond_std = np.array(ns["cond_std"], dtype=np.float32)
        else:
            # Back-compat with the original absolute-frame run, which stored
            # only xy stats and applied them to start[:2] and goal, leaving
            # start's sin/cos untouched. Reconstruct the equivalent 6-dim form
            # so that checkpoint stays evaluable against the new controls.
            xm = np.array(ns["xy_mean"], dtype=np.float32)
            xs = np.array(ns["xy_std"], dtype=np.float32)
            cond_mean = np.concatenate([xm, [0.0, 0.0], xm]).astype(np.float32)
            cond_std = np.concatenate([xs, [1.0, 1.0], xs]).astype(np.float32)

        trans_encoder = build_transformer(ns["clip_dim"])
        trans_encoder.load_state_dict(
            torch.load(os.path.join(ckpt_dir, "net_final.pth"), map_location="cpu")["trans"], strict=True)
        trans_encoder.eval().to(DEVICE)

        errs, cmd, ach, shown = [], [], [], []
        with torch.no_grad():
            for n, i in enumerate(idxs):
                d = test_manifest[i]
                feat = clip_model.encode_text(clip.tokenize([d["text"]], truncate=True).to(DEVICE)).float()
                if conditioned:
                    extra = torch.from_numpy(cond_extra(d, cond_mode, cond_mean, cond_std)).unsqueeze(0).to(DEVICE)
                    cond = torch.cat([feat, extra], dim=-1)
                else:
                    cond = feat

                tokens = trans_encoder.sample(cond, if_categorial=False)
                if tokens.numel() == 0:
                    continue
                world_xy = decode_place(tokens, d)
                assert np.linalg.norm(world_xy[0] - d["start"][:2]) < 1e-6, "SE(2) placement broken"

                errs.append(float(np.linalg.norm(world_xy[-1] - d["goal"])))
                cmd.append(world_to_local_xy(d["goal"], d["start"]))
                ach.append(world_to_local_xy(world_xy[-1], d["start"]))
                if len(shown) < args.n_renders:
                    shown.append((d, world_xy, errs[-1]))
                if (n + 1) % 50 == 0:
                    print(f"  [{ckpt_name}] {n + 1}/{len(idxs)}", flush=True)

        cmd, ach = np.array(cmd), np.array(ach)
        stats = summarize(errs)
        stats["cond_mode"] = cond_mode if conditioned else None
        stats["corr_commanded_vs_achieved"] = [
            float(np.corrcoef(cmd[:, k], ach[:, k])[0, 1]) for k in range(2)]
        report[ckpt_name] = stats
        renders[ckpt_name] = shown

        with open(os.path.join(ckpt_dir, "eval_summary.json"), "w") as f:
            json.dump(stats, f, indent=2)
        del trans_encoder
        torch.cuda.empty_cache()

    # --- controls, same clips ---
    oracle, null = [], []
    with torch.no_grad():
        for i in idxs:
            d = test_manifest[i]
            tok = torch.from_numpy(d["tokens"]).long().unsqueeze(0).to(DEVICE)
            oracle.append(float(np.linalg.norm(decode_place(tok, d)[-1] - d["goal"])))
            null.append(float(np.linalg.norm(d["start"][:2] - d["goal"])))
    report["ORACLE (GT tokens)"] = summarize(oracle)
    report["NULL (stay at start)"] = summarize(null)

    print(f"\n{'':<26}{'mean':>9} {'median':>9} {'sem':>8} {'n':>5}  {'corr cmd/achieved':>20}")
    for k, v in report.items():
        c = v.get("corr_commanded_vs_achieved")
        cs = f"{c[0]:+.3f}, {c[1]:+.3f}" if c else ""
        print(f"{k:<26}{v['mean']:>8.4f}m {v['median']:>8.4f}m {v['sem']:>7.4f}m {v['n']:>5}  {cs:>20}")

    for ckpt_name, shown in renders.items():
        render_dir = os.path.join(OUT_DIR, "renders", ckpt_name)
        os.makedirs(render_dir, exist_ok=True)
        for k, (d, world_xy, goal_err) in enumerate(shown):
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(world_xy[:, 0], world_xy[:, 1], "-o", color="tab:blue", markersize=2, label="generated")
            ax.plot(*d["start"][:2], "o", color="tab:green", markersize=10, label="fed start")
            ax.plot(*d["goal"], "*", color="tab:red", markersize=15, label="fed goal")
            ax.set_title(f"{d['text'][:40]}\ngoal_err={goal_err:.2f}m", fontsize=9)
            ax.set_aspect("equal")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(os.path.join(render_dir, f"example_{k:02d}.png"), dpi=100)
            plt.close(fig)
        print(f"renders -> {render_dir}")

    with open(os.path.join(OUT_DIR, "eval_report.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
