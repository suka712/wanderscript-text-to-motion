#!/usr/bin/env python3
"""Path B / Stage 1 / Step 0 — the falsifiable eval.

Measures whether distillation made the model itself scene-aware: does GREEDY decoding (no
guided_seg, no A*) now avoid obstacles? Target (inverts RESULTS §9/§13): greedy collision <
straight-line control, at equal goal error, on HELD-OUT scenes.

Discipline (CLAUDE.md risk #4):
 - straight-line polyline = the oracle control that makes a collision % readable.
 - HELD-OUT scenes only (eval_scenes.json from distill_guided): tests generalisation, not
   memorisation of the distilled chains.
 - HARD waypoints (same obstacle-biased sampler used for generation): on free-floor waypoints
   collision saturates at ~0 and the test is blind (the §8 saturation trap). Obstacles must be
   present for 'avoids obstacles' to be measurable.
 - goal error reported alongside collision: a model that avoids by not moving must not score as a
   win. Both models run on IDENTICAL scenes/waypoints; greedy is deterministic so no seed issue.
"""
import argparse
import json
import os
import sys

import clip
import numpy as np
import torch

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
for p in ["src", "scripts/track1", "scripts/chaining"]:
    sys.path.insert(0, os.path.join(REPO_ROOT, p))

import motion_features as mf  # noqa: E402
from humanise_join import build_flat_join, get_record, compute_track2  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402
from rollout import load_model  # noqa: E402
from collision_guided import run_chain, chain_metrics, straight_line_collision  # noqa: E402
from distill_guided import sample_hard_waypoints, T2M, HUMANISE, BEV, TALL  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True,
                    help="comma list label=dir,label=dir (e.g. pre=.../goalaug,post=.../distilled)")
    ap.add_argument("--vqvae-ckpt", required=True)
    ap.add_argument("--eval-json", required=True, help="eval_scenes.json from distill_guided")
    ap.add_argument("--n-eval", type=int, default=60)
    ap.add_argument("--n-segments", type=int, default=6)
    ap.add_argument("--min-step", type=float, default=0.6)
    ap.add_argument("--max-step", type=float, default=1.2)
    ap.add_argument("--hard-frac", type=float, default=0.8)
    ap.add_argument("--zero-occ", action="store_true",
                    help="zero the occupancy crop fed to every model -- the occ-ablation control. "
                         "If a model's collision/goal are unchanged vs real occ, it IGNORES the "
                         "scene input (architecture doesn't route occ into the path); if they "
                         "change, it USES occ (miscalibration is then a data-design issue).")
    ap.add_argument("--seed", type=int, default=7, help="eval seed (different from generation's)")
    args = ap.parse_args()

    labels, dirs = [], {}
    for kv in args.ckpts.split(","):
        lab, d = kv.split("=")
        labels.append(lab); dirs[lab] = os.path.expanduser(d)

    eval_scenes = set(json.load(open(args.eval_json))["eval_scenes"])
    print(f"eval on {len(eval_scenes)} held-out scenes; models={labels}\n", flush=True)

    net = load_vqvae(ckpt_path=args.vqvae_ckpt, device=DEV); net.eval()
    meta = f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta"
    mean = np.load(f"{meta}/mean.npy").astype(np.float32)
    std = np.load(f"{meta}/std.npy").astype(np.float32)
    cmodel, _ = clip.load("ViT-B/32", device=DEV, jit=False); cmodel.eval()
    models = {lab: load_model(dirs[lab]) for lab in labels}  # lab -> (trans, ns)

    flat = build_flat_join()
    walk_idx = [i for i, p in enumerate(flat) if p["action"] == "walk"]
    rng = np.random.RandomState(args.seed); rng.shuffle(walk_idx)

    agg = {lab: {"coll": [], "goal": []} for lab in labels}
    agg["line"] = []
    done = 0
    for idx in walk_idx:
        if done >= args.n_eval:
            break
        rec = get_record(int(idx))
        if rec.scene not in eval_scenes:
            continue
        fb, ft = os.path.join(BEV, f"{rec.scene}.npz"), os.path.join(TALL, f"{rec.scene}.npz")
        if not (os.path.exists(fb) and os.path.exists(ft)):
            continue
        zb, zt = np.load(fb), np.load(ft)
        occ, extent = zb["occ"].astype(np.float32), zb["extent"]
        tall = zt["occ"].astype(np.float32)
        # occ_model = what the MODEL sees (ablated if --zero-occ). occ stays REAL for waypoint
        # sampling + free-cell checks, so both runs get IDENTICAL waypoints -- only the model's
        # scene input differs. (Zeroing occ globally would make the sampler treat the whole room
        # as free and pick different waypoints, confounding the ablation.)
        occ_model = np.zeros_like(occ) if args.zero_occ else occ
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
        line = straight_line_collision(start_pose[:2], wps, tall, extent)

        row = {}
        ok = True
        for lab in labels:
            trans, ns = models[lab]
            segs = run_chain(trans, net, cmodel, mean, std, ns, texts, wps, start_pose, prefix,
                             occ_model, extent, tall, True, "greedy", 1, 0.0, rng)
            if len(segs) < args.n_segments:
                ok = False; break
            row[lab] = chain_metrics(segs, tall, extent)
        if not ok:
            continue
        for lab in labels:
            agg[lab]["coll"].append(row[lab]["coll"]); agg[lab]["goal"].append(row[lab]["goal"])
        agg["line"].append(line)
        done += 1
        if done % 10 == 0 or done == args.n_eval:
            msg = "  ".join(f"{lab}={row[lab]['coll']*100:.1f}%" for lab in labels)
            print(f"  [{done}/{args.n_eval}] {rec.scene}  line={line*100:.1f}%  {msg}", flush=True)

    if done == 0:
        print("no eval rollouts produced"); return
    print(f"\n=== {done} held-out rollouts x {args.n_segments} segments (greedy decode, seed {args.seed}) ===")
    print(f"  {'oracle straight-line':16s} collision {np.mean(agg['line'])*100:5.2f}%")
    for lab in labels:
        print(f"  {lab:16s} collision {np.mean(agg[lab]['coll'])*100:5.2f}%   "
              f"goal {np.mean(agg[lab]['goal']):.3f} m")
    print("\nTARGET: a post model with collision < straight-line at goal ~equal to pre "
          "=> the NET learned to avoid (inverts §9/§13).")


if __name__ == "__main__":
    main()
