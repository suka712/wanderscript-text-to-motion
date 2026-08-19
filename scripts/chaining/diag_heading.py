#!/usr/bin/env python3
"""Diagnostic: does the body TURN to face its direction of travel, or strafe?

A viewer noticed chained clips seem to hold one facing for the whole duration. Two
very different causes:
  (a) demo-specific: demo_interaction sets the start facing the furniture and points
      every walk hop straight at it, so no turn is needed.
  (b) real behaviour: the model reaches side goals by TRANSLATING while the body keeps
      facing forward (strafing), so facing never tracks travel.

This forces direction changes (free-floor waypoints in a band, various bearings) and,
per frame, compares:
  facing  = yaw_from_joints(world)      -- where the BODY points
  travel  = atan2 of pelvis velocity    -- where it is MOVING
If facing tracks travel, the model turns. If facing is ~constant while travel varies,
it strafes. Reported per segment and over the whole chain.
"""
import argparse
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
from rollout import rollout, load_model, yaw_from_joints  # noqa: E402
from demo_rollout import sample_waypoints  # noqa: E402

T2M = os.environ.get("WANDER_T2M_GPT_ROOT")
HUMANISE = os.environ.get("WANDER_HUMANISE_ROOT")
BEV = os.path.expanduser("~/wander_data/bev_cache")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def wrap_deg(a):
    return (a + 180) % 360 - 180


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vqvae-ckpt", required=True)
    ap.add_argument("--n-rollouts", type=int, default=6)
    ap.add_argument("--n-segments", type=int, default=6)
    ap.add_argument("--min-step", type=float, default=0.6)
    ap.add_argument("--max-step", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--reorient", action="store_true",
                    help="rotate each segment start to face its goal (inference heading fix)")
    args = ap.parse_args()

    net = load_vqvae(ckpt_path=args.vqvae_ckpt, device=DEV); net.eval()
    mean = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy").astype(np.float32)
    std = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy").astype(np.float32)
    cmodel, _ = clip.load("ViT-B/32", device=DEV, jit=False); cmodel.eval()
    trans, ns = load_model(args.ckpt)
    fa = ns["cond_mode"] in ("full_action", "full_action_head")

    flat = build_flat_join()
    walk_idx = [i for i, p in enumerate(flat) if p["action"] == "walk"]
    rng = np.random.RandomState(args.seed); rng.shuffle(walk_idx)

    done = 0
    facing_align, facing_const, turn_track = [], [], []
    for idx in walk_idx:
        if done >= args.n_rollouts:
            break
        rec = get_record(int(idx))
        f = os.path.join(BEV, f"{rec.scene}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f); occ, extent = z["occ"].astype(np.float32), z["extent"]
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
        wps = sample_waypoints(occ, extent, start_pose[:2], args.n_segments, args.min_step,
                               rng, max_step=args.max_step)
        if wps is None:
            continue
        acts = ["walk"] * args.n_segments if fa else None
        segs = rollout(trans, net, cmodel, clip, mean, std, ns,
                       ["walk to the target"] * args.n_segments, wps,
                       start_pose, prefix, occ, extent, actions=acts, reorient=args.reorient)
        if len(segs) < args.n_segments:
            continue

        done += 1
        print(f"\n=== rollout {done}  {rec.scene} ===")
        print(f"{'seg':<4}{'travel_brng':>12}{'facing_mean':>12}{'facing_rng':>11}"
              f"{'|face-travel|':>14}")
        all_face = []
        for si, s in enumerate(segs):
            w = s["world"]                       # (T,22,3)
            face = np.array([np.degrees(yaw_from_joints(w[t])) for t in range(len(w))])
            all_face.append(face)
            pel = w[:, 0, :2]
            vel = np.diff(pel, axis=0)
            step = np.linalg.norm(vel, axis=1)
            moving = step > 1e-3
            if moving.sum() < 2:
                travel = np.nan
            else:
                travel = np.degrees(np.arctan2(vel[moving][:, 1].sum(), vel[moving][:, 0].sum()))
            face_mean = np.degrees(np.arctan2(np.sin(np.radians(face)).mean(),
                                              np.cos(np.radians(face)).mean()))
            face_rng = wrap_deg(face.max() - face.min())
            dft = abs(wrap_deg(face_mean - travel)) if not np.isnan(travel) else np.nan
            print(f"{si+1:<4}{travel:>12.0f}{face_mean:>12.0f}{abs(face_rng):>11.0f}{dft:>14.0f}")
            if not np.isnan(dft):
                facing_align.append(dft)
            turn_track.append(abs(face_rng))
        chain_face = np.concatenate(all_face)
        # circular std of facing across the WHOLE chain (deg). ~0 => one direction.
        c = np.cos(np.radians(chain_face)).mean(); s2 = np.sin(np.radians(chain_face)).mean()
        R = np.hypot(c, s2)
        circ_std = np.degrees(np.sqrt(-2 * np.log(max(R, 1e-9))))
        facing_const.append(circ_std)
        print(f"  chain facing circular-std = {circ_std:.0f} deg "
              f"(0 = never turns; large = turns a lot)")

    print("\n================ SUMMARY ================")
    print(f"mean |facing - travel direction|   : {np.nanmean(facing_align):.0f} deg   "
          f"(small => body faces where it walks; ~90 => strafing)")
    print(f"mean within-segment facing range   : {np.mean(turn_track):.0f} deg   "
          f"(how much the body turns inside one segment)")
    print(f"mean chain facing circular-std     : {np.mean(facing_const):.0f} deg   "
          f"(0 => single fixed heading for the whole chain)")


if __name__ == "__main__":
    main()
