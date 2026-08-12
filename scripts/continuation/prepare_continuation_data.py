#!/usr/bin/env python3
"""Continuation probe data prep (docs/07_continuation_probe.md).

Splits each HUMANISE clip into two consecutive segments A and B and builds the
data for asking: can the transformer generate B starting from the body
configuration A ended in?

WHY THIS IS THE HARD PART (CLAUDE.md 2d). Motion is canonicalized per segment:
each segment's frame 0 is moved to the origin at a canonical heading. So B's
tokens are expressed in B's own frame, and A's tokens in A's. Conditioning B on
A's raw TOKENS is therefore ill-posed -- they describe a different frame. What
IS shared between them is the body configuration at the seam: root-relative,
heading-canonicalized joint positions, which are frame-independent by
construction. That is what we condition on (66 dims = 22 joints x 3), read
straight out of the 263-dim vector via motion_features.local_joint_positions.

Same lesson as the goal conditioning in docs/04_grounding.md: hand the model the
quantity in the frame it generates in, rather than something it must transform.

A = cm[:t+2] and B = cm[t:] overlap at the seam frame, so the same physical pose
appears at the end of A and the start of B. A extends one frame past the seam
because process_file drops the last frame when differencing for velocities.

Output per clip: B's tokens + B's start/goal (for the usual relative-frame goal
conditioning) + prefix_pose (the seam pose in B's frame -- the conditioning
input) + a_end_pose (the same seam pose in A's frame -- a diagnostic).
"""
import argparse
import os
import pickle
import sys

import numpy as np
import torch

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import motion_features as mf  # noqa: E402
from humanise_join import get_record, compute_track2  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402

HUMANISE_ROOT = os.environ.get("WANDER_HUMANISE_ROOT", "/media/user/2tb/motion_data/HUMANISE")
T2M_GPT_ROOT = os.environ.get("WANDER_T2M_GPT_ROOT", "/home/user/Khiem-ssh/T2M-GPT")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_T = 196
MIN_SEG_FRAMES = 16  # each segment must survive the /4 token downsample with room to spare


def crop_to_multiple(T, factor=4, max_t=MAX_T):
    return (min(T, max_t) // factor) * factor


def load_split(name):
    with open(os.path.join(HUMANISE_ROOT, f"{name}.txt")) as f:
        return [int(line.strip()) for line in f if line.strip()]


def encode(net, data263, mean, std):
    norm = (data263 - mean) / std
    with torch.no_grad():
        x = torch.from_numpy(norm).unsqueeze(0).to(DEVICE)
        return net.encode(x)[0].cpu().numpy().astype(np.int64)


def process_split(name, net, mean, std):
    out, skipped = [], 0
    for n_done, idx in enumerate(load_split(name)):
        cm = np.load(os.path.join(HUMANISE_ROOT, "contact_motion", "motions", f"{idx:05d}.npy"))
        T_raw = cm.shape[0]
        if T_raw < 2 * MIN_SEG_FRAMES + 2:
            skipped += 1
            continue
        t = (T_raw // 2 // 4) * 4                      # split point, token-aligned
        # A must extend one raw frame PAST the seam: process_file drops the last
        # frame when differencing for velocities, so cm[:t+2] yields a 263 array
        # whose index t is the seam frame -- the same physical frame as B's
        # index 0. Using cm[:t+1] leaves A ending at raw frame t-1, an off-by-one
        # worth ~28mm of joint motion at this frame rate.
        seg_a, seg_b = cm[:t + 2], cm[t:]
        if seg_a.shape[0] < MIN_SEG_FRAMES or seg_b.shape[0] < MIN_SEG_FRAMES:
            skipped += 1
            continue
        try:
            d_a, *_ = mf.humanise_positions_to_263(seg_a)
            d_b, *_ = mf.humanise_positions_to_263(seg_b)
        except Exception:
            skipped += 1
            continue
        Tb = crop_to_multiple(d_b.shape[0])
        if d_a.shape[0] < 4 or Tb < 4:
            skipped += 1
            continue
        # A is never tokenized here -- only its seam pose is needed -- so it is
        # deliberately NOT cropped to a multiple of 4, which would move the seam.
        d_a_full = d_a.astype(np.float32)
        d_b = d_b[:Tb].astype(np.float32)

        # Body configuration at the seam, two ways.
        #
        # a_end_pose: the seam frame as canonicalized within SEGMENT A.
        # gt_first:   the same physical frame as canonicalized within SEGMENT B.
        # These are NOT equal, even with the off-by-one fixed, because
        # process_file canonicalizes per segment: uniform_skeleton rescales from
        # the segment's own first frame, and the floor offset is the segment's
        # own minimum height. That residual is a real property of chaining
        # canonicalized segments and is reported by the eval as DATA FLOOR.
        #
        # The CONDITIONING input is gt_first, not a_end_pose. That is not
        # cheating: at inference, segment k+1's canonical frame is defined by the
        # seam pose itself, so canonicalizing segment k's world-frame ending pose
        # into k+1's frame yields exactly this quantity. Conditioning on
        # a_end_pose would instead hand the model a signal in the wrong frame --
        # precisely the mistake docs/04_grounding.md already paid for once.
        la = mf.local_joint_positions(d_a_full)
        seam_idx = min(t, la.shape[0] - 1)
        a_end_pose = la[seam_idx].astype(np.float32)                          # (22,3)
        gt_first = mf.local_joint_positions(d_b)[0].astype(np.float32)        # (22,3)

        rec = get_record(idx)
        _, xy, _, sincos = compute_track2(rec)
        # B's world-frame start is the split frame; its goal is B's own last frame
        if t + Tb - 1 >= xy.shape[0]:
            skipped += 1
            continue
        start_b = np.array([xy[t, 0], xy[t, 1], sincos[t, 0], sincos[t, 1]], dtype=np.float32)
        goal_b = xy[t + Tb - 1].astype(np.float32)

        out.append({
            "index": idx,
            "tokens": encode(net, d_b, mean, std),
            "start": start_b,
            "goal": goal_b,
            "prefix_pose": gt_first.ravel(),         # (66,) conditioning input
            "a_end_pose": a_end_pose.ravel(),        # (66,) A-frame view of the same seam
            "text": rec.utterance,
            "action": rec.action,
        })
        if (n_done + 1) % 2000 == 0:
            print(f"  [{name}] {n_done + 1}", flush=True)

    print(f"{name}: {len(out)} kept, {skipped} skipped")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    net = load_vqvae(ckpt_path=args.ckpt, device=DEVICE) if args.ckpt else load_vqvae(device=DEVICE)
    mean = np.load(f"{T2M_GPT_ROOT}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy").astype(np.float32)
    std = np.load(f"{T2M_GPT_ROOT}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy").astype(np.float32)
    print(f"VQ-VAE from {args.ckpt or 'frozen pretrained'} -> {args.out_dir}")

    for split in ["train", "test"]:
        data = process_split(split, net, mean, std)
        with open(os.path.join(args.out_dir, f"{split}.pkl"), "wb") as f:
            pickle.dump(data, f)
        print(f"wrote {split}.pkl ({len(data)} clips)")

    # Diagnostic: how far apart are the two canonicalizations of the SAME
    # physical seam frame? With the off-by-one fixed this is purely the
    # per-segment normalization residual (skeleton rescale + floor offset), and
    # it is the irreducible floor for chaining re-canonicalized segments.
    d = pickle.load(open(os.path.join(args.out_dir, "test.pkl"), "rb"))
    err = np.array([np.linalg.norm(
        r["a_end_pose"].reshape(22, 3) - r["prefix_pose"].reshape(22, 3), axis=-1).mean()
        for r in d])
    print(f"\nCANONICALIZATION RESIDUAL at the seam (A-frame vs B-frame view of the "
          f"same physical pose):\n  mean {err.mean()*1000:.2f} mm, median "
          f"{np.median(err)*1000:.2f} mm")


if __name__ == "__main__":
    main()
