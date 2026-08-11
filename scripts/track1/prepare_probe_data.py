#!/usr/bin/env python3
"""
Grounding probe data prep (docs/track_1/001_grounding_probe.md, "Data prep").

For every HUMANISE clip in the official train/test split (HUMANISE/train.txt,
test.txt -- reused as-is rather than inventing a new split), extracts:
  - motion tokens from the FROZEN T2M-GPT VQ-VAE (net.encode)
  - start pose (x, y, sin(yaw), cos(yaw)) at the first frame and goal (x, y)
    at the last frame, from the world-frame track (humanise_join.compute_track2)
  - the clip's text utterance (for CLIP conditioning)

Frame alignment note: process_file (via humanise_positions_to_263) drops the
last frame when computing per-frame velocities, so the 263-dim / token
sequence has one fewer frame than the raw (T, 22, 3) clip. xy/sincos from
compute_track2 are indexed on the raw clip's frame count and are aligned to
the 263-dim sequence starting from frame 0, so goal is read at index
T_263 - 1 (the last frame actually covered by the tokenized target), not the
raw clip's last frame.

Output: one pickle per split under WANDER_MOTION_DATA_ROOT/../track1_probe/tokens/
(a list of dicts: index, tokens (int64 array), start (float32 (4,)),
goal (float32 (2,)), text (str), action (str)).
"""
import os
import pickle
import sys

import numpy as np
import torch

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import motion_features as mf  # noqa: E402
from humanise_join import build_flat_join, get_record, compute_track2  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402

HUMANISE_ROOT = os.environ.get("WANDER_HUMANISE_ROOT", "/media/user/2tb/motion_data/HUMANISE")
T2M_GPT_ROOT = os.environ.get("WANDER_T2M_GPT_ROOT", "/home/user/Khiem-ssh/T2M-GPT")
OUT_DIR = os.environ.get(
    "WANDER_TRACK1_PROBE_ROOT",
    os.path.join(os.path.dirname(os.environ.get("WANDER_MOTION_DATA_ROOT",
                 "/media/user/2tb/motion_data")), "track1_probe"),
)
TOKENS_DIR = os.path.join(OUT_DIR, "tokens")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_T = 196  # same convention as scripts/verify/check14_mpjpe_evaluator_norm.py


def crop_to_multiple(T, factor=4, max_t=MAX_T):
    T = min(T, max_t)
    return (T // factor) * factor


def load_split(name):
    with open(os.path.join(HUMANISE_ROOT, f"{name}.txt")) as f:
        return [int(line.strip()) for line in f if line.strip()]


def process_split(name, net, mean, std):
    indices = load_split(name)
    out = []
    skipped = 0
    for n_done, idx in enumerate(indices):
        cm = np.load(os.path.join(HUMANISE_ROOT, "contact_motion", "motions", f"{idx:05d}.npy"))
        if cm.shape[0] < 8:
            skipped += 1
            continue
        data263_full, *_ = mf.humanise_positions_to_263(cm)
        T = crop_to_multiple(data263_full.shape[0])
        if T < 4:
            skipped += 1
            continue
        data263 = data263_full[:T].astype(np.float32)
        # VQ-VAE was trained on normalized input (its own checkpoint mean/std,
        # NOT raw H3D Mean.npy/Std.npy -- see check14's docstring / CLAUDE.md's
        # "normalization pitfall"). Encoding raw data263 still produces *some*
        # tokens (the encoder doesn't hard-fail on out-of-distribution scale),
        # but they decode to badly wrong absolute trajectories -- confirmed via
        # a ground-truth-token round-trip during eval debugging (~105m mean
        # error un-normalized vs ~0.5m normalized on the same 10 clips).
        norm = (data263 - mean) / std

        with torch.no_grad():
            x = torch.from_numpy(norm).unsqueeze(0).to(DEVICE)
            tokens = net.encode(x)[0].cpu().numpy().astype(np.int64)

        record = get_record(idx)
        _, xy, _, sincos = compute_track2(record)
        start = np.array([xy[0, 0], xy[0, 1], sincos[0, 0], sincos[0, 1]], dtype=np.float32)
        goal = xy[T - 1].astype(np.float32)

        out.append({
            "index": idx,
            "tokens": tokens,
            "start": start,
            "goal": goal,
            "text": record.utterance,
            "action": record.action,
        })
        if (n_done + 1) % 2000 == 0:
            print(f"  [{name}] {n_done + 1}/{len(indices)}", flush=True)

    print(f"{name}: {len(out)} clips processed, {skipped} skipped (too short)")
    return out


def main():
    os.makedirs(TOKENS_DIR, exist_ok=True)
    net = load_vqvae(device=DEVICE)
    mean = np.load(f"{T2M_GPT_ROOT}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy").astype(np.float32)
    std = np.load(f"{T2M_GPT_ROOT}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy").astype(np.float32)
    print(f"VQ-VAE loaded on {DEVICE}, using evaluator-consistent mean/std")

    for split in ["train", "test"]:
        data = process_split(split, net, mean, std)
        out_path = os.path.join(TOKENS_DIR, f"{split}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(data, f)
        print(f"wrote {out_path} ({len(data)} clips)")


if __name__ == "__main__":
    main()
