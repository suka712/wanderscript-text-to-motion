#!/usr/bin/env python3
"""
Track 2 -- per-category held-out MPJPE eval, callable as a function (for the
finetune training loop to call at every eval checkpoint) or standalone (for
the frozen-baseline "before" number and any ad-hoc checkpoint).

Held-out discipline: every number here comes ONLY from each dataset's
official test.txt split (H3D_ROOT/test.txt, HUMANISE_ROOT/test.txt) -- never
train.txt. This differs from check14/15/16/17/20 (frozen-checkpoint canary,
random-sampled from ALL 19,648 HUMANISE clips with no held-out split, which
was fine there because that model never trained on any of it). Once we are
the ones finetuning, train/eval separation matters, so this script re-derives
the same categories on the held-out subset only. Numbers will differ slightly
from check14/20's cited 45.3/47.9/67.6/72.6/139.8/90.1 for that reason (both
methodology AND that those were captured on the ORIGINAL frozen checkpoint);
this script also supports running against the frozen checkpoint to get an
apples-to-apples held-out "before" baseline, reported alongside the
finetuned "after" so the comparison is against the same eval methodology
(see docs/track_2/RESULTS.md).

Normalization: evaluator-consistent, i.e. the frozen checkpoint's own meta
mean/std (checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/{mean,std}.npy)
-- per the pitfall note in 001_tokenizer_finetune.md, NOT H3D_ROOT's
Mean.npy/Std.npy (using the latter inflates every number ~2-3x). This is also
what the finetune TRAINS with (see train_vqvae_joint_finetune.py) -- the
finetune warm-starts from the frozen checkpoint's embedding space and keeps
using its normalization throughout, rather than recomputing new stats from
the joint HumanML3D+HUMANISE training distribution. Kept fixed across
train/eval and across checkpoints so every number in RESULTS.md is directly
comparable.
"""
import argparse
import json
import os
import re
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import motion_features as mf  # noqa: E402
from humanise_join import build_flat_join  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402

T2M_GPT_ROOT = os.environ.get("WANDER_T2M_GPT_ROOT", "/home/user/Khiem/T2M-GPT")
H3D_ROOT = os.environ.get("WANDER_H3D_ROOT", "/media/user/2tb/motion_data/H3D")
HUMANISE_ROOT = os.environ.get("WANDER_HUMANISE_ROOT", "/media/user/2tb/motion_data/HUMANISE")
HUMANISE_MOTIONS = f"{HUMANISE_ROOT}/contact_motion/motions"
DEFAULT_CKPT = f"{T2M_GPT_ROOT}/pretrained/VQVAE/net_best_fid.pth"
EVAL_MEAN_PATH = f"{T2M_GPT_ROOT}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy"
EVAL_STD_PATH = f"{T2M_GPT_ROOT}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_T = 196
N_CLIPS = 200  # per-category sample cap, same convention as check14

LIE_PATTERN = re.compile(
    r"\blying\b|\blie down\b|\blies down\b|\blie on\b|\blies on\b|"
    r"\blaying\b|\blay down\b|\blays down\b|\blay on\b"
)


def crop_to_multiple(T, factor=4, max_t=MAX_T):
    T = min(T, max_t)
    return (T // factor) * factor


def roundtrip_mpjpe(net, data263, mean, std):
    T = crop_to_multiple(data263.shape[0])
    if T < 4:
        return None
    data263 = data263[:T].astype(np.float32)
    norm = (data263 - mean) / std
    x = torch.from_numpy(norm).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        x_out, _, _ = net(x)
    recon263 = x_out[0].cpu().numpy() * std + mean
    orig_local = mf.local_joint_positions(data263)
    recon_local = mf.local_joint_positions(recon263.astype(np.float32))
    T2 = min(orig_local.shape[0], recon_local.shape[0])
    return np.linalg.norm(orig_local[:T2] - recon_local[:T2], axis=-1).mean()


def _read_ids(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def h3d_test_ids(exclude_mirror=True):
    ids = _read_ids(f"{H3D_ROOT}/test.txt")
    if exclude_mirror:
        ids = [i for i in ids if not i.startswith("M")]
    return ids


def h3d_lie_test_ids():
    """H3D-lie control, restricted to held-out test.txt (see check20 for the
    original, non-held-out version of this keyword search)."""
    matches = []
    for name in h3d_test_ids():
        text_path = f"{H3D_ROOT}/texts/{name}.txt"
        if not os.path.exists(text_path):
            continue
        with open(text_path) as f:
            text = f.read().lower()
        if LIE_PATTERN.search(text):
            matches.append(name)
    return matches


def humanise_category_test_indices():
    """HUMANISE motion indices per action, restricted to held-out test.txt."""
    test_ids = set(int(x) for x in _read_ids(f"{HUMANISE_ROOT}/test.txt"))
    flat = build_flat_join()
    by_action = {}
    for i, p in enumerate(flat):
        if i not in test_ids:
            continue
        by_action.setdefault(p["action"], []).append(i)
    return by_action


def eval_h3d_baseline(net, mean, std, n_clips=N_CLIPS, seed=0):
    ids = h3d_test_ids()
    rng = np.random.RandomState(seed)
    sample = rng.choice(ids, size=min(n_clips, len(ids)), replace=False)
    errs = []
    for name in sample:
        data263 = np.load(f"{H3D_ROOT}/new_joint_vecs/{name}.npy").astype(np.float32)
        e = roundtrip_mpjpe(net, data263, mean, std)
        if e is not None:
            errs.append(e)
    return np.array(errs)


def eval_h3d_lie(net, mean, std, n_clips=N_CLIPS, seed=0):
    ids = h3d_lie_test_ids()
    rng = np.random.RandomState(seed)
    sample = rng.choice(ids, size=min(n_clips, len(ids)), replace=False) if ids else np.array([])
    errs = []
    for name in sample:
        data263 = np.load(f"{H3D_ROOT}/new_joint_vecs/{name}.npy").astype(np.float32)
        e = roundtrip_mpjpe(net, data263, mean, std)
        if e is not None:
            errs.append(e)
    return np.array(errs), len(ids)


def eval_humanise_category(net, mean, std, action, indices, n_clips=N_CLIPS, seed=1):
    rng = np.random.RandomState(seed)
    if not indices:
        return np.array([])
    chosen = rng.choice(indices, size=min(n_clips, len(indices)), replace=False)
    errs = []
    for i in chosen:
        cm = np.load(f"{HUMANISE_MOTIONS}/{i:05d}.npy")
        if cm.shape[0] < 6:
            continue
        data263, _, _, _ = mf.humanise_positions_to_263(cm)
        e = roundtrip_mpjpe(net, data263.astype(np.float32), mean, std)
        if e is not None:
            errs.append(e)
    return np.array(errs)


def run_full_eval(net, mean=None, std=None, n_clips=N_CLIPS, seed=0, verbose=True):
    """Runs the full held-out per-category suite against an already-loaded,
    net.eval()'d VQ-VAE. Returns a dict of {category: {mean_mm, median_mm,
    max_mm, n}} in millimeters, plus the raw ratio-to-H3D-baseline.
    """
    if mean is None:
        mean = np.load(EVAL_MEAN_PATH).astype(np.float32)
    if std is None:
        std = np.load(EVAL_STD_PATH).astype(np.float32)

    results = {}

    h3d_errs = eval_h3d_baseline(net, mean, std, n_clips, seed)
    results["h3d_baseline"] = _summarize(h3d_errs)

    h3d_lie_errs, n_h3d_lie_total = eval_h3d_lie(net, mean, std, n_clips, seed)
    results["h3d_lie"] = _summarize(h3d_lie_errs)
    results["h3d_lie"]["n_available_in_test"] = n_h3d_lie_total

    by_action = humanise_category_test_indices()
    for action in ["walk", "stand up", "sit", "lie"]:
        indices = by_action.get(action, [])
        errs = eval_humanise_category(net, mean, std, action, indices, n_clips, seed=1)
        results[f"humanise_{action.replace(' ', '_')}"] = _summarize(errs)
        results[f"humanise_{action.replace(' ', '_')}"]["n_available_in_test"] = len(indices)

    h3d_mean = results["h3d_baseline"]["mean_mm"]
    for key, val in results.items():
        if val["n"] > 0 and h3d_mean > 0:
            val["ratio_to_h3d_baseline"] = val["mean_mm"] / h3d_mean

    if verbose:
        print_results_table(results)

    return results


def _summarize(errs):
    if len(errs) == 0:
        return {"mean_mm": float("nan"), "median_mm": float("nan"), "max_mm": float("nan"), "n": 0}
    return {
        "mean_mm": float(errs.mean() * 1000),
        "median_mm": float(np.median(errs) * 1000),
        "max_mm": float(errs.max() * 1000),
        "n": int(len(errs)),
    }


def print_results_table(results):
    print("=" * 78)
    print(f"{'category':<20} {'mean(mm)':>10} {'median(mm)':>12} {'max(mm)':>10} {'n':>5} {'ratio-H3D':>10}")
    for key, val in results.items():
        ratio = val.get("ratio_to_h3d_baseline", float("nan"))
        print(f"{key:<20} {val['mean_mm']:>10.2f} {val['median_mm']:>12.2f} "
              f"{val['max_mm']:>10.2f} {val['n']:>5} {ratio:>10.2f}")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--n-clips", type=int, default=N_CLIPS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--label", default=None, help="free-text label stored in the output JSON")
    args = ap.parse_args()

    net = load_vqvae(ckpt_path=args.ckpt, device=DEVICE)
    net.eval()
    print(f"Loaded VQ-VAE from {args.ckpt} on {DEVICE}, evaluator-consistent normalization")

    results = run_full_eval(net, n_clips=args.n_clips, seed=args.seed)

    if args.out_json:
        payload = {"ckpt": args.ckpt, "label": args.label, "results": results}
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved -> {args.out_json}")


if __name__ == "__main__":
    main()
