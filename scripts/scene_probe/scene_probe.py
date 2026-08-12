#!/usr/bin/env python3
"""Scene-representation probe: does BEV + DINOv2 carry usable scene signal,
and does it beat the geometry we already have for free?

CLAUDE.md section 4 lists the scene encoder as SWAPPABLE and flags the specific
worry that DINOv2 is trained on natural images, so top-down renders are out of
its distribution. That has never been tested. This is the cheap test, run
BEFORE the expensive transformer work, for the same reason the grounding probe
ran before it: a scene representation that carries nothing is much cheaper to
discover now than at step 8.

TASK -- 4-way action classification (walk / sit / lie / stand up) from scene
context at the goal, with NO text input. Rationale: what the goal affords is
what determines which interaction is possible. HUMANISE's own object_label
confirms the labels are affordance-bearing (lie -> bed, sit -> chair/sofa), so
a representation that cannot separate these cannot inform interaction motion.
Text is deliberately excluded -- "lie on the bed" makes the label trivial and
would measure nothing about the scene features.

ARMS (all identical except the input representation):
  dinov2   DINOv2 ViT-S/14 CLS embedding of the RGB crop            (384-d)
  occ      the raw binary occupancy crop, downsampled               (n*n-d)
  rgb_raw  the RGB crop downsampled, no pretrained encoder          (3*n*n-d)
  prior    no scene input at all -- predicts the majority class
`rgb_raw` separates "the pretrained encoder is doing work" from "the crop
happens to be linearly separable"; without it, a dinov2 win is unattributable.

CROPS are taken in the AGENT'S OWN FRAME -- centered on the goal, rotated by
the start heading -- for the same reason the goal is fed start-relative in
docs/04_grounding.md. A world-axis-aligned crop would make the probe solve an
extra rotation it will never have to solve at inference.

SPLIT IS BY SCENE, not by clip. Many clips share a scene, so a clip-level
split leaks: the probe could memorize per-scene appearance and score well
while learning nothing transferable. This project has already published one
leaked number (see docs/02_baseline_calibration.md); this avoids a second.

Probe head is multinomial logistic regression -- deliberately linear, so the
result is about the representation rather than about a head that could learn
the task from anything.
"""
import argparse
import math
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "track1"))

from humanise_join import build_flat_join, get_record, compute_track2  # noqa: E402
from se2_utils import CANONICAL_YAW_OFFSET  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTIONS = ["walk", "sit", "lie", "stand up"]
CROP_M = 3.0   # half-width of the crop window, meters
CROP_PX = 112  # resampled crop size


def _install_sdpa_shim():
    """DINOv2 HEAD calls F.scaled_dot_product_attention (torch>=2.0); this env
    is torch 1.12. The explicit form below is mathematically identical for the
    non-causal, unmasked case DINOv2 uses."""
    if hasattr(nn.functional, "scaled_dot_product_attention"):
        return
    def sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
        a = (q @ k.transpose(-2, -1)) * scale
        if attn_mask is not None:
            a = a + attn_mask
        return a.softmax(-1) @ v
    nn.functional.scaled_dot_product_attention = sdpa


def load_dinov2(repo, weights, arch="vit_small"):
    _install_sdpa_shim()
    sys.path.insert(0, repo)
    import dinov2.models.vision_transformer as vits
    m = getattr(vits, arch)(patch_size=14, img_size=518, init_values=1.0, block_chunks=0)
    missing = m.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
    print(f"DINOv2 {arch}/14 loaded strict=True ({missing})")
    return m.eval().to(DEVICE)


def crop_agent_frame(raster, extent, goal_xy, yaw0, out_px=CROP_PX, half_m=CROP_M):
    """Bilinear-sample a square window centered at goal_xy, rotated so the
    agent's heading points 'up' in the crop. raster: (H,W) or (H,W,3)."""
    xmin, xmax, ymin, ymax = extent
    H, W = raster.shape[:2]
    th = yaw0 + CANONICAL_YAW_OFFSET
    c, s = np.cos(th), np.sin(th)
    lin = np.linspace(-half_m, half_m, out_px)
    gx, gy = np.meshgrid(lin, -lin)                      # image row 0 = +forward
    wx = goal_xy[0] + gx * c - gy * s
    wy = goal_xy[1] + gx * s + gy * c
    col = (wx - xmin) / (xmax - xmin) * W
    row = (ymax - wy) / (ymax - ymin) * H
    r0 = np.clip(np.round(row).astype(int), 0, H - 1)
    c0 = np.clip(np.round(col).astype(int), 0, W - 1)
    return raster[r0, c0]


def build_dataset(bev_dir, max_per_action, seed=0):
    flat = build_flat_join()
    by_action = {}
    for i, p in enumerate(flat):
        if p["action"] in ACTIONS:
            by_action.setdefault(p["action"], []).append(i)
    rng = np.random.RandomState(seed)
    chosen = []
    for a in ACTIONS:
        idx = by_action.get(a, [])
        chosen += list(rng.choice(idx, size=min(max_per_action, len(idx)), replace=False))
    rng.shuffle(chosen)

    rows, skipped = [], 0
    cache = {}
    for n, i in enumerate(chosen):
        p = flat[i]
        sid = p["scene"]
        path = os.path.join(bev_dir, f"{sid}.npz")
        if not os.path.exists(path):
            skipped += 1
            continue
        if sid not in cache:
            if len(cache) > 40:
                cache.clear()
            z = np.load(path)
            cache[sid] = (z["rgb"], z["occ"], z["extent"])
        rgb, occ, ext = cache[sid]
        try:
            rec = get_record(i)
            _, xy, _, sincos = compute_track2(rec)
        except Exception:
            skipped += 1
            continue
        goal = xy[-1].astype(np.float64)
        yaw0 = float(np.arctan2(sincos[0, 0], sincos[0, 1]))
        rows.append({
            "scene": sid,
            "y": ACTIONS.index(p["action"]),
            "rgb": crop_agent_frame(rgb, ext, goal, yaw0).astype(np.uint8),
            "occ": crop_agent_frame(occ.astype(np.float32), ext, goal, yaw0),
        })
        if (n + 1) % 500 == 0:
            print(f"  built {len(rows)} (skipped {skipped})", flush=True)
    print(f"dataset: {len(rows)} clips, {skipped} skipped, "
          f"{len({r['scene'] for r in rows})} scenes")
    return rows


def scene_split(rows, test_frac=0.3, seed=0):
    scenes = sorted({r["scene"] for r in rows})
    rng = np.random.RandomState(seed)
    rng.shuffle(scenes)
    n_test = max(1, int(len(scenes) * test_frac))
    test = set(scenes[:n_test])
    tr = [r for r in rows if r["scene"] not in test]
    te = [r for r in rows if r["scene"] in test]
    print(f"split by SCENE: {len(tr)} train / {len(te)} test clips, "
          f"{len(scenes)-n_test}/{n_test} scenes")
    return tr, te


IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def feats_dinov2(rows, model, bs=64):
    out = []
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        x = np.stack([(r["rgb"].astype(np.float32) / 255.0 - IMNET_MEAN) / IMNET_STD for r in chunk])
        t = torch.from_numpy(x).permute(0, 3, 1, 2).float().to(DEVICE)
        with torch.no_grad():
            out.append(model(t).cpu().numpy())
    return np.concatenate(out)


def feats_occ(rows, n=28):
    k = CROP_PX // n
    return np.stack([r["occ"].reshape(n, k, n, k).mean((1, 3)).ravel() for r in rows])


def feats_rgb_raw(rows, n=28):
    k = CROP_PX // n
    return np.stack([(r["rgb"].astype(np.float32) / 255.0)
                     .reshape(n, k, n, k, 3).mean((1, 3)).ravel() for r in rows])


def logreg(Xtr, ytr, Xte, yte, epochs=300, lr=1e-2, wd=1e-3):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    xtr = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    xte = torch.tensor(Xte, dtype=torch.float32, device=DEVICE)
    ttr = torch.tensor(ytr, dtype=torch.long, device=DEVICE)
    tte = torch.tensor(yte, dtype=torch.long, device=DEVICE)
    head = nn.Linear(xtr.shape[1], len(ACTIONS)).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        lossf(head(xtr), ttr).backward()
        opt.step()
    with torch.no_grad():
        pred = head(xte).argmax(1)
        acc = (pred == tte).float().mean().item()
        per = {ACTIONS[c]: ((pred == c) & (tte == c)).sum().item() / max((tte == c).sum().item(), 1)
               for c in range(len(ACTIONS))}
    return acc, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bev-dir", required=True)
    ap.add_argument("--dinov2-repo", default="/tmp/dinov2")
    ap.add_argument("--dinov2-weights", default="/tmp/dinov2_vits14.pth")
    ap.add_argument("--dinov2-weights-large", default=None,
                    help="optional second, bigger DINOv2 -- rules out 'the encoder was "
                         "just too small' as an explanation for a weak result")
    ap.add_argument("--dinov2-arch-large", default="vit_base")
    ap.add_argument("--max-per-action", type=int, default=1200)
    ap.add_argument("--cache", default=None, help="pickle path to cache the built crops")
    args = ap.parse_args()

    if args.cache and os.path.exists(args.cache):
        rows = pickle.load(open(args.cache, "rb"))
        print(f"loaded {len(rows)} rows from cache")
    else:
        rows = build_dataset(args.bev_dir, args.max_per_action)
        if args.cache:
            pickle.dump(rows, open(args.cache, "wb"))

    tr, te = scene_split(rows)
    ytr = np.array([r["y"] for r in tr])
    yte = np.array([r["y"] for r in te])

    counts = np.bincount(yte, minlength=len(ACTIONS))
    prior = counts.max() / counts.sum()
    print(f"\ntest class balance: " +
          ", ".join(f"{a}={c}" for a, c in zip(ACTIONS, counts)))

    results = {"prior (majority class)": (prior, None)}

    print("\nextracting features...")
    model = load_dinov2(args.dinov2_repo, args.dinov2_weights)
    results["dinov2 ViT-S/14"] = logreg(feats_dinov2(tr, model), ytr, feats_dinov2(te, model), yte)
    del model
    torch.cuda.empty_cache()
    if args.dinov2_weights_large:
        model = load_dinov2(args.dinov2_repo, args.dinov2_weights_large, args.dinov2_arch_large)
        results["dinov2 ViT-B/14"] = logreg(feats_dinov2(tr, model), ytr, feats_dinov2(te, model), yte)
        del model
        torch.cuda.empty_cache()
    results["occupancy crop"] = logreg(feats_occ(tr), ytr, feats_occ(te), yte)
    results["rgb crop (no encoder)"] = logreg(feats_rgb_raw(tr), ytr, feats_rgb_raw(te), yte)

    print(f"\n{'arm':<26}{'test acc':>10}   per-class recall")
    for k, (acc, per) in results.items():
        ps = "" if per is None else "  ".join(f"{a}={v:.2f}" for a, v in per.items())
        print(f"{k:<26}{acc*100:>9.1f}%   {ps}")


if __name__ == "__main__":
    main()
