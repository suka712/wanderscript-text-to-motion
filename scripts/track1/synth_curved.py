#!/usr/bin/env python3
"""Path B / Stage 1 / Step 1 (FEASIBILITY PROBE) — synthetic scene-CAUSED curved walks.

Step 0 failed because guided_seg avoidance is stochastic-selection luck: for a fixed occ_crop the
avoiding path is one lucky draw, so (occ_crop -> avoiding tokens) is not a function and distillation
can't learn it (memory path-b-scene-aware-plan). This makes the avoiding path CAUSED by the scene,
so occ_crop -> avoidance IS a function the model can learn.

METHOD. Real curved walk motion is on-manifold for free: HumanML3D new_joint_vecs ARE native 263
(the VQ-VAE's own training format; HUMANISE lacks curves — the whole problem). For each curved
window we SEARCH an SE(2) placement in a ScanNet scene where the CURVE stays in free space but the
straight CHORD (start->goal) CLIPS an obstacle. Then occ_crop(start) genuinely shows the obstacle
the motion routes around, the same way every time -> a learnable function. Plus STRAIGHT windows
placed in OPEN space (occ empty -> straight target) as negatives, so the model learns
obstacle->curve AND no-obstacle->straight (not 'always curve').

Placement uses the validated se2_place_full_body (handles the load-bearing yaw+pi/2 offset).
Tokenize with the SAME finetuned VQ-VAE the model decodes with, T2M-meta normalized (RESULTS §2).
Training data placed on GEN scenes only; eval (distill_eval) uses the HELD-OUT scenes -> tests
generalisation. Output schema = step10 train.pkl, so build_distill_manifest + train_probe
--init-ckpt + distill_eval are reused unchanged.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
for p in ["src", "scripts/track1", "scripts/chaining"]:
    sys.path.insert(0, os.path.join(REPO_ROOT, p))

import motion_features as mf  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402
from se2_utils import local_xy_trajectory, start_pose_rotation  # noqa: E402
from rollout import occ_crop as occ_crop_fn  # noqa: E402
from collision_guided import path_collision, straight_line_collision  # noqa: E402
from distill_guided import T2M, BEV, TALL  # noqa: E402

H3D = os.path.join(os.environ["WANDER_MOTION_DATA_ROOT"], "H3D")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def lateral_deviation(path_xy):
    """Max signed perpendicular offset of the (T,2) path from its own start->end chord.
    A curved (avoidance-capable) window bows to one side, leaving a notch on the chord for an
    obstacle. Returns (max_abs_dev, displacement)."""
    a, b = path_xy[0], path_xy[-1]
    chord = b - a
    L = np.linalg.norm(chord)
    if L < 1e-6:
        return 0.0, 0.0
    n = np.array([-chord[1], chord[0]]) / L      # unit normal to the chord
    dev = (path_xy - a) @ n                        # signed perpendicular distance per frame
    return float(np.max(np.abs(dev))), float(L)


def encode_tokens(net, mean, std, win263):
    """263 window -> VQ-VAE code indices (T2M-meta normalized; RESULTS §2)."""
    norm = (win263 - mean) / std
    x = torch.from_numpy(norm).float().unsqueeze(0).to(DEV)
    with torch.no_grad():
        return net.encode(x)[0].cpu().numpy().astype(np.int64)


def free_cells(occ, extent):
    """World xy of free (occ<0.5) cells, lightly eroded so placements are reachable. Vectorized."""
    from scipy import ndimage
    free = ndimage.binary_erosion(occ < 0.5, np.ones((6, 6)))
    rr, cc = np.nonzero(free)
    if len(rr) < 50:
        return None
    xmin, xmax, ymin, ymax = extent
    H, W = occ.shape
    wx = xmin + (cc + 0.5) / W * (xmax - xmin)
    wy = ymax - (rr + 0.5) / H * (ymax - ymin)
    return np.stack([wx, wy], 1)


def _place_root(cxy, sxy, yaw):
    """Cheap world root path for a canonical pelvis path cxy placed at (sxy, yaw) -- the SAME
    SE(2) compose as se2_place, but with recover_positions hoisted OUT of the search loop."""
    R = start_pose_rotation((sxy[0], sxy[1], np.sin(yaw), np.cos(yaw)))
    return cxy @ R.T + np.asarray(sxy)


def place_positive(cxy, prefix_pose, cells, occ, extent, tall, rng, n_try, curve_max, chord_min):
    """Find a placement where the CURVE is free but the CHORD is blocked. Best = highest chord
    collision among curve-free hits. Returns (record, chord_cl, curve_cl) or None."""
    best = None
    for _ in range(n_try):
        sxy = cells[rng.randint(len(cells))]
        yaw = rng.uniform(-np.pi, np.pi)
        pth = _place_root(cxy, sxy, yaw)
        curve_cl = path_collision(pth, tall, extent)
        if curve_cl > curve_max:
            continue
        chord_cl = straight_line_collision(sxy, [pth[-1]], tall, extent)
        if chord_cl < chord_min:
            continue
        if best is None or chord_cl > best[0]:
            best = (chord_cl, sxy, yaw, pth[-1], curve_cl)
    if best is None:
        return None
    chord_cl, sxy, yaw, goal, curve_cl = best
    return _record(prefix_pose, sxy, yaw, goal, occ, extent), chord_cl, curve_cl


def place_negative(cxy, prefix_pose, cells, occ, extent, tall, rng, n_try, free_max):
    """Place a STRAIGHT window in OPEN space (both curve and chord clear -> occ empty)."""
    for _ in range(n_try):
        sxy = cells[rng.randint(len(cells))]
        yaw = rng.uniform(-np.pi, np.pi)
        pth = _place_root(cxy, sxy, yaw)
        if path_collision(pth, tall, extent) > free_max:
            continue
        if straight_line_collision(sxy, [pth[-1]], tall, extent) > free_max:
            continue
        return _record(prefix_pose, sxy, yaw, pth[-1], occ, extent)
    return None


def _record(prefix_pose, sxy, yaw, goal, occ, extent):
    start_pose = np.array([sxy[0], sxy[1], np.sin(yaw), np.cos(yaw)], np.float32)
    return {
        "text": "walk to the target",
        "goal": np.asarray(goal, np.float32),
        "start": start_pose,
        "prefix_pose": prefix_pose,
        "occ_crop": occ_crop_fn(occ, extent, start_pose[:2], float(yaw)).astype(np.float32),
        "action": "walk",
    }


def iter_windows(ids, lengths, stride, rng):
    """Yield (clip_id, win263) windows from H3D clips (native 263), with VARIED length (drawn
    per window from `lengths`, all multiples of 4) so the synth segment-length distribution
    matches the real data instead of a single fixed length (a Step-1-probe over-fit source)."""
    lmin = min(lengths)
    for cid in ids:
        f = os.path.join(H3D, "new_joint_vecs", f"{cid}.npy")
        if not os.path.exists(f):
            continue
        try:
            d = np.load(f).astype(np.float32)
        except Exception:
            continue
        if d.shape[0] < lmin or d.shape[1] != 263:
            continue
        for s in range(0, d.shape[0] - lmin + 1, stride):
            L = lengths[rng.randint(len(lengths))]
            if s + L > d.shape[0]:
                L = lmin
            yield cid, d[s:s + L]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vqvae-ckpt", required=True)
    ap.add_argument("--eval-json", required=True, help="distill_stage1/eval_scenes.json (for GEN scenes)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-pos", type=int, default=400)
    ap.add_argument("--n-neg", type=int, default=200)
    ap.add_argument("--win-lengths", type=str, default="24,32,40,48",
                    help="comma list of window frame counts (each a multiple of 4); length drawn "
                         "per window so synth segment lengths match the real distribution")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--disp-lo", type=float, default=0.7)
    ap.add_argument("--disp-hi", type=float, default=1.6)
    ap.add_argument("--curve-dev", type=float, default=0.30, help="min lateral dev for a positive (m)")
    ap.add_argument("--straight-dev", type=float, default=0.10, help="max lateral dev for a negative (m)")
    ap.add_argument("--n-try", type=int, default=250, help="placement attempts per window")
    ap.add_argument("--curve-max", type=float, default=0.02, help="max curve collision (curve free)")
    ap.add_argument("--chord-min", type=float, default=0.10, help="min chord collision (chord blocked)")
    ap.add_argument("--free-max", type=float, default=0.02, help="max collision for a negative (open)")
    ap.add_argument("--n-scenes", type=int, default=60, help="subsample of gen scenes to place into")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    net = load_vqvae(ckpt_path=args.vqvae_ckpt, device=DEV); net.eval()
    meta = f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta"
    mean = np.load(f"{meta}/mean.npy").astype(np.float32)
    std = np.load(f"{meta}/std.npy").astype(np.float32)

    gen_scenes = json.load(open(args.eval_json))["gen_scenes"]  # place ONLY on gen scenes
    rng.shuffle(gen_scenes)
    gen_scenes = gen_scenes[:args.n_scenes]
    scene_npz = {}
    for sc in gen_scenes:
        fb, ft = os.path.join(BEV, f"{sc}.npz"), os.path.join(TALL, f"{sc}.npz")
        if os.path.exists(fb) and os.path.exists(ft):
            zb, zt = np.load(fb), np.load(ft)
            cells = free_cells(zb["occ"].astype(np.float32), zb["extent"])
            if cells is not None:
                scene_npz[sc] = (zb["occ"].astype(np.float32), zb["extent"], zt["occ"].astype(np.float32), cells)
    scenes = list(scene_npz)
    print(f"{len(scenes)} usable gen scenes; targeting {args.n_pos} pos + {args.n_neg} neg\n", flush=True)

    win_lengths = [int(x) for x in args.win_lengths.split(",")]
    ids = [l.strip() for l in open(os.path.join(H3D, "train.txt")) if l.strip()]
    rng.shuffle(ids)

    pos, neg = [], []
    n_curved_win = n_straight_win = 0
    for cid, win in iter_windows(ids, win_lengths, args.stride, rng):
        if len(pos) >= args.n_pos and len(neg) >= args.n_neg:
            break
        cxy = local_xy_trajectory(win, mf)          # recover_positions ONCE per window
        dev, disp = lateral_deviation(cxy)
        if not (args.disp_lo <= disp <= args.disp_hi):
            continue
        is_pos = dev >= args.curve_dev and len(pos) < args.n_pos
        is_neg = dev <= args.straight_dev and len(neg) < args.n_neg
        if not (is_pos or is_neg):
            continue
        prefix_pose = mf.local_joint_positions(win.astype(np.float32))[0].ravel().astype(np.float32)
        sc = scenes[rng.randint(len(scenes))]
        occ, extent, tall, cells = scene_npz[sc]
        if is_pos:
            n_curved_win += 1
            r = place_positive(cxy, prefix_pose, cells, occ, extent, tall, rng, args.n_try,
                               args.curve_max, args.chord_min)
            if r is not None:
                rec, chord_cl, curve_cl = r
                rec["tokens"] = encode_tokens(net, mean, std, win)
                rec.update(scene=sc, kind="pos", chord_coll=chord_cl, curve_coll=curve_cl)
                pos.append(rec)
                if len(pos) % 25 == 0:
                    print(f"  pos {len(pos)}/{args.n_pos}  (dev {dev:.2f} disp {disp:.2f} "
                          f"chord {chord_cl*100:.0f}% curve {curve_cl*100:.0f}%)", flush=True)
        else:
            n_straight_win += 1
            rec = place_negative(cxy, prefix_pose, cells, occ, extent, tall, rng, args.n_try, args.free_max)
            if rec is not None:
                rec["tokens"] = encode_tokens(net, mean, std, win)
                rec.update(scene=sc, kind="neg")
                neg.append(rec)
                if len(neg) % 25 == 0:
                    print(f"  neg {len(neg)}/{args.n_neg}", flush=True)

    recs = pos + neg
    import pickle
    with open(os.path.join(args.out, "synth.pkl"), "wb") as f:
        pickle.dump(recs, f)
    kept = [r for r in recs if r["tokens"].size > 0]
    print(f"\n=== {len(pos)} positives + {len(neg)} negatives = {len(recs)} synth segments ===")
    print(f"  curved windows tried {n_curved_win} -> {len(pos)} placed ({100*len(pos)/max(n_curved_win,1):.0f}%)")
    print(f"  straight windows tried {n_straight_win} -> {len(neg)} placed")
    if pos:
        print(f"  positives: chord coll {np.mean([r['chord_coll'] for r in pos])*100:.1f}%  "
              f"curve coll {np.mean([r['curve_coll'] for r in pos])*100:.1f}%")
    print(f"  token lens: {np.percentile([len(r['tokens']) for r in kept],[0,50,100]).astype(int)}")
    print(f"\nsaved {os.path.join(args.out, 'synth.pkl')}")


if __name__ == "__main__":
    main()
