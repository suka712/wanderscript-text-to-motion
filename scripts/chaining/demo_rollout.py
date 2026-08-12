#!/usr/bin/env python3
"""Free multi-segment rollout in a real ScanNet room — the demo, and the first
setting where scene conditioning is measurable.

Unlike eval_accumulation.py, goals here are NOT taken from a ground-truth clip.
They are waypoints chosen in the scene, so the chain walks a path longer than
any single HUMANISE segment. That matters twice over:

  - it is the demo (done-criteria 4 and 5: multi-segment motion, watchable mp4)
  - it is the only setting where the scene arm can be evaluated. RESULTS §8
    found every step-8 metric empty because HUMANISE segments average 0.63m and
    only 0.8% walk >1.5m; collision was a property of the start pose. Chained
    rollouts produce multi-metre paths, so collision finally reflects the path.

Waypoints are sampled on FREE floor space from the walkable map and spaced at
least `--min-step` apart, so the chain has to cross the room rather than
shuffle in place. Collision is scored on the 0.9m TALL-obstacle map (RESULTS
§8) -- never the 0.12m one, which counts sitting on the target as a collision.
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
from rollout import rollout, load_model  # noqa: E402

T2M = os.environ.get("WANDER_T2M_GPT_ROOT")
HUMANISE = os.environ.get("WANDER_HUMANISE_ROOT")
BEV = os.path.expanduser("~/wander_data/bev_cache")
TALL = os.path.expanduser("~/wander_data/bev_tall_cache")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def px_to_world(rc, extent, shape):
    xmin, xmax, ymin, ymax = extent
    H, W = shape
    r, c = rc
    return np.array([xmin + (c + .5) / W * (xmax - xmin),
                     ymax - (r + .5) / H * (ymax - ymin)])


def sample_waypoints(occ, extent, start_xy, n, min_step, rng, margin_px=12, max_step=None):
    """n free-space waypoints, each within [min_step, max_step] of the previous.

    The BAND matters, not just the lower bound. Sampling "anywhere beyond
    min_step" in a room yields ~4m hops regardless of min_step, and the model is
    trained on 0.63m mean displacement -- it then walks ~1.3m and stops, giving
    2.5m goal error that says nothing about chaining. Waypoints must be spaced at
    the per-segment displacement the model was actually trained on."""
    free = occ < 0.5
    try:
        from scipy import ndimage
        free = ndimage.binary_erosion(free, np.ones((margin_px, margin_px)))
    except ImportError:
        pass
    rr, cc = np.nonzero(free)
    if len(rr) < 50:
        return None
    pts = np.stack([px_to_world((r, c), extent, occ.shape) for r, c in zip(rr, cc)])
    out, cur = [], np.asarray(start_xy, float)
    for _ in range(n):
        d = np.linalg.norm(pts - cur, axis=1)
        hi = max_step if max_step else min_step * 1.6
        cand = np.nonzero((d > min_step) & (d <= hi))[0]
        if len(cand) == 0:                      # relax the ceiling before giving up
            cand = np.nonzero(d > min_step)[0]
        if len(cand) == 0:
            break
        pick = pts[cand[rng.randint(len(cand))]]
        out.append(pick)
        cur = pick
    return out if len(out) == n else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vqvae-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-rollouts", type=int, default=8)
    ap.add_argument("--n-segments", type=int, default=6)
    ap.add_argument("--min-step", type=float, default=0.6)
    ap.add_argument("--max-step", type=float, default=1.2,
                    help="upper bound on per-segment hop. Keep near the trained "
                         "displacement scale (HUMANISE mean 0.63m); far goals are "
                         "out of distribution and the model simply stops short.")
    ap.add_argument("--real-text", action="store_true",
                    help="use the clip's own utterance instead of a generic string")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    net = load_vqvae(ckpt_path=args.vqvae_ckpt, device=DEV); net.eval()
    mean = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy").astype(np.float32)
    std = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy").astype(np.float32)
    cmodel, _ = clip.load("ViT-B/32", device=DEV, jit=False); cmodel.eval()
    trans, ns = load_model(args.ckpt)

    flat = build_flat_join()
    walk_idx = [i for i, p in enumerate(flat) if p["action"] == "walk"]
    rng = np.random.RandomState(args.seed)
    rng.shuffle(walk_idx)

    results, done = [], 0
    for idx in walk_idx:
        if done >= args.n_rollouts:
            break
        rec = get_record(int(idx))
        fb, ft = os.path.join(BEV, f"{rec.scene}.npz"), os.path.join(TALL, f"{rec.scene}.npz")
        if not (os.path.exists(fb) and os.path.exists(ft)):
            continue
        zb, zt = np.load(fb), np.load(ft)
        occ, extent = zb["occ"].astype(np.float32), zb["extent"]
        tall = zt["occ"].astype(np.float32)

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
        # Diagnostic: does goal-following depend on the text being in
        # distribution? Training text is the clip's real utterance; a generic
        # string is OOD and may be why free-waypoint goals are not followed.
        texts = ([rec.utterance] if args.real_text else ["walk to the target"]) * args.n_segments
        segs = rollout(trans, net, cmodel, clip, mean, std, ns, texts, wps,
                       start_pose, prefix, occ, extent)
        if len(segs) < args.n_segments:
            continue

        path = np.concatenate([s["world"][:, 0, :2] for s in segs])
        xmin, xmax, ymin, ymax = extent
        H, W = tall.shape
        c = np.clip(((path[:, 0] - xmin) / (xmax - xmin) * W).astype(int), 0, W - 1)
        r = np.clip(((ymax - path[:, 1]) / (ymax - ymin) * H).astype(int), 0, H - 1)
        coll = float((tall[r, c] > .5).mean())
        # STRAIGHT-LINE control: walk the waypoint polyline directly. Without it
        # a 2% collision rate is uninterpretable -- it says nothing about whether
        # the model steered around anything or simply took the direct route.
        poly = [start_pose[:2].astype(float)]
        for w in wps:
            a0, b0 = poly[-1], np.asarray(w, float)
            k = max(2, int(np.linalg.norm(b0 - a0) / 0.05))
            poly += [a0 + (b0 - a0) * t for t in np.linspace(0, 1, k)[1:]]
        poly = np.stack(poly)
        pcx = np.clip(((poly[:, 0] - xmin) / (xmax - xmin) * W).astype(int), 0, W - 1)
        pry = np.clip(((ymax - poly[:, 1]) / (ymax - ymin) * H).astype(int), 0, H - 1)
        coll_line = float((tall[pry, pcx] > .5).mean())
        dist = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        commanded = float(np.sum([np.linalg.norm(np.asarray(wps[i]) - (start_pose[:2] if i == 0 else wps[i-1]))
                                  for i in range(len(wps))]))
        results.append({"scene": rec.scene, "idx": int(idx), "segs": segs, "wps": wps,
                        "occ": occ, "tall": tall, "extent": extent, "coll": coll,
                        "dist": dist, "commanded": commanded, "coll_line": coll_line,
                        "goal": float(np.mean([s["goal_err"] for s in segs])),
                        "seam": float(np.mean([s["seam_err"] for s in segs[1:]]))})
        done += 1
        print(f"  rollout {done}/{args.n_rollouts} {rec.scene} "
              f"path {dist:.1f}m (commanded {commanded:.1f}m) coll {coll*100:.1f}%", flush=True)

    if not results:
        print("no rollouts produced"); return
    np.save(os.path.join(args.out, "rollouts.npy"),
            np.array([{k: v for k, v in r.items() if k not in ("occ", "tall")} for r in results], dtype=object),
            allow_pickle=True)
    print(f"\n{len(results)} rollouts of {args.n_segments} segments")
    print(f"  path length      {np.mean([r['dist'] for r in results]):.2f} m "
          f"(commanded {np.mean([r['commanded'] for r in results]):.2f} m)")
    print(f"  collision (0.9m) {np.mean([r['coll'] for r in results])*100:.2f}%"
          f"   [straight-line control: {np.mean([r['coll_line'] for r in results])*100:.2f}%]")
    print(f"  goal err/segment {np.mean([r['goal'] for r in results]):.3f} m")
    print(f"  seam err         {np.mean([r['seam'] for r in results])*1000:.1f} mm")


if __name__ == "__main__":
    main()
