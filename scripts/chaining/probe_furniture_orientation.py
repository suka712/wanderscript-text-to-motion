#!/usr/bin/env python3
"""Can furniture orientation be PERCEIVED from scene geometry (no GT)? (Step 11, sit-facing)

The genuine fix for wrong sit orientation must read the furniture's facing from the SCENE
(works on a novel room), not from the seed clip's ground truth (the rejected hack). This tests
whether the geometry we already have carries it: a backrest is a tall mass behind the seat, so
"face AWAY from the nearby tall obstacle" (the 0.9 m raster) should approximate the seated facing.

Reports, over sit clips whose GT interacts:
  perceived vs GT   |away-from-tall-mass  -  GT seated facing|   -- is the signal real?
  approach vs GT    |clip start facing     -  GT seated facing|   -- how much people rotate to sit
                    (small => the model can copy the approach; large => it must turn, which one
                     short segment may not allow -- the per-segment turning cap, RESULTS §11)
No model, no GPU -- pure geometry. GT is used only to SCORE the perceiver (legitimate), never
to drive anything.
"""
import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from humanise_join import build_flat_join, get_record, compute_track2  # noqa: E402
import motion_features as mf  # noqa: E402
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "track1"))
from se2_utils import yup_to_zup  # noqa: E402
from humanise_join import J_PELVIS  # noqa: E402

HUMANISE = os.environ.get("WANDER_HUMANISE_ROOT")
BEV = os.path.expanduser("~/wander_data/bev_cache")
TALL = os.path.expanduser("~/wander_data/bev_tall_cache")
SIT_Z = 0.70


def wrapdeg(a):
    return abs((np.degrees(a) + 180) % 360 - 180)


def perceive_facing(tall, extent, sit_xy, r_m=0.7, min_cells=4):
    """Seated facing = away from the centroid of tall (>=0.9 m) mass within r_m of the seat.
    A backrest / adjacent wall is that mass; you sit facing away from it. None if no tall
    mass nearby (orientation genuinely ambiguous -- e.g. a backless stool)."""
    xmin, xmax, ymin, ymax = extent
    H, W = tall.shape
    rr, cc = np.nonzero(tall > 0.5)
    if len(rr) == 0:
        return None
    wx = xmin + (cc + 0.5) / W * (xmax - xmin)
    wy = ymax - (rr + 0.5) / H * (ymax - ymin)
    d = np.hypot(wx - sit_xy[0], wy - sit_xy[1])
    m = d <= r_m
    if m.sum() < min_cells:
        return None
    w = 1.0 / (d[m] + 0.1)                       # nearer mass dominates (the backrest)
    back = np.array([np.sum(w * (wx[m] - sit_xy[0])), np.sum(w * (wy[m] - sit_xy[1]))])
    if np.linalg.norm(back) < 1e-6:
        return None
    face = -back                                  # away from the backrest
    return float(np.arctan2(face[1], face[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--radius", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    flat = build_flat_join()
    idxs = [i for i, p in enumerate(flat) if p["action"] == "sit"]
    rng = np.random.RandomState(args.seed); rng.shuffle(idxs)
    perc, appr = [], []
    caches = {}
    used = ambiguous = 0
    for idx in idxs:
        if used >= args.n:
            break
        rec = get_record(int(idx))
        fb, ft = os.path.join(BEV, f"{rec.scene}.npz"), os.path.join(TALL, f"{rec.scene}.npz")
        if not (os.path.exists(fb) and os.path.exists(ft)):
            continue
        cm = np.load(os.path.join(HUMANISE, "contact_motion", "motions", f"{idx:05d}.npy"))
        if cm.shape[0] < 8:
            continue
        d263, *_ = mf.humanise_positions_to_263(cm)
        if yup_to_zup(mf.recover_positions(d263.astype(np.float32)))[-1, J_PELVIS, 2] >= SIT_Z:
            continue
        _, xy, _, sincos = compute_track2(rec)
        gt_yaw = float(np.arctan2(sincos[-1, 0], sincos[-1, 1]))
        appr_yaw = float(np.arctan2(sincos[0, 0], sincos[0, 1]))
        sit_xy = xy[-1].astype(np.float64)
        if rec.scene not in caches:
            if len(caches) > 60:
                caches.clear()
            zb = np.load(fb); zt = np.load(ft)
            caches[rec.scene] = (zt["occ"].astype(np.float32), zb["extent"])
        tall, extent = caches[rec.scene]
        pf = perceive_facing(tall, extent, sit_xy, r_m=args.radius)
        used += 1
        appr.append(wrapdeg(appr_yaw - gt_yaw))
        if pf is None:
            ambiguous += 1
        else:
            perc.append(wrapdeg(pf - gt_yaw))

    perc = np.array(perc); appr = np.array(appr)
    print(f"\n=== furniture-orientation perception, {used} sit clips "
          f"({ambiguous} had no nearby tall mass = ambiguous) ===")
    print(f"PERCEIVED (away from tall mass) vs GT seated facing:")
    print(f"   mean {perc.mean():.0f}deg  median {np.median(perc):.0f}deg  "
          f"<45deg {100*(perc<45).mean():.0f}%  <90deg {100*(perc<90).mean():.0f}%  (n={len(perc)})")
    print(f"   (random baseline would be ~90deg mean, 25% under 45deg)")
    print(f"APPROACH (clip start facing) vs GT seated facing:")
    print(f"   mean {appr.mean():.0f}deg  median {np.median(appr):.0f}deg  "
          f"<45deg {100*(appr<45).mean():.0f}%  (how much people rotate to sit; small => low turn)")


if __name__ == "__main__":
    main()
