#!/usr/bin/env python3
"""Add a target-HEADING field to an existing token manifest (Step 11 heading fix).

The model conditions on goal POSITION only, so it reaches goals by translating the root
without turning to face travel — free chains "moonwalk" (RESULTS §11 "Known limitation",
diag_heading.py). The fix is to also condition on where the BODY should FACE. Body facing
(hip/shoulder yaw) is orientation information the (x,y) goal does not carry, so it is a
non-redundant supervised signal — that is why we use the actual yaw, not the travel
direction (which is derivable from the goal and would be ignored).

This adds, per record, WITHOUT re-tokenizing (pure geometry from compute_track2):
  goal_heading (2,)  = (sin, cos) of (end_yaw - start_yaw), the facing at the segment end
                       expressed relative to its start heading. Frame-independent: it is the
                       net rotation the body makes, which is exactly what the canonicalized
                       generation must produce (the SE(2) start-yaw offset cancels).
  head_traj  (Tb,2)  = the same per frame, so truncation goal-aug can pick the heading at
                       the truncated endpoint too.

Alignment is exact: xy_traj is a contiguous slice of compute_track2's xy, so we recover the
frame offset by matching xy_traj[0] and read sincos over that window.
"""
import argparse
import os
import pickle
import sys

import numpy as np

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from humanise_join import get_record, compute_track2  # noqa: E402


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def process(records, name):
    _t2_cache = {}
    ok, bad, travel_err = 0, 0, []
    for r in records:
        idx = int(r["index"])
        if idx not in _t2_cache:
            if len(_t2_cache) > 4000:
                _t2_cache.clear()
            _, xy, _, sincos = compute_track2(get_record(idx))
            _t2_cache[idx] = (xy, sincos)
        xy, sincos = _t2_cache[idx]
        xt = np.asarray(r["xy_traj"], np.float64)
        Tb = len(xt)
        t = int(np.linalg.norm(xy - xt[0], axis=1).argmin())
        if t + Tb > len(xy) or np.linalg.norm(xy[t:t + Tb] - xt, axis=1).mean() > 0.02:
            bad += 1
            # fallback: face the net travel direction (still better than nothing)
            trav = np.arctan2(xt[-1, 1] - xt[0, 1], xt[-1, 0] - xt[0, 0])
            start_yaw = np.arctan2(r["start"][2], r["start"][3])
            rel = wrap(trav - start_yaw)
            r["goal_heading"] = np.array([np.sin(rel), np.cos(rel)], np.float32)
            r["head_traj"] = np.tile(r["goal_heading"], (Tb, 1)).astype(np.float32)
            continue
        start_yaw = np.arctan2(r["start"][2], r["start"][3])
        seg_yaw = np.arctan2(sincos[t:t + Tb, 0], sincos[t:t + Tb, 1])
        rel = wrap(seg_yaw - start_yaw)
        head_traj = np.stack([np.sin(rel), np.cos(rel)], axis=1).astype(np.float32)
        r["head_traj"] = head_traj
        r["goal_heading"] = head_traj[-1].copy()
        ok += 1
        # sanity for walk clips: does the body's END facing match its travel direction?
        if r["action"] == "walk" and Tb >= 6 and np.linalg.norm(xt[-1] - xt[0]) > 0.3:
            trav = np.arctan2(xt[-1, 1] - xt[0, 1], xt[-1, 0] - xt[0, 0])
            end_face = seg_yaw[-1]
            travel_err.append(abs(np.degrees(wrap(end_face - trav))))
    print(f"{name}: {ok} ok, {bad} fell back to travel-direction")
    if travel_err:
        te = np.array(travel_err)
        print(f"  GT walk end-facing vs travel: mean {te.mean():.0f}deg, median "
              f"{np.median(te):.0f}deg, frac<30deg {100*(te<30).mean():.0f}%  "
              f"(small => GT people DO face travel, so commanding facing=travel is right)")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for split in ["train", "test"]:
        with open(os.path.join(args.in_dir, f"{split}.pkl"), "rb") as f:
            recs = pickle.load(f)
        recs = process(recs, split)
        with open(os.path.join(args.out_dir, f"{split}.pkl"), "wb") as f:
            pickle.dump(recs, f)
        print(f"wrote {os.path.join(args.out_dir, f'{split}.pkl')} ({len(recs)} records)")


if __name__ == "__main__":
    main()
