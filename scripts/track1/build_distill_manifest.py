#!/usr/bin/env python3
"""Path B / Stage 1 / Step 0 — build the distillation finetune manifest.

Combine the real step10 manifest (preserves clean walking + interaction so the finetune does
NOT forget them) with the guided_seg-distilled avoidance demos (distill_guided.py), oversampling
the distilled set to a target FRACTION so its scene-avoidance gradient is not swamped by the 15.6k
real clips. Writes a train.pkl in a new tokens dir; feed it to train_probe.py --init-ckpt
step10/goalaug (which reuses goalaug's cond norm — see that flag).

Distilled entries carry no xy_traj, so goal-aug (walk-only, needs xy_traj) skips them (their tokens
are guided winners and must not be re-truncated); real walk clips keep xy_traj so goal-aug still
preserves arbitrary-goal following. Extra metadata keys (seg_coll/greedy_coll/goal_err) are harmless
— ProbeMotionDataset reads only the cond fields.
"""
import argparse
import os
import pickle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=os.path.expanduser("~/wander_data/step10/tokens/train.pkl"))
    ap.add_argument("--distill", default=os.path.expanduser("~/wander_data/pathb/distill_stage1/distill.pkl"))
    ap.add_argument("--out-dir", default=os.path.expanduser("~/wander_data/pathb/distill_tokens"))
    ap.add_argument("--distill-frac", type=float, default=0.4,
                    help="target fraction of the combined manifest that is distilled avoidance demos")
    ap.add_argument("--only-avoidance", action="store_true",
                    help="keep only distilled segments where the guided winner beat greedy on "
                         "collision (concentrates the avoidance signal; drops clear-walk demos)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    real = pickle.load(open(args.real, "rb"))
    dist = pickle.load(open(args.distill, "rb"))
    if args.only_avoidance:
        dist = [d for d in dist if d.get("seg_coll", 1) < d.get("greedy_coll", 0) - 1e-9]
    if not dist:
        raise SystemExit("no distilled segments after filtering")

    # K repeats of the distilled set so distilled/(distilled+real) ~= distill_frac
    f = args.distill_frac
    K = max(1, round(f * len(real) / ((1 - f) * len(dist))))
    combined = list(real) + dist * K
    frac = len(dist) * K / len(combined)

    out = os.path.join(args.out_dir, "train.pkl")
    with open(out, "wb") as fh:
        pickle.dump(combined, fh)
    print(f"real={len(real)}  distilled={len(dist)} (only_avoidance={args.only_avoidance})  "
          f"K={K}  -> combined={len(combined)}  distilled_frac={frac:.2f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
