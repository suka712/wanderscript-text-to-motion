#!/usr/bin/env python
"""
Combine TRUMANS + step10 (HUMANISE) manifests for training.

By default, merges both with TRUMANS oversampled to match the interaction-class
balance: step10 has 15632 clips but TRUMANS adds crucial seat-height diversity.
Goal-augmentation still works on walk clips (needs xy_traj); TRUMANS walk/stand
clips have xy_traj. Action distribution is printed for verification.

Usage:
  python build_combined_manifest.py \
    --step10 ~/wander_data/step10/tokens/train.pkl \
    --trumans ~/wander_data/trumans_tokens/train.pkl \
    --out-dir ~/wander_data/trumans_combined_tokens \
    [--trumans-frac 0.30]
"""
import argparse
import os
import pickle
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step10", default=os.path.expanduser(
        "~/wander_data/step10/tokens/train.pkl"))
    ap.add_argument("--trumans", default=os.path.expanduser(
        "~/wander_data/trumans_tokens/train.pkl"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--trumans-frac", type=float, default=0.30,
                    help="target fraction of combined that is TRUMANS (default 0.30)")
    ap.add_argument("--interaction-only", action="store_true",
                    help="only include TRUMANS sit/lie/stand clips (not walk)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    step10 = pickle.load(open(args.step10, "rb"))
    trumans = pickle.load(open(args.trumans, "rb"))

    if args.interaction_only:
        trumans = [d for d in trumans if d.get("action") in ("sit", "lie", "stand up")]
        print("TRUMANS filtered to interaction-only: %d clips" % len(trumans))

    # Print action distributions
    s10_actions = Counter(d.get("action", "?") for d in step10)
    tr_actions = Counter(d.get("action", "?") for d in trumans)
    print("step10 (%d): %s" % (len(step10), dict(s10_actions)))
    print("TRUMANS (%d): %s" % (len(trumans), dict(tr_actions)))

    # Compute oversampling factor K
    f = args.trumans_frac
    K = max(1, round(f * len(step10) / ((1 - f) * len(trumans))))
    combined = list(step10) + list(trumans) * K
    actual_frac = len(trumans) * K / len(combined)

    # Reindex
    for i, d in enumerate(combined):
        d["_orig_index"] = d.get("index", i)

    out_path = os.path.join(args.out_dir, "train.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(combined, f)

    print("\nCombined: step10=%d + TRUMANS=%d (x%d=%d) = %d total" %
          (len(step10), len(trumans), K, len(trumans)*K, len(combined)))
    print("TRUMANS fraction: %.1f%% (target %.1f%%)" %
          (actual_frac * 100, args.trumans_frac * 100))

    # Combined action distribution
    c_actions = Counter(d.get("action", "?") for d in combined)
    print("Combined actions: %s" % dict(c_actions))
    print("Saved: %s" % out_path)


if __name__ == "__main__":
    main()
