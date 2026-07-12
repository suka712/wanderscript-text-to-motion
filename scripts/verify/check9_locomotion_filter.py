#!/usr/bin/env python3
"""
STEP2 Task 3: run the locomotion filter (src/humanise_join.locomotion_filter)
over ALL 19,648 HUMANISE clips and log the per-category breakdown + surviving
count. Keep walk / stand up; drop sit / lie, per CLAUDE.md's locomotion-only
MVP scope and STEP1b's reconstruction-quality findings.

Also writes the kept motion_id list to
scratch_outputs/humanise_locomotion_ids.txt (gitignored) so it can be reused
directly as a filtered split file by Step 3 without re-deriving it.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import humanise_join as hj

if __name__ == "__main__":
    kept, breakdown = hj.locomotion_filter()

    print("Per-action breakdown (all 19,648 clips):")
    for a in hj.ACTIONS:
        n = breakdown[a]
        tag = "KEEP" if a in hj.LOCOMOTION_ACTIONS else "drop"
        print(f"  {a:>10s}: {n:6d}  [{tag}]")

    print("-" * 40)
    print(f"  locomotion_total (kept): {breakdown['locomotion_total']:6d}")
    print(f"  dropped_total (sit+lie): {breakdown['dropped_total']:6d}")
    print(f"  all_total:               {breakdown['all_total']:6d}")
    pct = 100.0 * breakdown["locomotion_total"] / breakdown["all_total"]
    print(f"  surviving fraction:      {pct:.2f}%")

    assert len(kept) == breakdown["locomotion_total"]
    assert breakdown["all_total"] == 19648, (
        f"expected 19648 total clips, got {breakdown['all_total']}"
    )

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "scratch_outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "humanise_locomotion_ids.txt")
    with open(out_path, "w") as f:
        for idx in kept:
            f.write(f"{idx:05d}\n")
    print(f"\nWrote {len(kept)} kept motion_ids to {out_path}")
    print("VERDICT: PASS" if len(kept) > 0 else "VERDICT: FAIL - no clips survived")
