#!/usr/bin/env python3
"""
Task 1 / Track 1, run-at-scale validation.

Runs the humanise_positions_to_263 adapter (src/motion_features.py, which
wraps the UNMODIFIED HumanML3D process_file extractor) over every clip in
HUMANISE/contact_motion/motions/*.npy (19,648 files) and logs:
  - success / failure counts
  - NaN outputs
  - clips too short to process (< 3 frames; process_file needs >= 2 frames
    after its internal off-by-one slicing, we use 3 as a safety margin)

This is the "confirm the adapter doesn't silently poison a subset of the
dataset" check called for in STEP1b_extend.md Task 1 -- joint order and axis
convention were already confirmed on samples (see src/motion_features.py
docstring); this check is about coverage, not correctness-in-principle.
"""
import sys
import time
import warnings
import numpy as np

sys.path.insert(0, "/home/user/Khiem-ssh/wander/src")
import motion_features as mf  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = "/media/user/2tb/motion_data/HUMANISE/contact_motion/motions"
N = 19648

if __name__ == "__main__":
    t0 = time.time()
    n_ok = 0
    n_nan = 0
    n_fail = 0
    n_short = 0
    fail_examples = []
    for i in range(N):
        cm = np.load(f"{ROOT}/{i:05d}.npy")
        if cm.shape[0] < 3:
            n_short += 1
            continue
        try:
            data263, gp, lp, lv = mf.humanise_positions_to_263(cm)
            if np.isnan(data263).any() or np.isinf(data263).any():
                n_nan += 1
                if len(fail_examples) < 10:
                    fail_examples.append((i, "nan_or_inf"))
            else:
                n_ok += 1
        except Exception as e:
            n_fail += 1
            if len(fail_examples) < 10:
                fail_examples.append((i, repr(e)))
        if i % 2000 == 0:
            print(f"...{i}/{N} ({time.time()-t0:.0f}s)", flush=True)

    dt = time.time() - t0
    print("=" * 60)
    print(f"Total clips:        {N}")
    print(f"OK (finite 263-d):  {n_ok}  ({100*n_ok/N:.2f}%)")
    print(f"Too short (<3 fr):  {n_short}")
    print(f"NaN/Inf output:     {n_nan}")
    print(f"Exceptions:         {n_fail}")
    print(f"Elapsed:            {dt:.1f}s")
    print("Examples of failures/short/nan (up to 10):", fail_examples)
