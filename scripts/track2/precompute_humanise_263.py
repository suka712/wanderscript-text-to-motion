#!/usr/bin/env python3
"""
Track 2 prerequisite: precompute the 263-dim HumanML3D feature vector for every
HUMANISE clip (all 19,648 -- both train.txt and test.txt need it, train for the
joint finetune, test for held-out per-category eval) and cache to disk.

Why cache: src/motion_features.humanise_positions_to_263 goes through the
unmodified HumanML3D process_file() extractor, ~33ms/clip single-threaded
(measured on this box) -- fine for a one-off precompute (~11 min for 19,648)
but far too slow to redo every dataloader __getitem__ across a multi-thousand-
iteration finetune.

Resumable: skips any {idx:05d}.npy already present in the cache dir, so a
killed/restarted run just picks up where it left off.

Cache dir defaults to /media/user/2tb/motion_data/HUMANISE_263_cache (data-like
artifact, lives with the rest of motion_data on the big disk -- NOT under this
repo's scratch_outputs/, override with WANDER_HUMANISE_263_CACHE if needed).
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import motion_features as mf  # noqa: E402

HUMANISE_ROOT = os.environ.get("WANDER_HUMANISE_ROOT", "/media/user/2tb/motion_data/HUMANISE")
HUMANISE_MOTIONS = f"{HUMANISE_ROOT}/contact_motion/motions"
CACHE_DIR = os.environ.get("WANDER_HUMANISE_263_CACHE", "/media/user/2tb/motion_data/HUMANISE_263_cache")
N_TOTAL = 19648


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    t0 = time.time()
    n_done = 0
    n_skipped_existing = 0
    n_skipped_short = 0
    n_failed = 0
    failed_ids = []

    for i in range(N_TOTAL):
        out_path = f"{CACHE_DIR}/{i:05d}.npy"
        if os.path.exists(out_path):
            n_skipped_existing += 1
            continue
        cm = np.load(f"{HUMANISE_MOTIONS}/{i:05d}.npy")
        if cm.shape[0] < 6:
            n_skipped_short += 1
            continue
        try:
            data263, _, _, _ = mf.humanise_positions_to_263(cm)
        except Exception as e:  # noqa: BLE001
            n_failed += 1
            failed_ids.append((i, str(e)))
            continue
        data263 = data263.astype(np.float32)
        if not np.isfinite(data263).all():
            n_failed += 1
            failed_ids.append((i, "non-finite output"))
            continue
        np.save(out_path, data263)
        n_done += 1

        if (n_done + n_skipped_existing) % 1000 == 0:
            elapsed = time.time() - t0
            print(
                f"[{elapsed:7.1f}s] processed {i + 1}/{N_TOTAL}  "
                f"(new={n_done} cached_skip={n_skipped_existing} "
                f"too_short={n_skipped_short} failed={n_failed})",
                flush=True,
            )

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"DONE in {elapsed:.1f}s")
    print(f"newly converted: {n_done}")
    print(f"already cached (resumed): {n_skipped_existing}")
    print(f"skipped (too short <6 frames): {n_skipped_short}")
    print(f"failed (exception / non-finite): {n_failed}")
    if failed_ids:
        print("Failed ids (first 20):", failed_ids[:20])
    total_cached = n_done + n_skipped_existing
    print(f"Total cached: {total_cached}/{N_TOTAL} ({100.0 * total_cached / N_TOTAL:.2f}%)")


if __name__ == "__main__":
    main()
