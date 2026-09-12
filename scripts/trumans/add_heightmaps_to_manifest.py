#!/usr/bin/env python
"""
Add per-clip start-frame heightmaps to a training manifest.

For each entry, loads the precomputed (T, 32, 32) heightmap cache, takes frame 0
(the segment start), and stores it as a flattened 1024-dim vector `heightmap_1024`.
Entries without a heightmap file get a flat-floor (zeros) vector, same as H3D clips
in the SceMoS tokenizer pipeline (scene_joint_dataset.py).

Two data sources:
  - HUMANISE clips (from extract_heightmaps.py): ~/wander_data/motion_data/HUMANISE_heightmap_cache/{index:05d}.npy
  - TRUMANS clips (from trumans/extract_heightmaps.py): ~/wander_data/trumans_heightmap_cache/{clip_id}.npy

The script distinguishes them by the presence of a 'session' key (TRUMANS only).

Usage:
  python add_heightmaps_to_manifest.py \
    --manifest ~/wander_data/trumans_combined_tokens/train.pkl \
    --humanise-hm ~/wander_data/motion_data/HUMANISE_heightmap_cache \
    --trumans-hm ~/wander_data/trumans_heightmap_cache \
    --out ~/wander_data/trumans_combined_tokens/train_hm.pkl
"""
import argparse
import os
import pickle
import sys

import numpy as np

GRID_N = 32
HM_DIM = GRID_N * GRID_N  # 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Input train.pkl")
    ap.add_argument("--humanise-hm", required=True,
                    help="HUMANISE heightmap cache dir ({idx:05d}.npy)")
    ap.add_argument("--trumans-hm", required=True,
                    help="TRUMANS heightmap cache dir ({clip_id}.npy)")
    ap.add_argument("--out", required=True, help="Output manifest .pkl")
    args = ap.parse_args()

    manifest = pickle.load(open(args.manifest, "rb"))
    print("Input manifest: %d entries" % len(manifest))

    n_humanise = n_trumans = n_has_hm = n_flat = 0

    for entry in manifest:
        is_trumans = "session" in entry
        if is_trumans:
            n_trumans += 1
            # TRUMANS clip: look up by clip_id (string)
            # In the combined manifest, TRUMANS entries may be oversampled (duplicated),
            # but they still have the original clip_id in 'index' field.
            # The _orig_index field (from build_combined_manifest) preserves the original.
            # The 'index' field for TRUMANS is the clip_id as int.
            idx = entry.get("_orig_index", entry["index"])
            hm_path = os.path.join(args.trumans_hm, "%05d.npy" % idx)
        else:
            n_humanise += 1
            idx = entry["index"]
            hm_path = os.path.join(args.humanise_hm, "%05d.npy" % idx)

        if os.path.exists(hm_path):
            hm = np.load(hm_path, mmap_mode="r")  # (T, 32, 32)
            # Take frame 0 (segment start): what the body "sees" at the start
            hm0 = hm[0].astype(np.float32).ravel()  # (1024,)
            entry["heightmap_1024"] = hm0
            n_has_hm += 1
        else:
            # Flat floor (no scene geometry available — like H3D)
            entry["heightmap_1024"] = np.zeros(HM_DIM, dtype=np.float32)
            n_flat += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(manifest, f)

    print("\nDone: %d entries" % len(manifest))
    print("  HUMANISE: %d, TRUMANS: %d" % (n_humanise, n_trumans))
    print("  With heightmap: %d, flat floor: %d" % (n_has_hm, n_flat))
    print("Saved: %s" % args.out)


if __name__ == "__main__":
    main()
