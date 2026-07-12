#!/usr/bin/env python3
"""
Task 1 / Track 2 prerequisite: full-scale ID-join across HUMANISE's three raw
data sources (align_data_release, pure_motion, contact_motion), for ALL 19,648
clips -- not a spot check.

Key finding (this script is the evidence): HUMANISE.py in the afford-motion /
HUMANISE data-prep repos (found at
/home/user/jered/T2M_test/afford-motion/prepare/datasets/HUMANISE/HUMANISE.py)
builds its final per-clip index by:
    1. natsorted glob over align_data_release/<action>/<clip>/anno.pkl,
       iterated in natsorted ACTION order (lie, sit, stand up, walk -- pure
       alphabetical) then natsorted CLIP order within each action.
    2. Each anno.pkl contains a *list* of placement dicts (not always length
       1 -- verified lengths from 1 to 9 in this data release), flattened in
       list order.
    3. anno_index (0-based, over the flattened stream) becomes the motion_id,
       written zero-padded to 6 digits.

This script reproduces that exact ordering against HUMANISE/align_data_release
(zipped per action instead of on-disk directories, but zipfile.namelist() is
natsort-compatible here) and cross-checks scene_id and utterance/others fields
against HUMANISE/contact_motion/anno.csv row-for-row (the file that pairs with
contact_motion/motions/{idx:05d}.npy and HUMANISE/all.txt), across ALL rows,
plus verifies every 'motion' key referenced resolves to an existing
pure_motion/<action>/<motion_id>/motion.pkl entry.

NOTE: the top-level HUMANISE/annotations.csv is a stub (header only, 0 data
rows) -- NOT usable for anything. HUMANISE/contact_motion/anno.csv is the real
per-clip metadata table and is what this script validates against.
"""
import csv
import io
import pickle
import zipfile

from natsort import natsorted

ROOT = "/media/user/2tb/motion_data/HUMANISE"
ACTIONS = natsorted(["walk", "sit", "stand up", "lie"])


def build_flat_join():
    flat = []
    for a in ACTIONS:
        with zipfile.ZipFile(f"{ROOT}/align_data_release/{a}.zip") as zf:
            members = natsorted(n for n in zf.namelist() if n.endswith("anno.pkl"))
            for m in members:
                with zf.open(m) as f:
                    d = pickle.load(io.BytesIO(f.read()))
                for p in d:
                    flat.append(p)
    return flat


def load_anno_csv():
    rows = []
    with open(f"{ROOT}/contact_motion/anno.csv") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


if __name__ == "__main__":
    flat = build_flat_join()
    rows = load_anno_csv()

    print(f"align_data_release flattened entries: {len(flat)}")
    print(f"contact_motion/anno.csv rows:          {len(rows)}")

    n = min(len(flat), len(rows))
    mismatches = []
    for i in range(n):
        p, row = flat[i], rows[i]
        if p["scene"] != row["scene_id"] or p["utterance"] != row["others"]:
            mismatches.append(i)

    count_mismatch = len(mismatches) + abs(len(flat) - len(rows))
    print(f"Length match:            {len(flat) == len(rows)}")
    print(f"Field mismatches (scene_id + others), over {n} joined rows: {len(mismatches)}")
    if mismatches:
        print("First 20 mismatched indices:", mismatches[:20])

    # verify every 'motion' key resolves inside pure_motion/<action>.zip
    zips = {a: zipfile.ZipFile(f"{ROOT}/pure_motion/{a}.zip") for a in ["walk", "sit", "stand up", "lie"]}
    namelists = {a: set(z.namelist()) for a, z in zips.items()}
    missing_pure = []
    for i, p in enumerate(flat):
        member = f"{p['action']}/{p['motion']}/motion.pkl"
        if member not in namelists[p["action"]]:
            missing_pure.append((i, member))

    print(f"pure_motion misses:      {len(missing_pure)}")
    if missing_pure:
        print("First 20 misses:", missing_pure[:20])

    n_clips = 19648
    coverage_pct = 100.0 * (len(flat) - len(mismatches) - len(missing_pure)) / n_clips
    print("=" * 60)
    print(f"Expected total clips (contact_motion/motions count): {n_clips}")
    print(f"Joined entries matching all three sources cleanly:   "
          f"{len(flat) - len(mismatches) - len(missing_pure)} / {n_clips}  ({coverage_pct:.4f}%)")
    verdict = "PASS - 100% coverage, 0 misses" if (
        len(flat) == n_clips == len(rows) and not mismatches and not missing_pure
    ) else "FAIL - see mismatches/misses above"
    print(f"VERDICT: {verdict}")
