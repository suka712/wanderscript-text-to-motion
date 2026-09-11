#!/usr/bin/env python
"""
Convert TRUMANS dataset → our training format (263-dim features + world-frame tracks).

TRUMANS provides concatenated numpy arrays (3.79M frames) with SMPL-X params + pre-computed
joints. We split into segments by action transitions, convert to 22-joint Y-up positions,
extract 263-dim HumanML3D features, compute world-frame root tracks, and save per-clip
.npy files matching the HUMANISE 263_cache format.

Key findings about TRUMANS action labels:
  - Label 0 is NOT "lie down" — it's idle/transit (pelvis ~0.93 m, standing/walking)
  - Labels 1-9 are object interactions (squat, mouse, keyboard, etc.)
  - Body posture (sit/stand/walk) must be inferred from pelvis height, not action label
  - TRUMANS has varied seat heights (0.52–0.93 m) — the diversity HUMANISE lacks

Input:  TRUMANS_ROOT/  (from Google Drive download)
Output: OUT_ROOT/
          trumans_263_cache/  — per-clip .npy files (T, 263)
          trumans_track2/     — per-clip world-frame root tracks {xy, yaw, sincos, height}
          trumans_meta.pkl    — metadata (action, scene, text, etc.)

Usage:
  python scripts/trumans/convert_trumans.py --trumans-root /media/user/2tb/motion_data/TRUMANS \
    --out-root /media/user/2tb/motion_data/TRUMANS_processed [--max-clips N]
"""
from __future__ import annotations
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# --- Project imports ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

T2M_GPT_ROOT = os.environ.get("T2M_GPT_ROOT", str(PROJECT_ROOT.parent / "T2M-GPT"))
sys.path.insert(0, T2M_GPT_ROOT)

from src.motion_features import extract_263

# Pelvis height thresholds for body-posture classification
SITTING_THRESHOLD = 0.65   # below this = seated (covers squat 0.52, low chairs)
LYING_THRESHOLD = 0.35     # below this = lying
STANDING_THRESHOLD = 0.85  # above this = standing/walking

# Our 4-way action labels
OUR_ACTIONS = {"walk": 0, "sit": 1, "stand up": 2, "lie": 3}


def load_trumans(root):
    """Load all TRUMANS arrays from the download directory."""
    root = Path(root)
    data = {}
    for name in [
        "human_joints", "human_transl", "human_orient", "human_pose",
        "idx_start", "seg_name", "action_label", "scene_flag",
        "scene_list", "frame_id", "betas",
    ]:
        path = root / ("%s.npy" % name)
        if path.exists():
            # Use mmap for large arrays to avoid loading everything into RAM
            if name in ("human_joints", "human_pose", "human_transl", "human_orient"):
                data[name] = np.load(str(path), mmap_mode="r")
            else:
                data[name] = np.load(str(path), allow_pickle=True)
            print("  Loaded %s: shape=%s, dtype=%s" % (name, data[name].shape, data[name].dtype))
        else:
            print("  MISSING: %s" % name)
    return data


def split_by_action_transitions(data):
    """Split concatenated arrays into segments at action-label transitions.

    Within each recording session (seg_name), find where the action label changes.
    Also split at session boundaries. This gives us individual action segments
    (e.g., "walking to the desk", "sitting at keyboard", "reaching for bottle").
    """
    seg_names = data["seg_name"]
    al = data["action_label"]
    acts = al.argmax(axis=1) if al.ndim == 2 else al
    n_frames = len(seg_names)

    segments = []
    current_name = seg_names[0]
    current_act = acts[0]
    start = 0

    for i in range(1, n_frames):
        # Split at session boundary OR action transition
        if seg_names[i] != current_name or acts[i] != current_act:
            segments.append({
                "start": start,
                "end": i,
                "session": str(current_name),
                "action_label": int(current_act),
            })
            current_name = seg_names[i]
            current_act = acts[i]
            start = i

    # Last segment
    segments.append({
        "start": start,
        "end": n_frames,
        "session": str(current_name),
        "action_label": int(current_act),
    })

    lengths = [s["end"] - s["start"] for s in segments]
    print("  Found %d action segments" % len(segments))
    print("    Length: min=%d, max=%d, median=%d, mean=%d" % (
        min(lengths), max(lengths), int(np.median(lengths)), int(np.mean(lengths))))

    return segments


def classify_posture(joints_22, action_label):
    """Classify body posture from pelvis height, not TRUMANS action labels.

    TRUMANS action_label 0 = idle/transit (NOT lying), 1 = squat, 2-9 = object interactions.
    We classify by pelvis height:
      - lying:    pelvis Y < 0.35 m
      - sitting:  pelvis Y < 0.65 m (includes squat, low chairs, couches)
      - standing: pelvis Y > 0.85 m (walking, standing at desk)
      - between:  ambiguous, classify by velocity (moving = walk, still = sit)
    """
    pelvis_y = joints_22[:, 0, 1]  # Y-up, pelvis = joint 0
    median_h = float(np.median(pelvis_y))

    if median_h < LYING_THRESHOLD:
        return "lie"
    elif median_h < SITTING_THRESHOLD:
        return "sit"
    elif median_h > STANDING_THRESHOLD:
        # Check if moving (walk) or stationary (stand)
        pelvis_xz = joints_22[:, 0, [0, 2]]  # horizontal position
        displacement = np.linalg.norm(pelvis_xz[-1] - pelvis_xz[0])
        if displacement > 0.5:  # moved > 0.5 m
            return "walk"
        else:
            return "stand up"  # standing in place
    else:
        # Ambiguous zone (0.35-0.65 or 0.65-0.85): check velocity
        pelvis_xz = joints_22[:, 0, [0, 2]]
        displacement = np.linalg.norm(pelvis_xz[-1] - pelvis_xz[0])
        if displacement > 0.5:
            return "walk"
        else:
            return "sit"  # perching / high sit


def compute_world_track_yup(joints_22):
    """Compute world-frame root track from Y-up (T, 22, 3) joints.

    In Y-up, horizontal = X,Z plane. We store (X, Z) as "xy" for downstream
    compatibility with the Z-up HUMANISE track2 format.
    """
    J_PELVIS = 0
    J_LHIP, J_RHIP = 1, 2
    J_LSHOULDER, J_RSHOULDER = 16, 17

    # Horizontal position
    xz = joints_22[:, J_PELVIS, [0, 2]].copy()
    height = joints_22[:, J_PELVIS, 1].copy()

    # Yaw from hip+shoulder cross product
    across = (joints_22[:, J_RHIP, [0, 2]] - joints_22[:, J_LHIP, [0, 2]]) + \
             (joints_22[:, J_RSHOULDER, [0, 2]] - joints_22[:, J_LSHOULDER, [0, 2]])
    forward_x = -across[:, 1]
    forward_z = across[:, 0]
    yaw = np.arctan2(forward_x, forward_z)
    sincos = np.stack([np.sin(yaw), np.cos(yaw)], axis=-1)

    return {
        "xy": xz.astype(np.float32),
        "yaw": yaw.astype(np.float32),
        "sincos": sincos.astype(np.float32),
        "height": height.astype(np.float32),
    }


def convert_one_segment(joints_22, min_frames=16, max_frames=300, feet_thre=0.002):
    """Convert a single segment's joints to 263-dim features + world track.

    Returns (features_263, track2_dict) or None if too short/long or extraction fails.
    """
    T = joints_22.shape[0]
    if T < min_frames or T > max_frames:
        return None

    try:
        result = extract_263(joints_22, feet_thre=feet_thre)
        features_263 = result[0]  # (T-1, 263) — process_file drops 1 frame
    except Exception as e:
        return None

    if features_263 is None or len(features_263) < min_frames - 1:
        return None

    track2 = compute_world_track_yup(joints_22)
    # Trim track2 to match 263 length (process_file drops frame 0)
    for k in track2:
        track2[k] = track2[k][1:]

    return features_263, track2


def parse_action_texts(actions_dir, session_name):
    """Parse action text annotations from the Actions/ directory.

    Each file: lines of "start_frame end_frame description"
    Returns list of {start, end, text} sorted by start frame.
    """
    # Session name may have _augment suffix; strip it for the text file
    base_session = session_name.split("_augment")[0]
    txt_path = Path(actions_dir) / ("%s.txt" % base_session)

    if not txt_path.exists():
        return []

    annotations = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: "start_frame end_frame description" (space-separated)
            parts = line.split(None, 2)  # split on whitespace, max 3 parts
            if len(parts) >= 3:
                try:
                    start = int(parts[0])
                    end = int(parts[1])
                    text = parts[2]
                    annotations.append({"start": start, "end": end, "text": text})
                except ValueError:
                    pass

    return sorted(annotations, key=lambda x: x["start"])


def main():
    parser = argparse.ArgumentParser(description="Convert TRUMANS to our format")
    parser.add_argument("--trumans-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--check-joints", action="store_true")
    parser.add_argument("--min-frames", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--max-clips", type=int, default=None)
    args = parser.parse_args()

    print("Loading TRUMANS arrays...")
    data = load_trumans(args.trumans_root)

    if args.check_joints:
        joints = data["human_joints"]
        avg = joints[:1000, :22, :].mean(axis=0)
        print("\n=== Joint order check ===")
        print("  Joint  0 pelvis:     Y=%.3f" % avg[0, 1])
        print("  Joint  1 L_hip:      Y=%.3f" % avg[1, 1])
        print("  Joint  2 R_hip:      Y=%.3f" % avg[2, 1])
        print("  Joint 15 head:       Y=%.3f" % avg[15, 1])
        print("  Joint 10 L_foot:     Y=%.3f" % avg[10, 1])
        print("  Joint 11 R_foot:     Y=%.3f" % avg[11, 1])
        head_y, pelvis_y, feet_y = avg[15, 1], avg[0, 1], min(avg[10, 1], avg[11, 1])
        if head_y > pelvis_y > feet_y:
            print("  OK: head > pelvis > feet")
        else:
            print("  WRONG vertical ordering!")
        return

    for required in ["human_joints", "seg_name", "action_label", "scene_flag", "scene_list"]:
        if required not in data:
            print("ERROR: missing %s.npy" % required)
            sys.exit(1)

    joints_all = data["human_joints"]  # (N, 24, 3) mmap
    print("  Total frames: %d, joints: %d" % (joints_all.shape[0], joints_all.shape[1]))

    # Split by action transitions
    print("\nSplitting by action transitions...")
    segments = split_by_action_transitions(data)

    # Prepare output
    out_root = Path(args.out_root)
    cache_dir = out_root / "trumans_263_cache"
    track_dir = out_root / "trumans_track2"
    cache_dir.mkdir(parents=True, exist_ok=True)
    track_dir.mkdir(parents=True, exist_ok=True)

    actions_dir = Path(args.trumans_root) / "Actions"
    scene_list = data["scene_list"]
    scene_flags = data["scene_flag"]

    # Precompute session start indices (global frame index of each session's first frame)
    # so we can convert global segment indices to session-local for annotation lookup
    session_starts = {}
    seg_names = data["seg_name"]
    for seg in segments:
        sn = seg["session"]
        if sn not in session_starts:
            session_starts[sn] = seg["start"]

    meta = []
    n_ok = 0
    n_skip_length = 0
    n_skip_extract = 0
    posture_counts = {}

    limit = args.max_clips or len(segments)

    for seg in tqdm(segments[:limit], desc="Converting"):
        s, e = seg["start"], seg["end"]
        seg_len = e - s

        # Quick length check before loading joints
        if seg_len < args.min_frames or seg_len > args.max_frames:
            n_skip_length += 1
            continue

        # Load 22-joint subset (copy from mmap)
        seg_joints = np.array(joints_all[s:e, :22, :])  # (T, 22, 3), Y-up

        # Classify body posture from pelvis height
        posture = classify_posture(seg_joints, seg["action_label"])
        posture_counts[posture] = posture_counts.get(posture, 0) + 1

        # Get scene
        scene_idx = int(scene_flags[s])
        scene_name = str(scene_list[scene_idx]) if scene_idx < len(scene_list) else "scene_%d" % scene_idx

        # Get text annotation (Actions/ files use session-local frame indices)
        annotations = parse_action_texts(actions_dir, seg["session"])
        local_s = s - session_starts[seg["session"]]  # convert global → session-local
        local_e = e - session_starts[seg["session"]]
        text = posture  # fallback: just the posture name
        for ann in annotations:
            # Overlap: annotation range intersects our segment
            if ann["start"] < local_e and ann["end"] > local_s:
                text = ann["text"]
                break

        # Convert to 263-dim
        result = convert_one_segment(
            seg_joints,
            min_frames=args.min_frames,
            max_frames=args.max_frames,
        )

        if result is None:
            n_skip_extract += 1
            continue

        features_263, track2 = result

        # Save
        clip_id = "%05d" % n_ok
        np.save(str(cache_dir / ("%s.npy" % clip_id)), features_263)
        np.save(str(track_dir / ("%s.npy" % clip_id)), track2, allow_pickle=True)

        meta.append({
            "clip_id": clip_id,
            "session": seg["session"],
            "action": posture,  # our 4-way label
            "action_label_trumans": seg["action_label"],
            "scene": scene_name,
            "text": text,
            "n_frames_raw": seg_len,
            "n_frames_263": len(features_263),
            "pelvis_height_median": float(np.median(seg_joints[:, 0, 1])),
            "displacement": float(np.linalg.norm(seg_joints[-1, 0, [0, 2]] - seg_joints[0, 0, [0, 2]])),
        })
        n_ok += 1

    # Save metadata
    meta_path = out_root / "trumans_meta.pkl"
    with open(str(meta_path), "wb") as f:
        pickle.dump(meta, f)

    print("\nDone: %d clips converted" % n_ok)
    print("  Skipped (length): %d" % n_skip_length)
    print("  Skipped (extract): %d" % n_skip_extract)
    print("  263 cache: %s" % cache_dir)
    print("  Track2:    %s" % track_dir)
    print("  Metadata:  %s" % meta_path)

    if meta:
        print("\n  Posture distribution: %s" % posture_counts)
        scenes = set(m["scene"] for m in meta)
        print("  Unique scenes: %d" % len(scenes))
        lengths = [m["n_frames_263"] for m in meta]
        print("  Frame lengths: min=%d, max=%d, mean=%d, median=%d" % (
            min(lengths), max(lengths), int(np.mean(lengths)), int(np.median(lengths))))
        heights = [m["pelvis_height_median"] for m in meta]
        print("  Pelvis heights: min=%.3f, max=%.3f, mean=%.3f" % (
            min(heights), max(heights), np.mean(heights)))

        # Per-posture height stats
        for p in sorted(posture_counts.keys()):
            ph = [m["pelvis_height_median"] for m in meta if m["action"] == p]
            if ph:
                print("    %s: n=%d, height=%.3f +/- %.3f" % (p, len(ph), np.mean(ph), np.std(ph)))


if __name__ == "__main__":
    main()
