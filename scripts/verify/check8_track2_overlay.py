#!/usr/bin/env python3
"""
Task 3 -- Validate Track 2 (world-frame root trajectory) by overlaying
reconstructed trajectories on their scene's ScanNet mesh. This IS the
Y-up/Z-up axis-convention correctness test called for in STEP1b_extend.md:
if the axis handling in src/humanise_join.py were wrong, trajectories would
float above or sink below the floor, or land outside the room's footprint.

For each of a handful of scenes, picks a few clips placed in that scene,
computes Track 2 via src/humanise_join.compute_track2, and renders:
  - a top-down (X, Y) view: scene mesh footprint (all vertices projected to
    XY) as a gray scatter, with each clip's root (x, y) trajectory as a
    colored line + start/end markers.
  - a side view: height (Z) of the root trajectory over time, with the
    scene's floor (min Z) and ceiling (max Z) marked as reference lines --
    the direct "sitting on the floor, not floating/sinking" check.

Pass/fail rule (recorded per scene): the root trajectory's Z stays within
[floor - 0.15m, floor + 2.2m] for the whole clip (generous human-height
envelope) AND the XY trajectory stays within the mesh's XY bounding box
(with a small margin for near-wall placements).
"""
import os
import sys
import warnings

import numpy as np
import trimesh

sys.path.insert(0, "/home/user/Khiem-ssh/wander/src")
warnings.filterwarnings("ignore")

import humanise_join as hj  # noqa: E402

SCANNET_ROOT = "/media/user/2tb/motion_data/scannet/scans"
OUT_DIR = "/home/user/Khiem-ssh/wander/scratch_outputs/overlay"
os.makedirs(OUT_DIR, exist_ok=True)


def load_scene_mesh(scene_id):
    path = f"{SCANNET_ROOT}/{scene_id}/{scene_id}_vh_clean_2.ply"
    return trimesh.load(path, process=False)


def pick_clips_for_scenes(scenes, per_scene=3):
    flat = hj.build_flat_join()
    chosen = {s: [] for s in scenes}
    for i, p in enumerate(flat):
        s = p["scene"]
        if s in chosen and len(chosen[s]) < per_scene:
            chosen[s].append(i)
        if all(len(v) >= per_scene for v in chosen.values()):
            break
    return chosen


def render_and_check(scene_id, clip_indices):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mesh = load_scene_mesh(scene_id)
    verts = mesh.vertices
    floor_z = mesh.bounds[0, 2]
    ceil_z = mesh.bounds[1, 2]
    xmin, ymin = mesh.bounds[0, :2]
    xmax, ymax = mesh.bounds[1, :2]
    margin = 1.0  # meters, generous margin for near-wall / registration slack

    fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(12, 5))

    # subsample mesh vertices for a fast top-down footprint scatter
    rng = np.random.RandomState(0)
    sub = verts[rng.choice(len(verts), size=min(20000, len(verts)), replace=False)]
    ax_top.scatter(sub[:, 0], sub[:, 1], s=0.2, c="lightgray", alpha=0.5, label="scene mesh (top-down)")

    results = []
    colors = plt.cm.tab10.colors
    for ci, idx in enumerate(clip_indices):
        rec = hj.get_record(idx)
        joints_world, xy, yaw, sincos = hj.compute_track2(rec)
        z = joints_world[:, hj.J_PELVIS, 2]

        color = colors[ci % len(colors)]
        ax_top.plot(xy[:, 0], xy[:, 1], "-", color=color, linewidth=1.5,
                    label=f"clip {idx} ({rec.action})")
        ax_top.scatter(xy[0, 0], xy[0, 1], color=color, marker="o", s=40)  # start
        ax_top.scatter(xy[-1, 0], xy[-1, 1], color=color, marker="x", s=40)  # end

        t = np.arange(len(z))
        ax_side.plot(t, z, "-", color=color, label=f"clip {idx} pelvis height")

        xy_ok = bool(np.all(xy[:, 0] >= xmin - margin) and np.all(xy[:, 0] <= xmax + margin) and
                     np.all(xy[:, 1] >= ymin - margin) and np.all(xy[:, 1] <= ymax + margin))
        z_ok = bool(np.all(z >= floor_z - 0.15) and np.all(z <= floor_z + 2.2))
        results.append({
            "clip": idx, "action": rec.action, "xy_ok": xy_ok, "z_ok": z_ok,
            "z_range": (float(z.min()), float(z.max())), "floor_z": float(floor_z),
        })

    ax_side.axhline(floor_z, color="k", linestyle="--", label="floor (mesh min z)")
    ax_side.axhline(ceil_z, color="gray", linestyle=":", label="ceiling (mesh max z)")
    ax_top.set_xlabel("X (m)")
    ax_top.set_ylabel("Y (m)")
    ax_top.set_title(f"{scene_id}: top-down root trajectory vs mesh footprint")
    ax_top.axis("equal")
    ax_top.legend(fontsize=6, loc="upper right")
    ax_side.set_xlabel("frame")
    ax_side.set_ylabel("pelvis Z (m, world)")
    ax_side.set_title(f"{scene_id}: pelvis height vs floor/ceiling")
    ax_side.legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    out_path = f"{OUT_DIR}/{scene_id}_overlay.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path, results


if __name__ == "__main__":
    scenes = ["scene0000_00", "scene0005_00", "scene0006_00", "scene0050_00", "scene0100_00"]
    chosen = pick_clips_for_scenes(scenes, per_scene=3)

    all_pass = True
    for scene_id in scenes:
        idxs = chosen[scene_id]
        if not idxs:
            print(f"{scene_id}: NO CLIPS FOUND, skipping")
            continue
        out_path, results = render_and_check(scene_id, idxs)
        print(f"\n{scene_id} -> {out_path}")
        for r in results:
            status = "PASS" if (r["xy_ok"] and r["z_ok"]) else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  clip {r['clip']:6d} action={r['action']:10s} "
                  f"z_range=({r['z_range'][0]:.2f},{r['z_range'][1]:.2f}) floor_z={r['floor_z']:.2f} "
                  f"xy_ok={r['xy_ok']} z_ok={r['z_ok']}  -> {status}")

    print("\n" + "=" * 60)
    print(f"OVERALL: {'ALL PASS' if all_pass else 'SOME FAILURES -- see above'}")
