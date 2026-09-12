#!/usr/bin/env python3
"""Extract per-frame local heightmaps for every TRUMANS clip, matching the HUMANISE
extractor (scripts/scene_tokenizer/extract_heightmaps.py) in output format: per-clip
(T263, 32, 32) float16, height-above-clip-floor in [HM_LOW, HM_HIGH].

TRUMANS → HUMANISE adaptation notes:
  - TRUMANS is Y-UP (vertical = Y, horizontal = X,Z).  HUMANISE/ScanNet is Z-UP.
    The existing scene_heightmap.py (build_scene_kdtree, local_heightmap) expects Z-UP:
    KDTree on XY, height = Z.  We swap Y↔Z in the mesh vertices before passing them in,
    mapping (X, Y_vert, Z_horiz) → (X, Z_horiz, Y_vert).  Now the KDTree searches
    (X, Z_horiz) = the horizontal plane, and height = Y_vert — correct.
  - TRUMANS track2 xy is (X, Z) in Y-UP world — the SAME column order as the swapped
    KDTree's first two dims, so center_xy is passed directly.
  - Clip floor ≈ 0 for TRUMANS (feet on ground at Y ≈ 0; measured min-joint-Y mean=0.03
    std=0.06 over first 1000 frames).  We use the per-clip minimum pelvis height minus a
    conservative margin (pelvis is ~0.9 m above feet for standing, but feet are always the
    minimum).  More precisely: for clips where track2['height'].min() > 0.3 (never lying),
    clip_floor ≈ 0; for lying clips clip_floor could be higher.  With only 4 lying clips
    in TRUMANS, we default clip_floor = 0 for all (error < 3 cm in practice).

Output: {CACHE_HM}/{clip_id}.npy, shape (T263, 32, 32) float16.
Resumable (skips existing outputs).  Per-scene KDTree cached across clips.

Usage:
  source ~/anaconda3/etc/profile.d/conda.sh && conda activate afford
  python scripts/trumans/extract_heightmaps.py \
    --processed-root /media/user/2tb/motion_data/TRUMANS_processed \
    --scene-mesh-dir /media/user/2tb/motion_data/TRUMANS/Scene_mesh \
    --out ~/wander_data/trumans_heightmap_cache
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(REPO, "src"))
from scene_heightmap import build_scene_kdtree, local_heightmap, to_clip_frame, GRID_N  # noqa: E402


def load_trumans_mesh_yup_to_zup(mesh_path):
    """Load a TRUMANS mesh (Y-up) and swap Y↔Z so the result is Z-up compatible
    with scene_heightmap.build_scene_kdtree.  Returns the swapped (N, 3) vertices."""
    import trimesh
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    v = mesh.vertices.copy()  # (N, 3) in Y-up: X, Y_vert, Z_horiz
    # Swap columns 1 and 2: (X, Y, Z) -> (X, Z, Y)
    # Now: col 0 = X, col 1 = Z_horiz, col 2 = Y_vert
    # build_scene_kdtree uses XY for horizontal, Z for height → correct.
    v[:, 1], v[:, 2] = mesh.vertices[:, 2].copy(), mesh.vertices[:, 1].copy()
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-root", required=True,
                    help="Output of convert_trumans.py (has trumans_meta.pkl, trumans_263_cache, trumans_track2)")
    ap.add_argument("--scene-mesh-dir", required=True,
                    help="Path to TRUMANS Scene_mesh/ with .obj files")
    ap.add_argument("--out", required=True,
                    help="Output directory for per-clip heightmap .npy files")
    ap.add_argument("--report-every", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    meta = pickle.load(open(os.path.join(args.processed_root, "trumans_meta.pkl"), "rb"))
    cache_263 = os.path.join(args.processed_root, "trumans_263_cache")
    cache_track2 = os.path.join(args.processed_root, "trumans_track2")

    print(f"Extracting heightmaps for {len(meta)} TRUMANS clips")
    print(f"  263 cache: {cache_263}")
    print(f"  track2:    {cache_track2}")
    print(f"  meshes:    {args.scene_mesh_dir}")
    print(f"  output:    {args.out}")

    scene_cache = {}  # scene_name -> (kdt, verts_z, floor_z) or None

    def get_scene(scene_name):
        if scene_name not in scene_cache:
            if len(scene_cache) > 30:
                scene_cache.clear()
            mesh_path = os.path.join(args.scene_mesh_dir, f"{scene_name}.obj")
            if not os.path.exists(mesh_path):
                print(f"  scene {scene_name}: mesh not found at {mesh_path}", flush=True)
                scene_cache[scene_name] = None
                return None
            try:
                verts_swapped = load_trumans_mesh_yup_to_zup(mesh_path)
                scene_cache[scene_name] = build_scene_kdtree(verts_swapped)
            except Exception as e:  # noqa: BLE001
                print(f"  scene {scene_name} failed to load: {e}", flush=True)
                scene_cache[scene_name] = None
        return scene_cache[scene_name]

    t0 = time.time()
    n_done = n_skip_exist = n_skip_no263 = n_fail = n_no_scene = 0
    fills = []

    for m in meta:
        clip_id = m["clip_id"]
        out_path = os.path.join(args.out, f"{clip_id}.npy")
        if os.path.exists(out_path):
            n_skip_exist += 1
            continue

        c263_path = os.path.join(cache_263, f"{clip_id}.npy")
        if not os.path.exists(c263_path):
            n_skip_no263 += 1
            continue

        try:
            t263 = int(np.load(c263_path, mmap_mode="r").shape[0])

            sc = get_scene(m["scene"])
            if sc is None:
                n_no_scene += 1
                continue
            kdt, verts_z, scene_floor_z = sc

            # Load track2 for world position + yaw
            t2 = np.load(os.path.join(cache_track2, f"{clip_id}.npy"),
                         allow_pickle=True).item()
            xy = t2["xy"]      # (T, 2) in Y-up (X, Z) = horizontal plane
            yaw = t2["yaw"]    # (T,)

            if xy.shape[0] < t263:
                n_fail += 1
                print(f"  #{clip_id}: track {xy.shape[0]} < t263 {t263}, skipping", flush=True)
                continue

            # Clip floor: for TRUMANS, feet are on the ground at Y ≈ 0.
            # After the Y↔Z swap, scene_floor_z from build_scene_kdtree = min of the
            # swapped Z column = min of original Y = the scene's floor height.
            # The clip floor in the 263 frame = min joint world Y over the clip.
            # For TRUMANS this is ≈ 0 (within 3 cm), which is close to scene_floor_z
            # (also ≈ 0).  Use 0.0 for all clips — the 263 features were extracted
            # with process_file which subtracts min-joint-Y (≈ 0) from all joints.
            clip_floor = 0.0

            hm = np.empty((t263, GRID_N, GRID_N), dtype=np.float16)
            f_acc = 0.0
            for t in range(t263):
                h_abs, f = local_heightmap(kdt, verts_z, xy[t], float(yaw[t]))
                hm[t] = to_clip_frame(h_abs, clip_floor).astype(np.float16)
                f_acc += f

            np.save(out_path, hm)
            fills.append(f_acc / t263)
            n_done += 1
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print(f"  #{clip_id} failed: {e}", flush=True)
            continue

        total_processed = n_done + n_skip_exist
        if total_processed % args.report_every == 0:
            el = time.time() - t0
            rate = n_done / el if el > 0 else 0
            print(f"[{el:7.1f}s] done={n_done} skip_exist={n_skip_exist} "
                  f"no263={n_skip_no263} no_scene={n_no_scene} fail={n_fail}  "
                  f"{rate:.1f} clips/s  fill~{np.mean(fills[-200:]) if fills else float('nan'):.2f}",
                  flush=True)

    el = time.time() - t0
    print("=" * 60)
    print(f"DONE in {el:.1f}s  new={n_done} skip_exist={n_skip_exist} "
          f"no263={n_skip_no263} no_scene={n_no_scene} fail={n_fail}  "
          f"mean_fill={np.mean(fills) if fills else float('nan'):.3f}")


if __name__ == "__main__":
    main()
