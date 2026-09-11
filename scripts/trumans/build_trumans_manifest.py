#!/usr/bin/env python
"""
Build a training manifest from converted TRUMANS clips.

Two phases:
  1. GEOMETRIC (CPU-only): start, goal, xy_traj, prefix_pose, a_end_pose, action, text,
     occ_crop from rasterized scene meshes — everything tokenizer-INDEPENDENT.
  2. TOKENIZE (GPU): encode each 263-dim clip through the finetuned VQ-VAE → tokens.
     Run with --tokenize after phase 1.

Output: OUT_DIR/train.pkl  (list of dicts matching step10/tokens format)

Usage:
  # Phase 1 (CPU-only, computes everything except tokens):
  python build_trumans_manifest.py \
    --processed-root /media/user/2tb/motion_data/TRUMANS_processed \
    --scene-mesh-dir /media/user/2tb/motion_data/TRUMANS/Scene_mesh \
    --out-dir ~/wander_data/trumans_tokens

  # Phase 2 (needs GPU):
  python build_trumans_manifest.py \
    --processed-root /media/user/2tb/motion_data/TRUMANS_processed \
    --out-dir ~/wander_data/trumans_tokens \
    --tokenize --vqvae-ckpt ~/wander_data/motion_data/track2_checkpoints/net_iter020000.pth
"""
from __future__ import annotations
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "scene_probe"))

T2M_GPT_ROOT = os.environ.get("T2M_GPT_ROOT", str(REPO_ROOT.parent / "T2M-GPT"))
sys.path.insert(0, T2M_GPT_ROOT)

import motion_features as mf  # noqa: E402

MAX_T = 196
OCC_N = 28  # matches step10 occ_crop resolution


def crop_to_multiple(T, factor=4, max_t=MAX_T):
    return (min(T, max_t) // factor) * factor


def rasterize_scene_occ(mesh_path, resolution=0.05, height_thresh=0.9):
    """Rasterize a scene mesh into a binary occupancy grid.

    Returns (occ, extent) where occ is a 2D binary array and extent is
    [x_min, x_max, y_min, y_max] in the mesh's coordinate frame (Y-up for TRUMANS,
    so horizontal = X,Z; we treat Z as "y" in the 2D grid to match downstream code).

    height_thresh: only voxels above this height count (walls, not floor).
    Same 0.9m threshold as the rest of the project (CLAUDE.md 2a).
    """
    import trimesh
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    verts = mesh.vertices  # (N, 3) — TRUMANS is Y-up: X,Y,Z where Y is vertical

    # Horizontal extents (X, Z)
    x_min, x_max = verts[:, 0].min() - 0.5, verts[:, 0].max() + 0.5
    z_min, z_max = verts[:, 2].min() - 0.5, verts[:, 2].max() + 0.5

    # Sample points on mesh surface
    n_samples = max(100000, len(mesh.faces) * 10)
    try:
        pts, _ = trimesh.sample.sample_surface(mesh, n_samples)
    except Exception:
        # Fallback: use vertices directly
        pts = verts

    # Filter tall obstacles (Y > height_thresh)
    tall = pts[pts[:, 1] > height_thresh]

    # Rasterize X,Z into grid
    nx = int(np.ceil((x_max - x_min) / resolution))
    nz = int(np.ceil((z_max - z_min) / resolution))
    occ = np.zeros((nz, nx), dtype=np.float32)

    if len(tall) > 0:
        xi = np.clip(((tall[:, 0] - x_min) / resolution).astype(int), 0, nx - 1)
        zi = np.clip(((tall[:, 2] - z_min) / resolution).astype(int), 0, nz - 1)
        occ[zi, xi] = 1.0

    extent = np.array([x_min, x_max, z_min, z_max], dtype=np.float32)
    return occ, extent


def crop_agent_frame_yup(occ, extent, agent_xz, agent_yaw, crop_m=3.0, crop_px=112):
    """Crop occupancy around the agent in its heading-aligned frame (Y-up coords).

    agent_xz: (2,) agent position in (X, Z) world coords
    agent_yaw: scalar heading angle
    Returns (crop_px, crop_px) binary array centered on agent, rotated to heading.
    """
    from scipy.ndimage import map_coordinates

    x_min, x_max, z_min, z_max = extent
    res = (x_max - x_min) / occ.shape[1]  # resolution

    # Build the output grid in agent-frame coordinates
    lin = np.linspace(-crop_m, crop_m, crop_px)
    gx, gz = np.meshgrid(lin, lin)  # agent-local

    # Rotate by heading to world
    cos_y, sin_y = np.cos(agent_yaw), np.sin(agent_yaw)
    wx = cos_y * gx - sin_y * gz + agent_xz[0]
    wz = sin_y * gx + cos_y * gz + agent_xz[1]

    # Map to pixel coords
    px = (wx - x_min) / res
    pz = (wz - z_min) / res

    # Bilinear sample
    crop = map_coordinates(occ, [pz.ravel(), px.ravel()], order=1, mode="constant", cval=0.0)
    return crop.reshape(crop_px, crop_px).astype(np.float32)


def build_geometric(args):
    """Phase 1: build all tokenizer-independent fields."""
    proc = Path(args.processed_root)
    meta = pickle.load(open(proc / "trumans_meta.pkl", "rb"))
    cache_dir = proc / "trumans_263_cache"
    track_dir = proc / "trumans_track2"

    scene_mesh_dir = Path(args.scene_mesh_dir) if args.scene_mesh_dir else None

    # Cache for rasterized scene occupancy grids
    occ_cache = {}
    occ_cache_dir = proc / "scene_occ_cache"
    occ_cache_dir.mkdir(exist_ok=True)

    manifest = []
    n_skip = 0
    n_no_scene = 0

    for m in tqdm(meta, desc="Building geometric fields"):
        clip_id = m["clip_id"]

        # Load 263-dim features
        data263 = np.load(str(cache_dir / ("%s.npy" % clip_id))).astype(np.float32)
        T263 = data263.shape[0]

        # Crop to token-aligned length
        T = crop_to_multiple(T263)
        if T < 4:
            n_skip += 1
            continue
        data263 = data263[:T]

        # Load track2
        track2 = np.load(str(track_dir / ("%s.npy" % clip_id)), allow_pickle=True).item()
        xy = track2["xy"][:T]
        sincos = track2["sincos"][:T]
        yaw = track2["yaw"][:T]

        # Start and goal
        start = np.array([xy[0, 0], xy[0, 1], sincos[0, 0], sincos[0, 1]], dtype=np.float32)
        goal = xy[T - 1].astype(np.float32)

        # xy trajectory for goal augmentation (walk segments only)
        xy_traj = xy.copy()

        # prefix_pose and a_end_pose: root-relative heading-canonicalized joints
        local_joints = mf.local_joint_positions(data263)
        prefix_pose = local_joints[0].ravel().astype(np.float32)    # (66,)
        a_end_pose = local_joints[-1].ravel().astype(np.float32)    # (66,)

        # Occupancy crop
        occ_crop = None
        if scene_mesh_dir is not None:
            scene_name = m["scene"]
            mesh_path = scene_mesh_dir / ("%s.obj" % scene_name)
            if mesh_path.exists():
                # Check cache
                cache_path = occ_cache_dir / ("%s.npz" % scene_name)
                if scene_name not in occ_cache:
                    if cache_path.exists():
                        z = np.load(str(cache_path))
                        occ_cache[scene_name] = (z["occ"], z["extent"])
                    else:
                        try:
                            occ, extent = rasterize_scene_occ(str(mesh_path))
                            np.savez_compressed(str(cache_path), occ=occ, extent=extent)
                            occ_cache[scene_name] = (occ, extent)
                        except Exception as e:
                            print("  Failed to rasterize %s: %s" % (scene_name, e))
                            n_no_scene += 1
                            continue
                    # Limit cache size
                    if len(occ_cache) > 50:
                        # Keep recent entries
                        keys = list(occ_cache.keys())
                        for k in keys[:20]:
                            del occ_cache[k]

                occ, ext = occ_cache[scene_name]
                yaw0 = float(yaw[0])
                crop = crop_agent_frame_yup(occ, ext, xy[0], yaw0)
                # Downsample to 28x28 like step10
                k = crop.shape[0] // OCC_N
                occ_crop = crop.reshape(OCC_N, k, OCC_N, k).mean((1, 3)).ravel().astype(np.float32)
            else:
                n_no_scene += 1
                # Still include the clip with zero occ_crop
                occ_crop = np.zeros(OCC_N * OCC_N, dtype=np.float32)

        entry = {
            "index": int(clip_id),
            "tokens": None,  # filled in phase 2
            "start": start,
            "goal": goal,
            "xy_traj": xy_traj,
            "prefix_pose": prefix_pose,
            "a_end_pose": a_end_pose,
            "text": m["text"],
            "action": m["action"],
            # TRUMANS-specific metadata (harmless to downstream, useful for debugging)
            "session": m["session"],
            "scene": m["scene"],
            "pelvis_height_median": m["pelvis_height_median"],
            "n_frames_263": T,
        }
        if occ_crop is not None:
            entry["occ_crop"] = occ_crop

        manifest.append(entry)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train_geometric.pkl"
    with open(str(out_path), "wb") as f:
        pickle.dump(manifest, f)

    print("\nPhase 1 done: %d entries (skipped %d short, %d no scene)" %
          (len(manifest), n_skip, n_no_scene))
    print("Saved: %s" % out_path)

    # Stats
    if manifest:
        actions = {}
        for e in manifest:
            actions[e["action"]] = actions.get(e["action"], 0) + 1
        print("Actions:", actions)
        heights = [e["pelvis_height_median"] for e in manifest]
        print("Height range: %.3f - %.3f" % (min(heights), max(heights)))


def tokenize(args):
    """Phase 2: encode 263-dim clips through VQ-VAE to get tokens."""
    import torch
    from vqvae_loader import load_vqvae

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = load_vqvae(args.vqvae_ckpt, device=device)

    mean = np.load(os.path.join(
        T2M_GPT_ROOT, "checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy"
    )).astype(np.float32)
    std = np.load(os.path.join(
        T2M_GPT_ROOT, "checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy"
    )).astype(np.float32)

    out_dir = Path(args.out_dir)
    geom_path = out_dir / "train_geometric.pkl"
    manifest = pickle.load(open(str(geom_path), "rb"))

    proc = Path(args.processed_root)
    cache_dir = proc / "trumans_263_cache"

    print("Tokenizing %d clips on %s..." % (len(manifest), device))
    n_ok = 0
    for entry in tqdm(manifest, desc="Tokenizing"):
        clip_id = "%05d" % entry["index"]
        data263 = np.load(str(cache_dir / ("%s.npy" % clip_id))).astype(np.float32)
        T = entry["n_frames_263"]
        data263 = data263[:T]

        norm = (data263 - mean) / std
        with torch.no_grad():
            x = torch.from_numpy(norm).unsqueeze(0).to(device)
            tokens = net.encode(x)[0].cpu().numpy().astype(np.int64)

        entry["tokens"] = tokens
        n_ok += 1

    # Remove entries with None tokens (shouldn't happen but be safe)
    manifest = [e for e in manifest if e["tokens"] is not None]

    out_path = out_dir / "train.pkl"
    with open(str(out_path), "wb") as f:
        pickle.dump(manifest, f)

    print("Phase 2 done: %d clips tokenized" % n_ok)
    print("Saved: %s" % out_path)

    # Token stats
    tok_lens = [len(e["tokens"]) for e in manifest]
    print("Token lengths: min=%d, max=%d, mean=%.1f" % (
        min(tok_lens), max(tok_lens), np.mean(tok_lens)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-root", required=True,
                    help="Output of convert_trumans.py (has trumans_263_cache, trumans_track2, trumans_meta.pkl)")
    ap.add_argument("--scene-mesh-dir", default=None,
                    help="Path to TRUMANS Scene_mesh/ with .obj files")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tokenize", action="store_true",
                    help="Run phase 2 (GPU tokenization). Requires --vqvae-ckpt")
    ap.add_argument("--vqvae-ckpt", default=None,
                    help="VQ-VAE checkpoint for tokenization")
    args = ap.parse_args()

    if args.tokenize:
        if not args.vqvae_ckpt:
            print("ERROR: --tokenize requires --vqvae-ckpt")
            sys.exit(1)
        tokenize(args)
    else:
        if not args.scene_mesh_dir:
            print("WARNING: --scene-mesh-dir not given, occ_crop will be zeros")
        build_geometric(args)


if __name__ == "__main__":
    main()
