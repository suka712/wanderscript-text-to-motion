#!/usr/bin/env python
"""
Evaluate seat-height correctness: does the model generate sits at different
heights when the heightmap shows different seat surfaces?

Metric: correlation between the heightmap's center surface height (the seat the
model is told about) and the generated pelvis height at the END of the sit segment.
A model that ignores the heightmap produces a FIXED seated pelvis (~0.5-0.6 m on
HUMANISE) regardless of the commanded seat height; a model that uses it produces
corr > 0 (pelvis tracks the surface).

For each run:
  - Sample N sit segments from the TRUMANS held-out set (or a test manifest with
    diverse seat heights).
  - For each, generate a sit segment conditioned on the entry's occ, heightmap,
    action=sit, and a near goal (within furniture reach).
  - Decode and place: pelvis height = world-frame pelvis Y at the last frame.
  - Compute corr(seat_height, pelvis_height).

Also reports:
  - Mean absolute contact gap |pelvis - seat| (lower = better).
  - Scatter data for plotting.

Usage:
  python eval_seat_height.py \
    --ckpt ~/wander_data/step_hm/checkpoints/action_hm \
    --vqvae-ckpt ~/wander_data/motion_data/track2_checkpoints/net_iter020000.pth \
    --tokens-dir ~/wander_data/trumans_combined_tokens \
    --manifest train_hm.pkl \
    --n 100 --seed 0
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
for p in ["src", "scripts/track1", "scripts/chaining", "scripts/scene_probe"]:
    sys.path.insert(0, os.path.join(REPO_ROOT, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint dir (norm_stats.json + net_final.pth)")
    ap.add_argument("--vqvae-ckpt", required=True, help="finetuned VQ-VAE checkpoint")
    ap.add_argument("--tokens-dir", required=True, help="dir with train_hm.pkl (or --manifest)")
    ap.add_argument("--manifest", default="train_hm.pkl", help="manifest file inside tokens-dir")
    ap.add_argument("--n", type=int, default=100, help="number of sit clips to evaluate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output dir for scatter data (optional)")
    args = ap.parse_args()

    import torch
    import clip
    from vqvae_loader import load_vqvae
    from rollout import load_model, build_cond, yaw_from_joints, OCC_N
    import motion_features as mf
    from se2_utils import se2_place_full_body

    T2M_GPT_ROOT = os.environ.get("WANDER_T2M_GPT_ROOT", "/home/user/Khiem-ssh/T2M-GPT")
    DEV = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    trans, ns = load_model(args.ckpt)
    net = load_vqvae(args.vqvae_ckpt, DEV)
    clip_model, _ = clip.load("ViT-B/32", device=DEV, jit=False)
    clip_model.eval()
    import clip as clip_mod

    cmean = np.array(ns["cond_mean"], np.float32)
    cstd = np.array(ns["cond_std"], np.float32)
    vqvae_mean = np.load(os.path.join(
        T2M_GPT_ROOT, "checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy")).astype(np.float32)
    vqvae_std = np.load(os.path.join(
        T2M_GPT_ROOT, "checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy")).astype(np.float32)

    # Load manifest
    manifest_path = os.path.join(args.tokens_dir, args.manifest)
    manifest = pickle.load(open(manifest_path, "rb"))
    print(f"Manifest: {len(manifest)} entries from {manifest_path}")

    # Filter to sit clips with valid heightmaps (non-zero center surface)
    sit_clips = []
    for d in manifest:
        if d.get("action") != "sit":
            continue
        hm = d.get("heightmap_1024")
        if hm is None:
            continue
        # Center surface height (14:18 of 32x32 = center 4x4)
        hm_2d = hm.reshape(32, 32)
        center_h = float(hm_2d[14:18, 14:18].mean())
        if center_h > 0.1:  # has actual seat surface (not floor)
            sit_clips.append((d, center_h))

    print(f"Sit clips with seat surface (center_h > 0.1): {len(sit_clips)}")
    if len(sit_clips) == 0:
        print("ERROR: no sit clips with visible seat surface in the heightmap")
        return

    rng = np.random.RandomState(args.seed)
    rng.shuffle(sit_clips)
    sit_clips = sit_clips[:args.n]
    print(f"Evaluating {len(sit_clips)} clips")

    # Seat height range
    seat_heights = [ch for _, ch in sit_clips]
    print(f"Seat height range: {min(seat_heights):.3f} - {max(seat_heights):.3f} "
          f"(std={np.std(seat_heights):.3f})")

    results = []
    with torch.no_grad():
        for i, (d, seat_h) in enumerate(sit_clips):
            # Build conditioning (same as training: start-frame features)
            try:
                from train_probe import cond_extra_raw, ACTION_IDS
                raw = cond_extra_raw(d, ns["cond_mode"])
                extra = ((raw - cmean) / cstd).astype(np.float32)

                # Encode text
                feat = clip_model.encode_text(
                    clip_mod.tokenize([d["text"]], truncate=True).to(DEV)).float()
                cond = torch.cat([feat, torch.from_numpy(extra).unsqueeze(0).to(DEV)], -1)

                # Generate
                tok = trans.sample(cond, if_categorial=False)
                tok = tok[tok < 512]
                if len(tok) < 2:
                    continue

                # Decode
                decoded = net.forward_decoder(tok.unsqueeze(0))
                local_motion = (decoded[0].cpu().numpy() * vqvae_std + vqvae_mean)

                # Place in world frame: se2_place_full_body(data263, start_pose, mf_module)
                start_pose = d["start"]
                world_joints = se2_place_full_body(local_motion, start_pose, mf)

                # Pelvis height at end of segment (world frame Z for Z-up, Y for Y-up)
                # Our pipeline is Z-up (ScanNet frame), so pelvis Z = height
                pelvis_end_z = float(world_joints[-1, 0, 2])

                results.append({
                    "seat_height": seat_h,
                    "pelvis_height": pelvis_end_z,
                    "gap": abs(pelvis_end_z - seat_h),
                    "text": d["text"],
                })

                if (i + 1) % 20 == 0:
                    print(f"  [{i+1}/{len(sit_clips)}] seat_h={seat_h:.3f} "
                          f"pelvis={pelvis_end_z:.3f} gap={abs(pelvis_end_z-seat_h):.3f}")
            except Exception as e:
                print(f"  clip {i} failed: {e}")
                continue

    if not results:
        print("ERROR: no successful generations")
        return

    seat_arr = np.array([r["seat_height"] for r in results])
    pelvis_arr = np.array([r["pelvis_height"] for r in results])
    gap_arr = np.array([r["gap"] for r in results])

    corr = float(np.corrcoef(seat_arr, pelvis_arr)[0, 1])
    print(f"\n{'='*60}")
    print(f"RESULTS ({len(results)} clips):")
    print(f"  seat height:   mean={seat_arr.mean():.3f}  std={seat_arr.std():.3f}")
    print(f"  pelvis height: mean={pelvis_arr.mean():.3f}  std={pelvis_arr.std():.3f}")
    print(f"  corr(seat, pelvis): {corr:.3f}")
    print(f"  mean |gap|: {gap_arr.mean():.3f} m")
    print(f"  median |gap|: {np.median(gap_arr):.3f} m")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        np.savez(os.path.join(args.out, "seat_height_eval.npz"),
                 seat=seat_arr, pelvis=pelvis_arr, gap=gap_arr)
        print(f"  Scatter data saved to {args.out}/seat_height_eval.npz")


if __name__ == "__main__":
    main()
