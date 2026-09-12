#!/usr/bin/env python3
"""Re-extract tokens with a scene-grounded VQ-VAE, reusing geometric fields from an existing
manifest.

Only `tokens` depends on the tokenizer; prefix_pose, occ_crop, xy_traj, start, goal, action, text
are geometric and tokenizer-INDEPENDENT. We load the existing manifest, re-encode each clip's 263
with the new encoder+quantizer, and SWAP the tokens field.

Supports mixed HUMANISE+TRUMANS manifests: TRUMANS clips (identified by having a `scene` key) load
263 from the TRUMANS 263 cache; HUMANISE clips load from contact_motion and convert to 263.
"""
import argparse
import os
import pickle
import sys

import numpy as np
import torch

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "track1"))
import motion_features as mf  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402
from scene_vqvae import SceneVQVAE  # noqa: E402
from prepare_probe_data import crop_to_multiple  # noqa: E402

HUMANISE = os.environ.get("WANDER_HUMANISE_ROOT")
T2M = os.environ.get("WANDER_T2M_GPT_ROOT")
TRUMANS_263_CACHE = os.environ.get(
    "WANDER_TRUMANS_263_CACHE",
    "/media/user/2tb/motion_data/TRUMANS_processed/trumans_263_cache",
)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _load_263(rec):
    """Load the 263-dim motion for a manifest record, dispatching by source."""
    idx = rec["index"]
    if "scene" in rec:
        # TRUMANS clip — 263 already computed in cache
        return np.load(os.path.join(TRUMANS_263_CACHE, f"{idx:05d}.npy")).astype(np.float32)
    else:
        # HUMANISE clip — derive 263 from contact_motion
        cm = np.load(os.path.join(HUMANISE, "contact_motion", "motions", f"{idx:05d}.npy"))
        d263, *_ = mf.humanise_positions_to_263(cm)
        return d263


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-vqvae", required=True)
    ap.add_argument("--base-vqvae", default=os.path.expanduser(
        "~/Khiem/T2M-GPT/pretrained/VQVAE/net_best_fid.pth"),
        help="base VQ-VAE for architecture (weights overwritten by scene-vqvae state)")
    ap.add_argument("--src-tokens", default=os.path.expanduser(
        "~/wander_data/trumans_combined_tokens"),
        help="existing manifest dir (train.pkl and optionally test.pkl)")
    ap.add_argument("--out", required=True, help="output tokens dir")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    base = load_vqvae(args.base_vqvae, device=DEV)
    scene = SceneVQVAE(base).to(DEV)
    scene.load_state_dict(torch.load(args.scene_vqvae, map_location=DEV)["net"])
    scene.eval()
    mean = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy").astype(np.float32)
    std = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy").astype(np.float32)

    n_changed = n_len = n_hum = n_tru = 0
    with torch.no_grad():
        for split in ["train", "test"]:
            src_path = os.path.join(args.src_tokens, f"{split}.pkl")
            if not os.path.exists(src_path):
                print(f"{split}: skipped (no {src_path})", flush=True)
                continue
            with open(src_path, "rb") as f:
                manifest = pickle.load(f)
            for rec in manifest:
                d263 = _load_263(rec)
                is_trumans = "scene" in rec
                if is_trumans:
                    n_tru += 1
                else:
                    n_hum += 1
                old_tok = rec["tokens"]
                # Reproduce the exact crop: T = len(old_tokens) * 4 frames, so the new tokens
                # describe the SAME motion the reused geometric fields were computed for.
                T = len(old_tok) * 4
                norm = (d263[:T].astype(np.float32) - mean) / std
                x = torch.from_numpy(norm).unsqueeze(0).to(DEV)
                new_tok = scene.encode(x)[0].cpu().numpy().astype(np.int64)
                if len(new_tok) != len(old_tok):
                    n_len += 1
                    L = min(len(new_tok), len(old_tok))
                    new_tok = new_tok[:L]
                if not np.array_equal(new_tok, old_tok[:len(new_tok)]):
                    n_changed += 1
                rec["tokens"] = new_tok
            with open(os.path.join(args.out, f"{split}.pkl"), "wb") as f:
                pickle.dump(manifest, f)
            print(f"{split}: {len(manifest)} clips re-tokenized -> {args.out}/{split}.pkl", flush=True)
    print(f"changed: {n_changed}  len-mismatch: {n_len}  humanise: {n_hum}  trumans: {n_tru}")


if __name__ == "__main__":
    main()
