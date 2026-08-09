#!/usr/bin/env python3
"""
H3D-lie control: is HUMANISE-lie's 139.8mm MPJPE (vs. 45.3mm H3D baseline,
3.09x) actually about the HUMANISE data/motion category, or would H3D's OWN
lying-down clips show similarly elevated error under the exact same frozen
VQ-VAE? If H3D-lie is also elevated relative to the H3D baseline, that's
evidence the effect is about "lying down is intrinsically harder to
reconstruct" (pose ambiguity, less training-distribution mass, whatever),
not something specific to HUMANISE's SMPL-X-derived data. If H3D-lie
reconstructs close to the H3D baseline, that would instead point at
something HUMANISE-specific (its data prep, its motion distribution).

H3D has no per-clip action/category label (unlike HUMANISE's action=lie/sit/
walk/stand-up taxonomy) -- clips are identified here by text-description
keyword search (same technique as check10_h3d_walk_renders.py's 'walk'
search), matching lying/laying phrasing. Same methodology as check14's H3D
baseline (45.3mm): evaluator-consistent normalization (VQ-VAE checkpoint's
own meta mean/std, NOT H3D_ROOT's Mean.npy/Std.npy), same frozen VQ-VAE,
same local (root-relative) MPJPE metric -- so this number is directly
comparable to check14's H3D-baseline and HUMANISE-lie numbers.
"""
import os
import re
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import motion_features as mf  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402

T2M_GPT_ROOT = os.environ.get("WANDER_T2M_GPT_ROOT", "/home/user/Khiem-ssh/T2M-GPT")
H3D_ROOT = os.environ.get("WANDER_H3D_ROOT", "/media/user/2tb/motion_data/H3D")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_T = 196

LIE_PATTERN = re.compile(
    r"\blying\b|\blie down\b|\blies down\b|\blie on\b|\blies on\b|"
    r"\blaying\b|\blay down\b|\blays down\b|\blay on\b"
)

# Reference numbers from check14_mpjpe_evaluator_norm.py (evaluator-consistent
# norm, same VQ-VAE, same metric) -- for direct comparison, not re-derived here.
H3D_BASELINE_MM = 45.3
HUMANISE_LIE_MM = 139.8


def crop_to_multiple(T, factor=4, max_t=MAX_T):
    T = min(T, max_t)
    return (T // factor) * factor


def mpjpe(pos_a, pos_b):
    T = min(pos_a.shape[0], pos_b.shape[0])
    return np.linalg.norm(pos_a[:T] - pos_b[:T], axis=-1).mean()


def roundtrip_mpjpe(net, data263, mean, std):
    T = crop_to_multiple(data263.shape[0])
    if T < 4:
        return None
    data263 = data263[:T].astype(np.float32)
    norm = (data263 - mean) / std
    x = torch.from_numpy(norm).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        x_out, _, _ = net(x)
    recon263 = x_out[0].cpu().numpy() * std + mean
    orig_local = mf.local_joint_positions(data263)
    recon_local = mf.local_joint_positions(recon263.astype(np.float32))
    return mpjpe(orig_local, recon_local)


def find_h3d_lie_clips():
    files = sorted(os.listdir(f"{H3D_ROOT}/new_joint_vecs"))
    files = [f for f in files if not f.startswith("M")]  # skip mirrored-augmentation copies
    matches = []
    for f in files:
        name = f[:-4]
        text_path = f"{H3D_ROOT}/texts/{name}.txt"
        if not os.path.exists(text_path):
            continue
        with open(text_path) as tf:
            text = tf.read().lower()
        if LIE_PATTERN.search(text):
            matches.append(name)
    return matches


def main():
    net = load_vqvae(device=DEVICE)
    mean = np.load(f"{T2M_GPT_ROOT}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy").astype(
        np.float32
    )
    std = np.load(f"{T2M_GPT_ROOT}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy").astype(
        np.float32
    )
    print(f"VQ-VAE loaded on {DEVICE}, evaluator-consistent normalization")

    lie_ids = find_h3d_lie_clips()
    print(f"Found {len(lie_ids)} H3D clips with lying/laying text description")

    errs = []
    skipped = 0
    for name in lie_ids:
        data263 = np.load(f"{H3D_ROOT}/new_joint_vecs/{name}.npy").astype(np.float32)
        e = roundtrip_mpjpe(net, data263, mean, std)
        if e is None:
            skipped += 1
            continue
        errs.append((name, e))

    errs_arr = np.array([e for _, e in errs])
    print(f"Evaluated {len(errs)} clips (skipped too-short: {skipped})")
    print(f"H3D-lie MPJPE: mean={errs_arr.mean()*1000:.2f}mm  median={np.median(errs_arr)*1000:.2f}mm  "
          f"std={errs_arr.std()*1000:.2f}mm  min={errs_arr.min()*1000:.2f}mm  max={errs_arr.max()*1000:.2f}mm")

    print()
    print("Per-clip (sorted worst to best):")
    for name, e in sorted(errs, key=lambda x: -x[1])[:10]:
        with open(f"{H3D_ROOT}/texts/{name}.txt") as f:
            caption = f.readline().split("#")[0]
        print(f"  {name}: {e*1000:6.1f}mm  \"{caption.strip()[:70]}\"")

    print()
    print("=" * 60)
    print("COMPARISON (all evaluator-consistent norm, same VQ-VAE, same local MPJPE metric):")
    print(f"  H3D baseline (general, n=200, check14): {H3D_BASELINE_MM:.1f}mm")
    print(f"  H3D-lie      (this script, n={len(errs)}):        {errs_arr.mean()*1000:.1f}mm  "
          f"({errs_arr.mean()*1000/H3D_BASELINE_MM:.2f}x H3D baseline)")
    print(f"  HUMANISE-lie (check14, n~subset of 200):  {HUMANISE_LIE_MM:.1f}mm  "
          f"({HUMANISE_LIE_MM/H3D_BASELINE_MM:.2f}x H3D baseline)")


if __name__ == "__main__":
    main()
