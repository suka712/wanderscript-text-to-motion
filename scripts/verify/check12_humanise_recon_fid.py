#!/usr/bin/env python3
"""
Per-category reconstruction FID on HUMANISE (walk / stand up / sit / lie),
through the frozen T2M-GPT VQ-VAE. Extends Check 3's MPJPE canary
(scripts/verify/check7_vqvae_canary.py) with the SAME convention-independent
metric (recon FID) that Step 2 Task 2 just validated against the paper on
H3D (ours 0.066 vs paper 0.070) -- see docs/STEP2_baseline_calibration.md.

Why this matters: MPJPE isn't what the paper reports, so Check 3's raw
sit/lie MPJPE numbers (293mm / 703mm) aren't directly comparable to anything
external. FID is. This gives per-category numbers on the same footing as
every other FID reported in this project, and doubles as the pre-finetune
baseline Step 3 (VQ-VAE joint finetune) needs to show it improved anything.

Uses T2M-GPT's own EvaluatorModelWrapper.get_motion_embeddings (motion-only,
no text/caption needed -- HUMANISE utterances would need spacy POS-tagging
to go through the text branch, which get_motion_embeddings sidesteps
entirely since FID only needs the motion-embedding side). Reuses
calculate_activation_statistics / calculate_frechet_distance from
T2M-GPT's utils/eval_trans.py unmodified, same as VQ_eval.py does.

Repeats 3x per category (bootstrap resample of clips within the category),
matching the repeat_time=3 convention already used for Step 2 Task 1/2 under
GPU contention -- reports mean +/- 1.96*std/sqrt(3) in the same format.
"""
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

T2M_GPT_ROOT = os.environ.get("WANDER_T2M_GPT_ROOT", "/home/user/Khiem-ssh/T2M-GPT")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import motion_features as mf  # noqa: E402
from humanise_join import build_flat_join  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402

HUMANISE_MOTIONS = os.environ.get("WANDER_HUMANISE_ROOT", "/media/user/2tb/motion_data/HUMANISE") + "/contact_motion/motions"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_T = 196
UNIT_LENGTH = 4
CAP_PER_CATEGORY = 2000  # FID needs n >> embedding dim (512) for a stable covariance
                         # estimate -- the paper's own eval aggregates ~4384 clips for
                         # exactly this reason. 2000 fits within the smallest category
                         # (lie, 2343 clips) so all four categories use a comparable n.
N_REPEATS = 3


def crop_to_multiple(T, factor=UNIT_LENGTH, max_t=MAX_T):
    T = min(T, max_t)
    return (T // factor) * factor


def encode_decode(net, data263: np.ndarray, mean, std):
    """Returns (motion_norm, recon_norm): both (T, 263), Z-normalized (H3D
    mean/std), T cropped to a multiple of UNIT_LENGTH. None if too short."""
    T = crop_to_multiple(data263.shape[0])
    if T < UNIT_LENGTH:
        return None
    data263 = data263[:T]
    norm = (data263 - mean) / std
    x = torch.from_numpy(norm).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        x_out, _, _ = net(x)
    return norm.astype(np.float32), x_out[0].cpu().numpy().astype(np.float32)


def category_clip_indices():
    flat = build_flat_join()
    by_action = {}
    for i, p in enumerate(flat):
        by_action.setdefault(p["action"], []).append(i)
    return by_action


MINI_BATCH = 32  # matches VQ_eval.py's own batch-size convention; keeps peak GPU
                 # memory low and comparable to the paper's own per-batch protocol
                 # instead of one huge padded batch (which is what pushed total GPU
                 # memory over the safety threshold and got the job killed once).


def _mini_batches(chosen, n):
    for start in range(0, len(chosen), n):
        yield chosen[start : start + n]


def category_embeddings(eval_wrapper, net, mean, std, indices, rng, n):
    """Samples up to n clips from indices, processes them through the VQ-VAE
    and the evaluator's motion encoder in small mini-batches (not one big
    padded batch), and returns concatenated (em_real, em_recon) embeddings."""
    chosen = rng.choice(indices, size=min(n, len(indices)), replace=False)
    em_real_all, em_recon_all = [], []

    for mb in _mini_batches(chosen, MINI_BATCH):
        reals, recons, lengths = [], [], []
        for i in mb:
            cm = np.load(f"{HUMANISE_MOTIONS}/{i:05d}.npy")
            if cm.shape[0] < 6:
                continue
            data263, _, _, _ = mf.humanise_positions_to_263(cm)
            result = encode_decode(net, data263.astype(np.float32), mean, std)
            if result is None:
                continue
            norm, recon = result
            reals.append(norm)
            recons.append(recon)
            lengths.append(norm.shape[0])
        if not reals:
            continue

        max_t = max(lengths)
        real_batch = np.zeros((len(reals), max_t, 263), dtype=np.float32)
        recon_batch = np.zeros((len(recons), max_t, 263), dtype=np.float32)
        for j, (r, c, L) in enumerate(zip(reals, recons, lengths)):
            real_batch[j, :L] = r
            recon_batch[j, :L] = c
        real_t = torch.from_numpy(real_batch)
        recon_t = torch.from_numpy(recon_batch)
        len_t = torch.tensor(lengths, dtype=torch.long)

        em_real_all.append(eval_wrapper.get_motion_embeddings(real_t, len_t).cpu().numpy())
        em_recon_all.append(eval_wrapper.get_motion_embeddings(recon_t, len_t).cpu().numpy())

    if not em_real_all:
        return None
    return np.concatenate(em_real_all, axis=0), np.concatenate(em_recon_all, axis=0)


def category_fid(eval_wrapper, net, mean, std, indices, rng):
    result = category_embeddings(eval_wrapper, net, mean, std, indices, rng, CAP_PER_CATEGORY)
    if result is None:
        return None, 0
    em_real, em_recon = result

    import eval_trans

    mu_r, cov_r = eval_trans.calculate_activation_statistics(em_real)
    mu_p, cov_p = eval_trans.calculate_activation_statistics(em_recon)
    # Proactive regularization (not just T2M-GPT's own reactive fallback, which
    # only catches non-finite sqrtm output, not finite-but-badly-conditioned
    # results): badly-reconstructed categories (lie) produce a recon-embedding
    # covariance whose matrix product sqrtm is numerically unstable even
    # though every value involved is finite. This is a standard FID numerical
    # fix, done here rather than by editing T2M-GPT's vendored eval_trans.py.
    eps = 1e-6
    cov_r = cov_r + np.eye(cov_r.shape[0]) * eps
    cov_p = cov_p + np.eye(cov_p.shape[0]) * eps
    fid = eval_trans.calculate_frechet_distance(mu_r, cov_r, mu_p, cov_p)
    return fid, em_real.shape[0]


def main():
    net = load_vqvae(device=DEVICE)
    print(f"VQ-VAE loaded on {DEVICE}")

    # T2M-GPT's evaluator wrapper resolves paths (checkpoints/, glove/)
    # relative to its own repo root.
    os.chdir(T2M_GPT_ROOT)
    sys.path.insert(0, T2M_GPT_ROOT)
    sys.path.insert(0, f"{T2M_GPT_ROOT}/utils")
    from models.evaluator_wrapper import EvaluatorModelWrapper
    from options.get_eval_option import get_opt

    wrapper_opt = get_opt("checkpoints/t2m/Comp_v6_KLD005/opt.txt", torch.device(DEVICE))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    # CRITICAL: the evaluator's motion encoder was trained expecting ITS OWN
    # normalization stats, not H3D_ROOT's Mean.npy/Std.npy (they differ: max
    # abs diff 0.028 in mean, 0.34 in std -- enough to badly distort every
    # embedding, real or reconstructed, uniformly). Confirmed by a control
    # test: running this exact pipeline on H3D itself gave FID=5.25 with
    # H3D_ROOT's stats vs FID=0.072 (matches the paper) with these. Official
    # T2M-GPT eval code loads this same file (dataset_TM_eval.py's
    # Text2MotionDatasetEval.meta_dir) for exactly this reason -- use it for
    # BOTH the VQ-VAE encode/decode input and the evaluator embedding step.
    mean = np.load("checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy").astype(
        np.float32
    )
    std = np.load("checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy").astype(
        np.float32
    )

    by_action = category_clip_indices()
    print({a: len(idxs) for a, idxs in by_action.items()})

    print(f"{'category':<12} {'FID (3 reps)':<20} {'n_clips/rep'}")
    results = {}
    for action in ["walk", "stand up", "sit", "lie"]:
        indices = by_action.get(action, [])
        if not indices:
            print(f"{action:<12} no clips found")
            continue
        fids = []
        n_clips = 0
        for rep in range(N_REPEATS):
            rng = np.random.RandomState(rep)
            fid, n = category_fid(eval_wrapper, net, mean, std, indices, rng)
            if fid is not None:
                fids.append(fid)
                n_clips = n
        fids = np.array(fids)
        mean_fid = fids.mean()
        conf = fids.std() * 1.96 / np.sqrt(len(fids)) if len(fids) > 1 else float("nan")
        results[action] = (mean_fid, conf, n_clips)
        print(f"{action:<12} {mean_fid:.3f} +/- {conf:.3f}{'':<8} {n_clips}")

    print("\nSUMMARY (recon FID, walk-relative ratio):")
    walk_fid = results.get("walk", (None,))[0]
    for action, (fid, conf, n) in results.items():
        ratio = fid / walk_fid if walk_fid else float("nan")
        print(f"  {action:<10} FID {fid:.3f} +/- {conf:.3f}  ({ratio:.2f}x walk, n={n})")


if __name__ == "__main__":
    main()
