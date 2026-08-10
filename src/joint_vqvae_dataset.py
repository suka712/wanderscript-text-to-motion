"""
Track 2 -- joint balanced dataset/dataloader for VQ-VAE finetuning on
HumanML3D + HUMANISE (see docs/track_2/001_tokenizer_finetune.md).

Uses the OFFICIAL train/test splits shipped with each dataset (H3D_ROOT/
{train,test}.txt, HUMANISE_ROOT/{train,test}.txt) so finetune-time training
and eval never touch the same clips -- unlike the frozen-checkpoint canary
scripts (check7/14/15/16/17/20), which sampled randomly from the full 19,648
HUMANISE clips with no held-out split (fine for a diagnostic on a FROZEN
model that never trains on that data; not fine once we're the ones training
on it).

Windowing/normalization contract mirrors T2M-GPT's own dataset_VQ.py
(VQMotionDataset) exactly -- same window_size, same "normalize with a fixed
mean/std, random crop per __getitem__" recipe -- so the training loop and
loss functions need no changes to consume this.

HUMANISE clips must be pre-converted to 263-dim and cached first via
scripts/track2/precompute_humanise_263.py (run once, ~11 min, resumable).
"""
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

H3D_ROOT = os.environ.get("WANDER_H3D_ROOT", "/media/user/2tb/motion_data/H3D")
HUMANISE_ROOT = os.environ.get("WANDER_HUMANISE_ROOT", "/media/user/2tb/motion_data/HUMANISE")
HUMANISE_263_CACHE = os.environ.get(
    "WANDER_HUMANISE_263_CACHE", "/media/user/2tb/motion_data/HUMANISE_263_cache"
)


def _read_ids(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


class WindowMotionDataset(Dataset):
    """Generic windowed 263-dim motion dataset over a list of pre-loaded
    (T, 263) float32 arrays (T >= window_size guaranteed by the loader
    functions below). Random-crops a window and Z-normalizes per __getitem__,
    same contract as T2M-GPT's dataset_VQ.VQMotionDataset.
    """

    def __init__(self, motions, mean, std, window_size=64):
        self.motions = motions
        self.mean = mean
        self.std = std
        self.window_size = window_size

    def __len__(self):
        return len(self.motions)

    def __getitem__(self, idx):
        motion = self.motions[idx]
        start = random.randint(0, len(motion) - self.window_size)
        window = motion[start : start + self.window_size]
        return ((window - self.mean) / self.std).astype(np.float32)


def load_h3d_split(split, window_size=64, verbose=True):
    """Loads every H3D clip listed in {split}.txt into memory (mirrors
    T2M-GPT's VQMotionDataset load pattern -- proven to fit in RAM on this
    box, ~4.2GB on disk for the full new_joint_vecs dir). Skips non-finite
    clips (the 2 known NaN-corrupted files, 007975/M007975 -- see CLAUDE.md
    section 3 -- plus a defensive general check) and clips shorter than
    window_size.
    """
    ids = _read_ids(f"{H3D_ROOT}/{split}.txt")
    motions = []
    n_skipped_short = 0
    n_skipped_nan = 0
    n_missing = 0
    for name in ids:
        path = f"{H3D_ROOT}/new_joint_vecs/{name}.npy"
        if not os.path.exists(path):
            n_missing += 1
            continue
        m = np.load(path).astype(np.float32)
        if not np.isfinite(m).all():
            n_skipped_nan += 1
            continue
        if m.shape[0] < window_size:
            n_skipped_short += 1
            continue
        motions.append(m)
    stats = {
        "requested": len(ids),
        "loaded": len(motions),
        "skipped_short": n_skipped_short,
        "skipped_nan": n_skipped_nan,
        "missing": n_missing,
    }
    if verbose:
        print(f"[H3D:{split}] loaded {stats['loaded']}/{stats['requested']} "
              f"(short={n_skipped_short} nan={n_skipped_nan} missing={n_missing})")
    return motions, stats


def load_humanise_split(split, window_size=64, cache_dir=HUMANISE_263_CACHE, verbose=True):
    """Loads every HUMANISE clip listed in {split}.txt (6-digit ids, e.g.
    '000000') from the precomputed 263-dim cache. Run
    scripts/track2/precompute_humanise_263.py first.
    """
    ids = _read_ids(f"{HUMANISE_ROOT}/{split}.txt")
    motions = []
    n_skipped_short = 0
    n_skipped_nan = 0
    n_missing_cache = 0
    for name in ids:
        idx = int(name)
        path = f"{cache_dir}/{idx:05d}.npy"
        if not os.path.exists(path):
            n_missing_cache += 1
            continue
        m = np.load(path).astype(np.float32)
        if not np.isfinite(m).all():
            n_skipped_nan += 1
            continue
        if m.shape[0] < window_size:
            n_skipped_short += 1
            continue
        motions.append(m)
    stats = {
        "requested": len(ids),
        "loaded": len(motions),
        "skipped_short": n_skipped_short,
        "skipped_nan": n_skipped_nan,
        "missing_cache": n_missing_cache,
    }
    if verbose:
        print(f"[HUMANISE:{split}] loaded {stats['loaded']}/{stats['requested']} "
              f"(short={n_skipped_short} nan={n_skipped_nan} missing_cache={n_missing_cache})")
    return motions, stats


def _cycle(loader):
    while True:
        for x in loader:
            yield x


class BalancedJointLoader:
    """Cycles two DataLoaders (H3D, HUMANISE) forever, yielding batches split
    h3d_frac : (1 - h3d_frac) between them (default 1:1) -- CLAUDE.md-mandated
    balanced sampling so neither dataset dominates regardless of underlying
    size skew (H3D train ~23.4k incl. mirror augmentation vs HUMANISE train
    16.5k). Each sub-batch is drawn with replacement across epochs (standard
    shuffle=True + infinite cycling), so the exact size ratio of the two
    source datasets doesn't matter -- only h3d_frac controls the mix.
    """

    def __init__(self, h3d_dataset, humanise_dataset, batch_size, h3d_frac=0.5,
                 num_workers=4, seed=0):
        n_h3d = max(1, round(batch_size * h3d_frac))
        n_hum = max(1, batch_size - n_h3d)
        self.n_h3d, self.n_hum = n_h3d, n_hum
        self.batch_size = n_h3d + n_hum
        g1 = torch.Generator().manual_seed(seed)
        g2 = torch.Generator().manual_seed(seed + 1)
        self.h3d_loader = DataLoader(
            h3d_dataset, batch_size=n_h3d, shuffle=True,
            num_workers=num_workers, drop_last=True, generator=g1,
            persistent_workers=num_workers > 0,
        )
        self.hum_loader = DataLoader(
            humanise_dataset, batch_size=n_hum, shuffle=True,
            num_workers=num_workers, drop_last=True, generator=g2,
            persistent_workers=num_workers > 0,
        )

    def __iter__(self):
        h3d_it = _cycle(self.h3d_loader)
        hum_it = _cycle(self.hum_loader)
        while True:
            a = next(h3d_it)
            b = next(hum_it)
            yield torch.cat([a, b], dim=0)
