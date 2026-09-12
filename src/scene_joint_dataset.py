"""Joint H3D+HUMANISE(+TRUMANS) dataset for the geometry-grounded VQ-VAE finetune: like
joint_vqvae_dataset but each item is (window_263_normalized, window_heightmap).

HUMANISE/TRUMANS clips carry their precomputed per-frame local heightmap (extract_heightmaps.py),
cropped with the SAME random window as the motion. H3D clips have no scene -> a flat-floor
heightmap (zeros). The crop alignment is load-bearing: motion frame t and heightmap frame t must
be the same physical frame (offset 0, verified 2026-08-28), so the crop uses one shared `start`.

TRUMANS (added 2026-09-12): 6200 clips with seat-height variety (σ=0.108, range 0.38–0.85 m vs
HUMANISE's near-zero variation). Same (263, heightmap) format as HUMANISE after conversion by
scripts/trumans/convert_trumans.py + extract_heightmaps.py. No train/test split file — all
clips are loaded and the caller controls the split (or uses all for training, since TRUMANS is
supplementary data for seat-height diversity, not a benchmark).
"""
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from joint_vqvae_dataset import (  # noqa: F401  reuse the tested loaders/splits
    H3D_ROOT, HUMANISE_ROOT, HUMANISE_263_CACHE, _read_ids, load_h3d_split,
)
from scene_heightmap import GRID_N

HUMANISE_HM_CACHE = os.environ.get(
    "WANDER_HUMANISE_HM_CACHE",
    os.path.expanduser("~/wander_data/motion_data/HUMANISE_heightmap_cache"),
)

TRUMANS_263_CACHE = os.environ.get(
    "WANDER_TRUMANS_263_CACHE",
    "/media/user/2tb/motion_data/TRUMANS_processed/trumans_263_cache",
)
TRUMANS_HM_CACHE = os.environ.get(
    "WANDER_TRUMANS_HM_CACHE",
    os.path.expanduser("~/wander_data/trumans_heightmap_cache"),
)


def load_humanise_split_with_hm(split, window_size=64, cache_263=HUMANISE_263_CACHE,
                                cache_hm=HUMANISE_HM_CACHE, verbose=True):
    """Loads HUMANISE (263, heightmap) pairs for `split`. A clip is kept only if BOTH its 263 and
    its heightmap exist, are finite, length-match, and are >= window_size. Returns
    (motions, heightmaps, stats) index-aligned."""
    ids = _read_ids(f"{HUMANISE_ROOT}/{split}.txt")
    motions, hms = [], []
    n_short = n_nan = n_missing263 = n_missinghm = n_mismatch = 0
    for name in ids:
        idx = int(name)
        p263 = f"{cache_263}/{idx:05d}.npy"
        phm = f"{cache_hm}/{idx:05d}.npy"
        if not os.path.exists(p263):
            n_missing263 += 1
            continue
        if not os.path.exists(phm):
            n_missinghm += 1
            continue
        m = np.load(p263).astype(np.float32)
        h = np.load(phm)  # float16 (T,32,32)
        if m.shape[0] != h.shape[0]:
            n_mismatch += 1
            continue
        if not np.isfinite(m).all():
            n_nan += 1
            continue
        if m.shape[0] < window_size:
            n_short += 1
            continue
        motions.append(m)
        hms.append(h)  # keep float16 to save RAM; cast per __getitem__
    stats = dict(requested=len(ids), loaded=len(motions), short=n_short, nan=n_nan,
                 missing263=n_missing263, missing_hm=n_missinghm, mismatch=n_mismatch)
    if verbose:
        print(f"[HUMANISE+HM:{split}] loaded {stats['loaded']}/{stats['requested']} "
              f"(short={n_short} nan={n_nan} miss263={n_missing263} misshm={n_missinghm} "
              f"mismatch={n_mismatch})")
    return motions, hms, stats


def load_trumans_with_hm(window_size=64, cache_263=TRUMANS_263_CACHE,
                         cache_hm=TRUMANS_HM_CACHE, verbose=True):
    """Loads all TRUMANS (263, heightmap) pairs. No split file — all 6200 clips are loaded.
    Same contract as load_humanise_split_with_hm."""
    motions, hms = [], []
    n_short = n_nan = n_missinghm = n_mismatch = 0
    # TRUMANS ids are 0-based, 5-digit
    all_263 = sorted(f for f in os.listdir(cache_263) if f.endswith(".npy"))
    for fname in all_263:
        idx = int(fname[:5])
        p263 = os.path.join(cache_263, fname)
        phm = os.path.join(cache_hm, fname)
        if not os.path.exists(phm):
            n_missinghm += 1
            continue
        m = np.load(p263).astype(np.float32)
        h = np.load(phm)  # float16 (T,32,32)
        if m.shape[0] != h.shape[0]:
            n_mismatch += 1
            continue
        if not np.isfinite(m).all():
            n_nan += 1
            continue
        if m.shape[0] < window_size:
            n_short += 1
            continue
        motions.append(m)
        hms.append(h)
    stats = dict(requested=len(all_263), loaded=len(motions), short=n_short, nan=n_nan,
                 missing_hm=n_missinghm, mismatch=n_mismatch)
    if verbose:
        print(f"[TRUMANS+HM] loaded {stats['loaded']}/{stats['requested']} "
              f"(short={n_short} nan={n_nan} misshm={n_missinghm} mismatch={n_mismatch})")
    return motions, hms, stats


class HMWindowDataset(Dataset):
    """Windowed (motion, heightmap) pairs. If heightmaps is None (H3D), yields a flat-floor
    (zeros) heightmap. Motion is Z-normalized; heightmap is passed through in metres."""

    def __init__(self, motions, heightmaps, mean, std, window_size=64, grid_n=GRID_N):
        self.motions = motions
        self.heightmaps = heightmaps  # list aligned to motions, or None (flat floor)
        self.mean = mean
        self.std = std
        self.window_size = window_size
        self.grid_n = grid_n

    def __len__(self):
        return len(self.motions)

    def __getitem__(self, idx):
        motion = self.motions[idx]
        start = random.randint(0, len(motion) - self.window_size)
        w = motion[start:start + self.window_size]
        w = ((w - self.mean) / self.std).astype(np.float32)
        if self.heightmaps is None:
            hm = np.zeros((self.window_size, self.grid_n, self.grid_n), dtype=np.float32)
        else:
            hm = self.heightmaps[idx][start:start + self.window_size].astype(np.float32)
        return w, hm


def _cycle(loader):
    while True:
        for x in loader:
            yield x


class BalancedJointHMLoader:
    """Same balanced H3D:HUMANISE mix as BalancedJointLoader, but yields (motion, heightmap)
    batches. H3D sub-batch gets flat-floor heightmaps; HUMANISE gets real ones."""

    def __init__(self, h3d_ds, hum_ds, batch_size, h3d_frac=0.5, num_workers=4, seed=0):
        n_h3d = max(1, round(batch_size * h3d_frac))
        n_hum = max(1, batch_size - n_h3d)
        self.n_h3d, self.n_hum, self.batch_size = n_h3d, n_hum, n_h3d + n_hum
        g1 = torch.Generator().manual_seed(seed)
        g2 = torch.Generator().manual_seed(seed + 1)
        self.h3d_loader = DataLoader(h3d_ds, batch_size=n_h3d, shuffle=True,
                                     num_workers=num_workers, drop_last=True, generator=g1,
                                     persistent_workers=num_workers > 0)
        self.hum_loader = DataLoader(hum_ds, batch_size=n_hum, shuffle=True,
                                     num_workers=num_workers, drop_last=True, generator=g2,
                                     persistent_workers=num_workers > 0)

    def __iter__(self):
        h3d_it = _cycle(self.h3d_loader)
        hum_it = _cycle(self.hum_loader)
        while True:
            ma, ha = next(h3d_it)
            mb, hb = next(hum_it)
            yield torch.cat([ma, mb], dim=0), torch.cat([ha, hb], dim=0)


class BalancedThreeSourceHMLoader:
    """Three-source balanced loader: H3D (flat floor) + HUMANISE (real hm) + TRUMANS (real hm).
    Each batch is split h3d_frac : hum_frac : trumans_frac. TRUMANS adds seat-height variety
    that HUMANISE lacks (σ=0.108, range 0.38–0.85 m vs near-zero)."""

    def __init__(self, h3d_ds, hum_ds, trumans_ds, batch_size,
                 h3d_frac=0.34, hum_frac=0.33, num_workers=4, seed=0):
        n_h3d = max(1, round(batch_size * h3d_frac))
        n_hum = max(1, round(batch_size * hum_frac))
        n_tru = max(1, batch_size - n_h3d - n_hum)
        self.batch_size = n_h3d + n_hum + n_tru
        g1 = torch.Generator().manual_seed(seed)
        g2 = torch.Generator().manual_seed(seed + 1)
        g3 = torch.Generator().manual_seed(seed + 2)
        self.h3d_loader = DataLoader(h3d_ds, batch_size=n_h3d, shuffle=True,
                                     num_workers=num_workers, drop_last=True, generator=g1,
                                     persistent_workers=num_workers > 0)
        self.hum_loader = DataLoader(hum_ds, batch_size=n_hum, shuffle=True,
                                     num_workers=num_workers, drop_last=True, generator=g2,
                                     persistent_workers=num_workers > 0)
        self.tru_loader = DataLoader(trumans_ds, batch_size=n_tru, shuffle=True,
                                     num_workers=num_workers, drop_last=True, generator=g3,
                                     persistent_workers=num_workers > 0)

    def __iter__(self):
        h3d_it = _cycle(self.h3d_loader)
        hum_it = _cycle(self.hum_loader)
        tru_it = _cycle(self.tru_loader)
        while True:
            ma, ha = next(h3d_it)
            mb, hb = next(hum_it)
            mc, hc = next(tru_it)
            yield (torch.cat([ma, mb, mc], dim=0),
                   torch.cat([ha, hb, hc], dim=0))
