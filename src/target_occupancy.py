"""Target-excluded occupancy: the collision map a scene-aware motion model
should actually be scored against.

WHY. docs/08 showed the naive collision metric is meaningless on HUMANISE:
ground-truth `lie` motion sits in occupied space 100% of the time, because the
person is on the bed. On an interaction dataset, being inside the target
furniture IS the objective. Scoring it as collision measures the goal, not a
failure.

WHAT WOULD BE IDEAL. Exclude the goal object's ScanNet instance. HUMANISE gives
`object_id` per clip and ScanNet publishes per-instance segmentation
(`*.aggregation.json` + `*.segs.json`) -- but those files are NOT in the local
mirror, which holds only `*_vh_clean_2.ply`. Fetching them needs ScanNet
download credentials. If they are ever obtained, `exclude_instance` below is the
better path and this module should be replaced.

WHAT THIS DOES INSTEAD. Take the occupied connected component that the goal lies
in (or is nearest to) and treat it as the target. If the goal is on a bed, the
contiguous occupied blob containing it is the bed. This needs nothing beyond the
occupancy raster we already render.

KNOWN LIMITATION, stated because it affects how the number should be read:
furniture pushed against a wall merges into one component with the wall, so the
target blob can swallow the wall and under-report collisions. `max_area_frac`
guards the worst case by refusing to exclude a component larger than that
fraction of occupied space -- a wall-sized blob is not a chair. Clips where the
guard fires are counted and reported, so the rate is never quoted without
knowing how often the proxy gave up.
"""
import numpy as np

try:
    from scipy import ndimage
except ImportError:  # pragma: no cover
    ndimage = None

# 8-connectivity: furniture footprints are blobby, and 4-connectivity splits
# diagonal chair legs into separate components.
_STRUCT = np.ones((3, 3), dtype=int)


def world_to_rc(xy, extent, shape):
    """Same mapping as bev_render.BEVExtent.world_to_pixel, kept in sync here so
    callers need only the cached raster + extent, not a BEVExtent object."""
    xmin, xmax, ymin, ymax = extent
    H, W = shape
    xy = np.atleast_2d(np.asarray(xy, dtype=float))
    col = np.clip(((xy[:, 0] - xmin) / (xmax - xmin) * W).astype(int), 0, W - 1)
    row = np.clip(((ymax - xy[:, 1]) / (ymax - ymin) * H).astype(int), 0, H - 1)
    return row, col


def target_component(occ, goal_xy, extent, search_px=40, max_area_frac=0.25):
    """Mask of the occupied component treated as the goal object.

    Returns (mask, status) where status is one of:
      'hit'      the goal pixel is itself occupied -- the component under it
      'nearest'  the goal is on free space; used the nearest occupied component
                 within search_px (a goal beside a chair, not on it)
      'none'     no occupied pixel within search_px -- nothing excluded
      'too_big'  the component exceeded max_area_frac and was NOT excluded,
                 because it is almost certainly furniture merged with a wall
    """
    if ndimage is None:
        raise ImportError("scipy is required for target_component")
    occ_b = occ > 0.5
    lab, n = ndimage.label(occ_b, structure=_STRUCT)
    if n == 0:
        return np.zeros_like(occ_b), "none"

    r, c = world_to_rc(goal_xy, extent, occ.shape)
    r, c = int(r[0]), int(c[0])

    comp = lab[r, c]
    status = "hit"
    if comp == 0:
        # goal on free space: nearest occupied pixel within a window
        r0, r1 = max(0, r - search_px), min(occ.shape[0], r + search_px + 1)
        c0, c1 = max(0, c - search_px), min(occ.shape[1], c + search_px + 1)
        win = occ_b[r0:r1, c0:c1]
        if not win.any():
            return np.zeros_like(occ_b), "none"
        rr, cc = np.nonzero(win)
        d = (rr + r0 - r) ** 2 + (cc + c0 - c) ** 2
        k = int(np.argmin(d))
        comp = lab[rr[k] + r0, cc[k] + c0]
        status = "nearest"

    mask = lab == comp
    if mask.sum() > max_area_frac * occ_b.sum():
        return np.zeros_like(occ_b), "too_big"
    return mask, status


def collision_map(occ, goal_xy, extent, **kw):
    """Occupancy with the target object removed. Returns (map, status)."""
    mask, status = target_component(occ, goal_xy, extent, **kw)
    out = (occ > 0.5).astype(np.float32)
    out[mask] = 0.0
    return out, status


def collision_rate(world_xy, cmap, extent):
    """Fraction of frames whose root lands on occupied, non-target space."""
    row, col = world_to_rc(world_xy, extent, cmap.shape)
    hits = cmap[row, col] > 0.5
    return float(hits.mean()), bool(hits.any())
