#!/usr/bin/env python3
"""Cache one BEV render (RGB + occupancy + extent) per ScanNet scene used by
HUMANISE, for the scene-representation probe (docs/06_scene_probe.md).

Rendering is the expensive part and is scene-level, while the probe is
clip-level with many clips per scene -- so it is done once, here, and cached.
"""
import argparse
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import bev_render as bv  # noqa: E402
from humanise_join import build_flat_join  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None, help="cap number of scenes (for a smoke test)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    scenes = sorted({p["scene"] for p in build_flat_join()})
    if args.limit:
        scenes = scenes[:args.limit]
    print(f"{len(scenes)} distinct scenes")

    t0, done, failed = time.time(), 0, []
    for i, sid in enumerate(scenes):
        out = os.path.join(args.out_dir, f"{sid}.npz")
        if os.path.exists(out):
            done += 1
            continue
        try:
            rgb, occ, ext = bv.render_scene_bev(sid, resolution_px=args.resolution)
            np.savez_compressed(
                out, rgb=rgb.astype(np.uint8), occ=occ.astype(np.bool_),
                extent=np.array([ext.xmin, ext.xmax, ext.ymin, ext.ymax], dtype=np.float64),
                shape=np.array([ext.height_px, ext.width_px], dtype=np.int32))
            done += 1
        except Exception as e:  # noqa: BLE001 - want the id, keep going
            failed.append((sid, repr(e)[:120]))
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(scenes)}  ok={done} failed={len(failed)}  "
                  f"{el/(i+1):.2f}s/scene  eta {el/(i+1)*(len(scenes)-i-1)/60:.1f}min", flush=True)

    print(f"\ndone: {done} cached, {len(failed)} failed")
    for sid, err in failed[:10]:
        print(f"  FAIL {sid}: {err}")


if __name__ == "__main__":
    main()
