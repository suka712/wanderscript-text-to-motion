#!/usr/bin/env python3
"""
Check 4 (BEV occupancy raster) completion. Step 1 confirmed all 643 HUMANISE
ScanNet meshes load fine but no renderer existed. This builds and validates
one: for a handful of scenes, render the RGB + occupancy pair via
src/bev_render.py, and independently verify the world->pixel mapping by
placing a marker of known world position and checking it lands where the
formula predicts.

Outputs saved under scratch_outputs/bev/ (gitignored).
"""
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from PIL import Image

from src.bev_render import _load_scene_mesh, _make_extent, render_occupancy, render_rgb

SCENES = ["scene0000_00", "scene0005_00", "scene0006_00", "scene0050_00", "scene0100_00"]
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "scratch_outputs", "bev")


def verify_mapping(scene_id: str, resolution_px: int = 512) -> float:
    """Places a bright marker at a known world (x, y), renders it, and
    returns the pixel error (in px) between the formula's prediction and
    where the marker actually shows up in the render."""
    import pyrender
    import trimesh

    mesh = _load_scene_mesh(scene_id)
    ext = _make_extent(mesh, resolution_px)
    cx, cy = (ext.xmin + ext.xmax) / 2, (ext.ymin + ext.ymax) / 2
    half = (ext.xmax - ext.xmin) / 2
    marker_world = np.array([cx + half * 0.3, cy - half * 0.2, mesh.bounds[1, 2] + 0.1])

    marker = trimesh.creation.icosphere(radius=half * 0.03)
    marker.visual.vertex_colors = [255, 0, 255, 255]
    marker.apply_translation(marker_world)

    scene = pyrender.Scene(bg_color=[1, 1, 1, 1], ambient_light=[0.8, 0.8, 0.8])
    scene.add(pyrender.Mesh.from_trimesh(marker, smooth=False))
    cam = pyrender.OrthographicCamera(xmag=half, ymag=half, znear=0.01, zfar=50)
    cam_pose = np.array(
        [[1, 0, 0, cx], [0, 1, 0, cy], [0, 0, 1, mesh.bounds[1, 2] + 5], [0, 0, 0, 1]],
        dtype=float,
    )
    scene.add(cam, pose=cam_pose)
    r = pyrender.OffscreenRenderer(resolution_px, resolution_px)
    try:
        color, _ = r.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
    finally:
        r.delete()

    mask = (color[:, :, 0] > 200) & (color[:, :, 2] > 200) & (color[:, :, 1] < 100)
    ys, xs = np.where(mask)
    actual_row, actual_col = ys.mean(), xs.mean()
    pred_row, pred_col = ext.world_to_pixel(marker_world[0], marker_world[1])
    err = float(np.hypot(actual_row - pred_row, actual_col - pred_col))
    return err


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"{'scene':<16} {'mapping_err_px':<16} {'occupied_%':<12} {'walkable_%'}")
    for scene_id in SCENES:
        mesh = _load_scene_mesh(scene_id)
        ext = _make_extent(mesh, 512)
        rgb = render_rgb(mesh, ext)
        occ = render_occupancy(mesh, ext)
        err = verify_mapping(scene_id)

        occ_pct = 100 * occ.mean()
        walk_pct = 100 * (~occ).mean()
        print(f"{scene_id:<16} {err:<16.3f} {occ_pct:<12.1f} {walk_pct:.1f}")

        Image.fromarray(rgb).save(os.path.join(OUT_DIR, f"{scene_id}_rgb.png"))
        occ_vis = np.where(occ[:, :, None], np.array([0, 0, 0]), np.array([255, 255, 255])).astype(
            np.uint8
        )
        Image.fromarray(occ_vis).save(os.path.join(OUT_DIR, f"{scene_id}_occupancy.png"))

        # composite: RGB with occupied pixels tinted red, for eyeball alignment check
        composite = rgb.copy()
        composite[occ] = (0.5 * composite[occ] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
        Image.fromarray(composite).save(os.path.join(OUT_DIR, f"{scene_id}_composite.png"))


if __name__ == "__main__":
    main()
