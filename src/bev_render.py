"""
BEV (bird's-eye-view) renderer for ScanNet scenes: two pixel-aligned rasters
per scene -- an RGB photographic top-down render (for DINOv2 scene
conditioning) and a binary walkable/occupied occupancy raster (for
collision-guided decoding + non-collision metrics). See docs/STEP1_plumbing.md
Check 4 for the original finding this completes (meshes present, no renderer
built yet).

Both rasters share the same square world-frame extent, centered on the
scene's XY bounds, and the same pixel resolution -- so they are exactly
pixel-aligned to each other and to a known world->pixel affine mapping. That
mapping (see `world_to_pixel` / `BEVExtent`) is what lets rollout coordinates
(x, y) map directly to raster pixels without a second calibration step.

Two things this module deliberately does NOT do:
  - It does not use pyrender's depth buffer to derive occupancy. Orthographic
    depth in this pyrender version has poor / unusable metric precision (a
    scene ~8m from the camera returned depth values ~0.011 -- not invertible
    to world height with any confidence). Occupancy is instead computed by
    directly rasterizing the mesh's own triangles (height-sliced) with
    matplotlib, which is exact and avoids the GL depth-buffer entirely.
  - It does not attempt semantic segmentation. The RGB raster is a plain
    top-down photographic render (vertex-colored ScanNet mesh); DINOv2
    consumes it as-is.

Rendering the RGB raster requires an EGL-capable offscreen GL context
(`os.environ['PYOPENGL_PLATFORM'] = 'egl'`, set below at import time -- must
happen before pyrender is imported anywhere in the process). Verified working
headless via the NVIDIA driver's libEGL on this machine.
"""
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from dataclasses import dataclass

import numpy as np
import trimesh

SCANNET_ROOT = os.environ.get(
    "WANDER_SCANNET_ROOT", "/media/user/2tb/motion_data/scannet/scans"
)

# Height slicing (meters above floor). Matches the ~2m "clear the ceiling so
# the camera can see furniture" convention used in the RGB render, and a
# smaller obstacle threshold for occupancy (anything taller than this blocks
# a ground robot; the floor/rug/low threshold itself does not).
CEILING_CUTOFF_M = 2.0
OBSTACLE_HEIGHT_M = 0.12


@dataclass
class BEVExtent:
    """World-frame square extent + pixel resolution shared by both rasters.

    world->pixel mapping (verified empirically against a marker of known
    world position, see scripts/verify/check11_bev_render.py):
        col = (x - xmin) / (xmax - xmin) * width_px
        row = (ymax - y) / (ymax - ymin) * height_px
    (row 0 = ymax, i.e. image top = +Y, matching pyrender's camera-up = +Y
    for the identity-rotation top-down camera pose used here.)
    """

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    width_px: int
    height_px: int

    def world_to_pixel(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        col = (x - self.xmin) / (self.xmax - self.xmin) * self.width_px
        row = (self.ymax - y) / (self.ymax - self.ymin) * self.height_px
        return row, col


def _load_scene_mesh(scene_id: str) -> trimesh.Trimesh:
    path = f"{SCANNET_ROOT}/{scene_id}/{scene_id}_vh_clean_2.ply"
    return trimesh.load(path, process=False)


def _make_extent(mesh: trimesh.Trimesh, resolution_px: int) -> BEVExtent:
    bounds = mesh.bounds
    cx = (bounds[0, 0] + bounds[1, 0]) / 2
    cy = (bounds[0, 1] + bounds[1, 1]) / 2
    extent = max(bounds[1, 0] - bounds[0, 0], bounds[1, 1] - bounds[0, 1]) * 1.05
    half = extent / 2
    return BEVExtent(cx - half, cx + half, cy - half, cy + half, resolution_px, resolution_px)


def render_rgb(mesh: trimesh.Trimesh, ext: BEVExtent) -> np.ndarray:
    """Top-down photographic render, ceiling clipped so furniture is visible."""
    import pyrender

    bounds = mesh.bounds
    floor_z = bounds[0, 2]
    cutoff = floor_z + CEILING_CUTOFF_M
    verts, faces = mesh.vertices, mesh.faces
    face_z = verts[faces][:, :, 2]
    keep = (face_z < cutoff).all(axis=1)
    cropped = mesh.copy()
    cropped.faces = faces[keep]
    cropped.remove_unreferenced_vertices()

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.6, 0.6, 0.6])
    scene.add(pyrender.Mesh.from_trimesh(cropped, smooth=False))

    cx, cy = (ext.xmin + ext.xmax) / 2, (ext.ymin + ext.ymax) / 2
    half = (ext.xmax - ext.xmin) / 2
    cam_z = bounds[1, 2] + 5
    cam = pyrender.OrthographicCamera(xmag=half, ymag=half, znear=0.01, zfar=50)
    cam_pose = np.array(
        [[1, 0, 0, cx], [0, 1, 0, cy], [0, 0, 1, cam_z], [0, 0, 0, 1]], dtype=float
    )
    scene.add(cam, pose=cam_pose)
    scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=4.0), pose=cam_pose)

    r = pyrender.OffscreenRenderer(ext.width_px, ext.height_px)
    try:
        color, _ = r.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
    finally:
        r.delete()
    return color


def render_occupancy(mesh: trimesh.Trimesh, ext: BEVExtent) -> np.ndarray:
    """Binary occupancy raster: True = occupied/blocked, False = walkable.

    Computed by rasterizing mesh triangles directly (not the GL depth
    buffer): floor-level triangles (height <= OBSTACLE_HEIGHT_M above the
    scene's minimum Z) mark walkable floor extent; triangles with any vertex
    between OBSTACLE_HEIGHT_M and CEILING_CUTOFF_M mark occupied obstacles,
    drawn on top. Anything not covered by either (outside the room's
    footprint, or a genuine hole in the scan) stays occupied by default --
    the safe assumption for a robot that doesn't know what's there.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    bounds = mesh.bounds
    floor_z = bounds[0, 2]
    verts, faces = mesh.vertices, mesh.faces
    tri_z = verts[faces][:, :, 2] - floor_z  # (F, 3) height above floor per vertex
    tri_xy = verts[faces][:, :, :2]  # (F, 2, ... ) -> actually (F,3,2)

    floor_faces = tri_z.max(axis=1) <= OBSTACLE_HEIGHT_M
    obstacle_faces = (tri_z.min(axis=1) <= CEILING_CUTOFF_M) & (
        tri_z.max(axis=1) > OBSTACLE_HEIGHT_M
    )

    dpi = 100
    fig = plt.figure(
        figsize=(ext.width_px / dpi, ext.height_px / dpi), dpi=dpi
    )
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(ext.xmin, ext.xmax)
    ax.set_ylim(ext.ymin, ext.ymax)
    ax.set_facecolor("black")  # default = occupied/unknown
    ax.axis("off")

    if floor_faces.any():
        ax.add_collection(
            PolyCollection(tri_xy[floor_faces], facecolors="white", edgecolors="none")
        )
    if obstacle_faces.any():
        ax.add_collection(
            PolyCollection(tri_xy[obstacle_faces], facecolors="black", edgecolors="none")
        )

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)

    walkable = buf[:, :, 0] > 128  # white pixels
    return ~walkable  # True = occupied


def render_scene_bev(scene_id: str, resolution_px: int = 512):
    """Returns (rgb, occupancy, extent) for one ScanNet scene, pixel-aligned."""
    mesh = _load_scene_mesh(scene_id)
    ext = _make_extent(mesh, resolution_px)
    rgb = render_rgb(mesh, ext)
    occupancy = render_occupancy(mesh, ext)
    return rgb, occupancy, ext
