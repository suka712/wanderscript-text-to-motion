#!/usr/bin/env python3
"""
Check 4: confirm ScanNet meshes for HUMANISE scenes are present and loadable.
Does NOT attempt BEV rendering -- only existence + trimesh load + bounds sanity.
"""
import os
import trimesh

HUMANISE_SCENES = "/media/user/2tb/motion_data/HUMANISE/scenes"
SCANNET_SCANS = "/media/user/2tb/motion_data/scannet/scans"

humanise_ids = sorted(os.listdir(HUMANISE_SCENES))
scannet_ids = sorted(os.listdir(SCANNET_SCANS))

missing = sorted(set(humanise_ids) - set(scannet_ids))
print(f"HUMANISE scene ids: {len(humanise_ids)}")
print(f"ScanNet scan dirs:  {len(scannet_ids)}")
print(f"HUMANISE scene ids missing a ScanNet mesh dir: {len(missing)}")
if missing:
    print("  e.g.", missing[:10])

# spot check a handful of meshes actually load
sample = humanise_ids[:5]
for sid in sample:
    mesh_path = f"{SCANNET_SCANS}/{sid}/{sid}_vh_clean_2.ply"
    exists = os.path.isfile(mesh_path)
    size = os.path.getsize(mesh_path) if exists else 0
    print(f"{sid}: exists={exists} size={size}")
    if exists:
        m = trimesh.load(mesh_path)
        print(f"  vertices={m.vertices.shape} bounds=\n{m.bounds}")
