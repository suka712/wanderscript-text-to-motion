#!/usr/bin/env python3
"""Check 1 & 2: inspect HUMANISE motion files and H3D new_joint_vecs for shape/format."""
import numpy as np
import os

ROOT = "/media/user/2tb/motion_data"

def inspect_npy(path):
    arr = np.load(path, allow_pickle=True)
    print(f"  path={path}")
    print(f"  type={type(arr)} shape={getattr(arr,'shape',None)} dtype={getattr(arr,'dtype',None)}")
    if arr.dtype == object:
        try:
            item = arr.item()
            print(f"  object item type={type(item)}")
            if isinstance(item, dict):
                for k, v in item.items():
                    if hasattr(v, 'shape'):
                        print(f"    key={k} shape={v.shape} dtype={v.dtype}")
                    else:
                        print(f"    key={k} type={type(v)} value={str(v)[:200]}")
        except Exception as e:
            print("  could not .item():", e)
    return arr

def inspect_npz(path):
    d = np.load(path, allow_pickle=True)
    print(f"  path={path}")
    print(f"  keys={list(d.keys())}")
    for k in d.keys():
        v = d[k]
        print(f"    key={k} shape={v.shape} dtype={v.dtype}")
        if v.size < 20:
            print(f"      values={v}")
    return d

print("=" * 60)
print("HUMANISE contact_motion/motions/00000.npy")
inspect_npy(f"{ROOT}/HUMANISE/contact_motion/motions/00000.npy")

print("=" * 60)
print("HUMANISE contact_motion/motions/00001.npy")
inspect_npy(f"{ROOT}/HUMANISE/contact_motion/motions/00001.npy")

print("=" * 60)
print("HUMANISE contact_motion/contacts/00000.npz")
inspect_npz(f"{ROOT}/HUMANISE/contact_motion/contacts/00000.npz")

print("=" * 60)
print("HUMANISE contact_motion/target_mask/00000.npy")
inspect_npy(f"{ROOT}/HUMANISE/contact_motion/target_mask/00000.npy")

print("=" * 60)
print("HUMANISE pred_contact/00036.npy (top-level)")
inspect_npy(f"{ROOT}/HUMANISE/pred_contact/00036.npy")

print("=" * 60)
print("H3D new_joint_vecs/000000.npy")
inspect_npy(f"{ROOT}/H3D/new_joint_vecs/000000.npy")

print("=" * 60)
print("H3D Mean.npy / Std.npy")
mean = np.load(f"{ROOT}/H3D/Mean.npy")
std = np.load(f"{ROOT}/H3D/Std.npy")
print("Mean shape", mean.shape, "Std shape", std.shape)

print("=" * 60)
print("Mean_Std_CM_HUMANISE_pos.npz")
inspect_npz(f"{ROOT}/Mean_Std_CM_HUMANISE_pos.npz")

print("=" * 60)
print("Mean_Std_Cont_HUMANISE_contact_cont_joints_0.8.npz")
inspect_npz(f"{ROOT}/Mean_Std_Cont_HUMANISE_contact_cont_joints_0.8.npz")
