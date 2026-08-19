#!/usr/bin/env python3
"""Does the model actually SIT / STAND, or just navigate to the xy? (Step 11 diagnostic)

Step 11a reported sit/stand-up "work" via goal error -- but the goal is only (x, y),
so walking to the seated pelvis's xy scores well WITHOUT ever lowering the body.
The composed demo (demo_interaction.py) exposed this: 0/3 chains sat, pelvis stayed
~0.92 m through the sit segment. Goal error is structurally blind to sitting, exactly
the CLAUDE.md risk #4 failure mode.

This measures the thing goal error cannot: PELVIS HEIGHT. A real sit drops the pelvis
0.95 -> ~0.52 m; a real stand-up raises it back. We generate ONE segment per clip with
the exact training conditioning (real start pose, real frame-0 prefix, real seated goal,
the clip's own utterance) and ask whether the generated pelvis actually moves in Z.

Restricted to clips whose GROUND TRUTH clearly interacts (sit end-z < SIT_Z, stand-up
end-z > STAND_Z), so the question is the fair one: when the data sits, does the model?

Compare >=2 checkpoints on the SAME clips to localize cause -- in particular step8/full
(no goal augmentation) vs step10/goalaug, since goal augmentation trains pure xy-reaching
and is the prime suspect for teaching the model to navigate instead of interact.
"""
import argparse
import os
import sys

import clip
import numpy as np
import torch

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
for p in ["src", "scripts/track1", "scripts/chaining"]:
    sys.path.insert(0, os.path.join(REPO_ROOT, p))

import motion_features as mf  # noqa: E402
from humanise_join import build_flat_join, get_record, compute_track2, J_PELVIS  # noqa: E402
from se2_utils import yup_to_zup  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402
from rollout import rollout, load_model  # noqa: E402

T2M = os.environ.get("WANDER_T2M_GPT_ROOT")
HUMANISE = os.environ.get("WANDER_HUMANISE_ROOT")
BEV = os.path.expanduser("~/wander_data/bev_cache")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SIT_Z = 0.70    # pelvis below this = seated
STAND_Z = 0.80  # pelvis above this = standing


def gt_pelvis_z(cm):
    d263, *_ = mf.humanise_positions_to_263(cm)
    j = yup_to_zup(mf.recover_positions(d263.astype(np.float32)))
    return j[:, J_PELVIS, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="one or more model dirs")
    ap.add_argument("--vqvae-ckpt", required=True)
    ap.add_argument("--n", type=int, default=60, help="clips per action (that GT-interact)")
    ap.add_argument("--actions", nargs="+", default=["sit", "stand up"])
    ap.add_argument("--prefix-mode", choices=["own", "walk"], default="own",
                    help="'own' = the clip's real frame-0 pose (standing-still, in distribution). "
                         "'walk' = a mid-stride walking pose, to test the hypothesis that a "
                         "WALKING prefix (what a composed walk->sit seam hands the sit segment) "
                         "suppresses the sit. Everything else identical, so any drop isolates the "
                         "prefix. HUMANISE sit clips start from standstill, so 'walk' is OOD.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    net = load_vqvae(ckpt_path=args.vqvae_ckpt, device=DEV); net.eval()
    mean = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy").astype(np.float32)
    std = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy").astype(np.float32)
    cmodel, _ = clip.load("ViT-B/32", device=DEV, jit=False); cmodel.eval()
    models = {os.path.basename(c.rstrip("/")): load_model(c) for c in args.ckpts}
    for name, (_, ns) in models.items():
        print(f"model {name}: cond_mode={ns['cond_mode']} goal_aug={ns.get('goal_aug')}")

    flat = build_flat_join()
    rng = np.random.RandomState(args.seed)
    scenes = {}

    # A fixed mid-stride WALKING pose, for --prefix-mode walk. Taken from the middle
    # frame of a real walk clip (on-manifold), so the only thing that differs from the
    # 'own' run is that the sit segment is handed a walking body instead of a standstill.
    walk_prefix = None
    if args.prefix_mode == "walk":
        for i, p in enumerate(flat):
            if p["action"] != "walk":
                continue
            cmw = np.load(os.path.join(HUMANISE, "contact_motion", "motions", f"{i:05d}.npy"))
            if cmw.shape[0] < 20:
                continue
            dw, *_ = mf.humanise_positions_to_263(cmw)
            lw = mf.local_joint_positions(dw.astype(np.float32))
            walk_prefix = lw[len(lw) // 2].ravel().astype(np.float32)  # mid-stride
            print(f"walk prefix from clip {i} (mid frame)")
            break

    for action in args.actions:
        want_sit = action in ("sit", "lie")
        idxs = [i for i, p in enumerate(flat) if p["action"] == action]
        rng.shuffle(idxs)
        # per-model accumulators
        gen_end = {m: [] for m in models}
        gen_goal = {m: [] for m in models}
        gt_end = []
        used = 0
        for idx in idxs:
            if used >= args.n:
                break
            rec = get_record(int(idx))
            f = os.path.join(BEV, f"{rec.scene}.npz")
            if not os.path.exists(f):
                continue
            cm = np.load(os.path.join(HUMANISE, "contact_motion", "motions", f"{idx:05d}.npy"))
            if cm.shape[0] < 8:
                continue
            z = gt_pelvis_z(cm)
            interacts = (z[-1] < SIT_Z) if want_sit else (z[-1] > STAND_Z and z[0] < SIT_Z)
            if not interacts:
                continue
            try:
                d0, *_ = mf.humanise_positions_to_263(cm)
            except Exception:
                continue
            if d0.shape[0] < 8:
                continue
            _, xy, _, sincos = compute_track2(rec)
            start_pose = np.array([xy[0, 0], xy[0, 1], sincos[0, 0], sincos[0, 1]], np.float32)
            goal = xy[-1].astype(np.float32)
            prefix = (walk_prefix if walk_prefix is not None
                      else mf.local_joint_positions(d0.astype(np.float32))[0].ravel())
            if rec.scene not in scenes:
                if len(scenes) > 40:
                    scenes.clear()
                zz = np.load(f)
                scenes[rec.scene] = (zz["occ"].astype(np.float32), zz["extent"])
            occ, extent = scenes[rec.scene]

            gt_end.append(float(z[-1]))
            for m, (trans, ns) in models.items():
                acts = [action] if ns["cond_mode"] in ("full_action", "full_action_head") else None
                segs = rollout(trans, net, cmodel, clip, mean, std, ns, [rec.utterance],
                               [goal], start_pose, prefix, occ, extent, actions=acts)
                if not segs:
                    gen_end[m].append(np.nan); gen_goal[m].append(np.nan); continue
                gz = segs[0]["world"][:, J_PELVIS, 2]
                gen_end[m].append(float(gz[-1]))
                gen_goal[m].append(float(segs[0]["goal_err"]))
            used += 1

        thr, verb = (SIT_Z, "sat (end-z<%.2f)" % SIT_Z) if want_sit else (STAND_Z, "stood (end-z>%.2f)" % STAND_Z)
        print(f"\n=== {action}: {used} clips whose GROUND TRUTH interacts "
              f"(GT mean end-z {np.mean(gt_end):.2f}) ===")
        print(f"{'model':<12}{'gen end-z':>11}{'  '+verb:>20}{'goal-err':>10}")
        for m in models:
            e = np.array(gen_end[m]); g = np.array(gen_goal[m])
            ok = (e < thr).mean() if want_sit else (e > thr).mean()
            print(f"{m:<12}{np.nanmean(e):>11.2f}{ok*100:>18.0f}%{np.nanmean(g):>10.2f}")
        print(f"  (GT interacts 100% by construction; a model that matches the data should "
              f"approach 100% and a low gen end-z for sit / high for stand.)")


if __name__ == "__main__":
    main()
