#!/usr/bin/env python3
"""Does chaining accumulate error? (build-order step 9, the open research question)

RESULTS §7 validated conditional continuation at ONE seam, with a ground-truth
previous segment. Chaining runs the model on its OWN output N times, so the
question this answers is whether that compounds.

METHOD. Take a real HUMANISE clip, cut it into N consecutive segments, and feed
the model each segment's TRUE endpoint as its goal. Then chain: segment 1 from
the real start, every later segment from the previous segment's generated
ending pose. Because the goals are ground truth, the chained trajectory can be
compared directly against the real one.

N=1 is the single-segment case and reproduces the earlier setting; rising error
with N is accumulation. Reported per N:
  final drift   distance between the chain's last root position and the clip's
  seam err      mean over seams, i.e. does continuity survive repetition
  goal err      mean over segments, per-segment goal reaching

CONTROLS:
  ORACLE  the same clip's own tokens decoded and chained through the identical
          placement path. Non-zero (VQ-VAE reconstruction), and it is the floor
          drift is measured against -- some drift is the tokenizer, not chaining.
  NULL    never move. Its "drift" is the clip's total displacement, i.e. what
          you get for doing nothing.
"""
import argparse
import os
import pickle
import sys

import clip
import numpy as np
import torch

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
for p in ["src", "scripts/track1", "scripts/chaining"]:
    sys.path.insert(0, os.path.join(REPO_ROOT, p))

import motion_features as mf  # noqa: E402
from humanise_join import get_record, compute_track2  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402
from se2_utils import se2_place_full_body  # noqa: E402
from rollout import rollout, load_model, yaw_from_joints  # noqa: E402

T2M = os.environ.get("WANDER_T2M_GPT_ROOT")
HUMANISE = os.environ.get("WANDER_HUMANISE_ROOT")
BEV = os.path.expanduser("~/wander_data/bev_cache")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MIN_SEG = 16


def cut(cm, n):
    """n consecutive raw-frame ranges, each >= MIN_SEG frames, token-aligned."""
    T = cm.shape[0]
    if T < n * (MIN_SEG + 2):
        return None
    bnds = [(i * T // n // 4) * 4 for i in range(n)] + [T]
    for a, b in zip(bnds, bnds[1:]):
        if b - a < MIN_SEG + 2:
            return None
    return list(zip(bnds, bnds[1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vqvae-ckpt", required=True)
    ap.add_argument("--tokens-dir", required=True)
    ap.add_argument("--n-clips", type=int, default=150)
    ap.add_argument("--chain-lengths", type=int, nargs="+", default=[1, 2, 3, 4, 6])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--by-action", action="store_true",
                    help="additionally print a per-action breakdown (walk/sit/stand up/lie). "
                         "Purely additive -- the default aggregate table is unchanged. Step 11: "
                         "the aggregate hides interaction seams inside a walk-dominated average; "
                         "this surfaces them, and the N=1 stand-up row tests the hardest case "
                         "(a fully-seated start prefix -> a full stand-up, which the midpoint-cut "
                         "continuation training never saw as a generation target).")
    args = ap.parse_args()

    test = pickle.load(open(os.path.join(args.tokens_dir, "test.pkl"), "rb"))
    net = load_vqvae(ckpt_path=args.vqvae_ckpt, device=DEV); net.eval()
    mean = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy").astype(np.float32)
    std = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy").astype(np.float32)
    cmodel, _ = clip.load("ViT-B/32", device=DEV, jit=False); cmodel.eval()
    trans, ns = load_model(args.ckpt)
    print(f"model cond_mode={ns['cond_mode']}")

    rng = np.random.RandomState(args.seed)
    picks = rng.choice(len(test), size=min(args.n_clips * 3, len(test)), replace=False)

    scenes = {}
    out = {n: {"drift": [], "seam": [], "goal": [], "oracle": [], "null": []} for n in args.chain_lengths}
    by_action = {n: {} for n in args.chain_lengths} if args.by_action else None
    used = 0
    for i in picks:
        if used >= args.n_clips:
            break
        d = test[int(i)]
        rec = get_record(int(d["index"]))
        cm = np.load(os.path.join(HUMANISE, "contact_motion", "motions", f"{d['index']:05d}.npy"))
        jw, xy, _, sincos = compute_track2(rec)
        occ = extent = None
        if ns["cond_mode"] in ("full", "full_action", "full_action_head"):
            f = os.path.join(BEV, f"{rec.scene}.npz")
            if not os.path.exists(f):
                continue
            if rec.scene not in scenes:
                if len(scenes) > 30:
                    scenes.clear()
                z = np.load(f)
                scenes[rec.scene] = (z["occ"].astype(np.float32), z["extent"])
            occ, extent = scenes[rec.scene]

        ok = False
        for n in args.chain_lengths:
            cuts = cut(cm, n)
            if cuts is None:
                continue
            # per-segment goals = true world position at each segment end
            goals, texts = [], []
            for a, b in cuts:
                gi = min(b - 1, xy.shape[0] - 1)
                goals.append(xy[gi])
                texts.append(rec.utterance)
            # first segment starts at the clip's real start pose and its real first pose
            seg0 = cm[cuts[0][0]:cuts[0][1] + 1]
            try:
                d0, *_ = mf.humanise_positions_to_263(seg0)
            except Exception:
                continue
            if d0.shape[0] < 4:
                continue
            start_pose = np.array([xy[0, 0], xy[0, 1], sincos[0, 0], sincos[0, 1]], np.float32)
            prefix = mf.local_joint_positions(d0.astype(np.float32))[0].ravel()

            acts = [rec.action] * len(goals) if ns["cond_mode"] in ("full_action", "full_action_head") else None
            segs = rollout(trans, net, cmodel, clip, mean, std, ns, texts, goals,
                           start_pose, prefix, occ, extent, actions=acts)
            if len(segs) != len(goals):
                continue
            gt_end = xy[min(cuts[-1][1] - 1, xy.shape[0] - 1)]
            drift_v = float(np.linalg.norm(segs[-1]["world"][-1, 0, :2] - gt_end))
            goal_v = float(np.mean([s["goal_err"] for s in segs]))
            null_v = float(np.linalg.norm(xy[0] - gt_end))
            seam_v = float(np.mean([s["seam_err"] for s in segs[1:]])) if n > 1 else None
            out[n]["drift"].append(drift_v)
            out[n]["goal"].append(goal_v)
            if seam_v is not None:
                out[n]["seam"].append(seam_v)
            out[n]["null"].append(null_v)
            ba = (by_action[n].setdefault(rec.action,
                    {"drift": [], "seam": [], "goal": [], "oracle": [], "null": []})
                  if by_action is not None else None)
            if ba is not None:
                ba["drift"].append(drift_v)
                ba["goal"].append(goal_v)
                ba["null"].append(null_v)
                if seam_v is not None:
                    ba["seam"].append(seam_v)

            # oracle: same cuts, ground-truth tokens, identical placement path
            pose = start_pose.copy()
            last = None
            with torch.no_grad():
                for a, b in cuts:
                    seg = cm[a:b + 1]
                    try:
                        ds, *_ = mf.humanise_positions_to_263(seg)
                    except Exception:
                        last = None; break
                    T = (min(ds.shape[0], 196) // 4) * 4
                    if T < 4:
                        last = None; break
                    ds = ds[:T].astype(np.float32)
                    x = torch.from_numpy((ds - mean) / std).unsqueeze(0).to(DEV)
                    m = net(x)[0][0].cpu().numpy() * std + mean
                    w = se2_place_full_body(m.astype(np.float32), pose, mf)
                    yaw = yaw_from_joints(w[-1])
                    pose = np.array([w[-1, 0, 0], w[-1, 0, 1], np.sin(yaw), np.cos(yaw)], np.float32)
                    last = w
            if last is not None:
                oracle_v = float(np.linalg.norm(last[-1, 0, :2] - gt_end))
                out[n]["oracle"].append(oracle_v)
                if ba is not None:
                    ba["oracle"].append(oracle_v)
            ok = True
        if ok:
            used += 1
            if used % 30 == 0:
                print(f"  {used}/{args.n_clips} clips", flush=True)

    print(f"\nchained on {used} clips\n")
    print(f"{'N segs':<8}{'final drift':>13}{'oracle drift':>14}{'null':>9}{'seam':>10}{'goal':>9}{'n':>6}")
    for n in args.chain_lengths:
        r = out[n]
        if not r["drift"]:
            continue
        seam = f"{np.mean(r['seam'])*1000:>9.1f}" if r["seam"] else f"{'-':>9}"
        orc = f"{np.mean(r['oracle']):>13.3f}" if r["oracle"] else f"{'-':>13}"
        print(f"{n:<8}{np.mean(r['drift']):>12.3f}m{orc}m{np.mean(r['null']):>8.3f}m"
              f"{seam}mm{np.mean(r['goal']):>8.3f}m{len(r['drift']):>6}")
    print("\nfinal drift = |chain end - ground-truth end|. Compare against ORACLE (same cuts,\n"
          "GT tokens, same placement path) -- drift at the oracle level is the tokenizer, not\n"
          "chaining. NULL is the clip's total displacement, i.e. the do-nothing score.")

    if by_action is not None:
        print("\n=== per-action breakdown (Step 11) ===")
        print("N=1 drift/goal test each action's single-segment generation from its OWN true\n"
              "start prefix -- for 'stand up' that prefix is a FULLY-SEATED frame 0, the hardest\n"
              "case. Read drift/goal against that action's ORACLE, never against walk's.")
        for n in args.chain_lengths:
            actions = by_action[n]
            if not any(v["drift"] for v in actions.values()):
                continue
            print(f"\nN={n}")
            print(f"{'action':<10}{'drift':>11}{'oracle':>11}{'null':>9}{'seam':>10}{'goal':>9}{'n':>6}")
            for a in ["walk", "sit", "stand up", "lie"]:
                r = actions.get(a)
                if not r or not r["drift"]:
                    continue
                seam = f"{np.mean(r['seam'])*1000:>9.1f}" if r["seam"] else f"{'-':>9}"
                orc = f"{np.mean(r['oracle']):>10.3f}" if r["oracle"] else f"{'-':>10}"
                print(f"{a:<10}{np.mean(r['drift']):>10.3f}m{orc}m{np.mean(r['null']):>8.3f}m"
                      f"{seam}mm{np.mean(r['goal']):>8.3f}m{len(r['drift']):>6}")


if __name__ == "__main__":
    main()
