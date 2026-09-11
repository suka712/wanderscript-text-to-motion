#!/usr/bin/env python3
"""Path B / Stage 1 / Step 0 — distillation-data generator (make the model scene-AWARE).

THE PROBLEM (docs/RESULTS §9, §13; memory scene-blind-model-data-limit): the trained transformer
is scene-BLIND. Fed the occupancy crop and decoded greedily it collides >= a straight line — the
scene input gets no gradient in HUMANISE (0.63 m mean displacement, nobody detours), so it is
learned-to-be-ignored. Every scene-respecting behaviour in the demo comes from OUTSIDE the net
(A* planner, guided_seg selector). This makes the NET itself learn to avoid, by distillation.

MECHANISM (ReST / behaviour-cloning from the guided_seg teacher). guided_seg already picks, per
~1 m segment, a WINNING token sequence that minimises goal_err + w*collision over n_cand samples
of the model's own distribution (collision_guided.py §13). Those winners route around obstacles a
straight/greedy path clips. We log each winner as a (conditioning -> tokens) pair in the EXACT
step10 train.pkl schema and finetune the model to make GREEDY reproduce them (distill_finetune via
train_probe --init-ckpt). No motion synthesis, no VQ-VAE round trip: the winner IS a token
sequence, which is precisely the training target.

CEILING (honest): guided_seg only selects among samples the model can already draw, so this can
only reinforce detour behaviour already in the model's support. §13 shows that support beats a
straight line, so there is real signal to reinforce; if greedy still won't cross below the line
after distillation, that is the evidence to escalate to synthetic curved-walk data (Stage 1 Step 1).

SCENE SPLIT: generation runs on GEN scenes only; a held-out EVAL scene list is written alongside so
distill_eval measures GENERALISATION (new scenes, fresh waypoints), not memorisation (risk #4).

Reuses collision_guided/rollout/demo_rollout helpers unchanged; collision_guided.py stays untouched.
"""
import argparse
import hashlib
import json
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
from humanise_join import build_flat_join, get_record, compute_track2  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402
from rollout import load_model, build_cond, occ_crop as occ_crop_fn, OCC_N  # noqa: E402
from collision_guided import (safe_sample, decode_place, path_collision,  # noqa: E402
                              straight_line_collision, _reorient_pose, _handoff)
from demo_rollout import sample_waypoints, px_to_world  # noqa: E402

T2M = os.environ["WANDER_T2M_GPT_ROOT"]
HUMANISE = os.environ["WANDER_HUMANISE_ROOT"]
BEV = os.path.expanduser("~/wander_data/bev_cache")
TALL = os.path.expanduser("~/wander_data/bev_tall_cache")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def scene_is_eval(scene, eval_frac):
    """Deterministic per-scene gen/eval split (stable across runs/seeds)."""
    h = int(hashlib.md5(scene.encode()).hexdigest(), 16) % 1000
    return h < int(eval_frac * 1000)


def _leg_collision(a, b, tall, extent, step=0.05):
    """Fraction of the straight a->b segment that lands on the 0.9 m tall raster."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    k = max(2, int(np.linalg.norm(b - a) / step))
    return path_collision(a + (b - a) * np.linspace(0, 1, k)[:, None], tall, extent)


def sample_hard_waypoints(occ, tall, extent, start_xy, n, min_step, max_step, rng,
                          hard_frac=0.7, lo=0.05, hi=0.6, margin_px=6):
    """Like demo_rollout.sample_waypoints, but BIASED so each leg has an obstacle partly in
    the way. WHY: the free-space sampler produces clear legs (faithful to A*-fed inference,
    but nothing to avoid -> a distilled 'walk straight' with no scene gradient). For the PoC
    we need AVOIDANCE demonstrations: legs where the straight line clips the tall raster
    (collision in (lo, hi] -- something to route around, but not a full wall) yet the endpoint
    is free/reachable. With prob hard_frac pick such a leg; else a normal clear leg (so the
    distilled set still contains straight-when-clear walking). Uses a SMALL erosion so
    waypoints may sit near obstacles (where clipping happens); the endpoint free-check keeps
    them reachable."""
    free = occ < 0.5
    try:
        from scipy import ndimage
        free = ndimage.binary_erosion(free, np.ones((margin_px, margin_px)))
    except ImportError:
        pass
    rr, cc = np.nonzero(free)
    if len(rr) < 50:
        return None
    pts = np.stack([px_to_world((r, c), extent, occ.shape) for r, c in zip(rr, cc)])
    out, cur = [], np.asarray(start_xy, float)
    for _ in range(n):
        d = np.linalg.norm(pts - cur, axis=1)
        band = np.nonzero((d > min_step) & (d <= max_step))[0]
        if len(band) == 0:
            band = np.nonzero(d > min_step)[0]
        if len(band) == 0:
            break
        # subsample candidates for the (cheap but not free) per-leg collision check
        if len(band) > 200:
            band = rng.choice(band, 200, replace=False)
        want_hard = rng.rand() < hard_frac
        pick = None
        if want_hard:
            hard = [j for j in band if lo < _leg_collision(cur, pts[j], tall, extent) <= hi]
            if hard:
                pick = pts[hard[rng.randint(len(hard))]]
        if pick is None:
            pick = pts[band[rng.randint(len(band))]]
        out.append(pick)
        cur = pick
    return out if len(out) == n else None


def gen_chain_records(trans, net, cmodel, mean, std, ns, texts, wps, start_pose, prefix,
                      occ, extent, tall, n_cand, coll_weight, reorient=True):
    """One guided_seg chain. Returns (records, chain_coll, chain_goal). Each record is a
    distillation entry in step10 train.pkl schema, captured at the WINNING segment: the
    conditioning that produced the winning tokens + those tokens."""
    cmean = np.array(ns["cond_mean"], np.float32)
    cstd = np.array(ns["cond_std"], np.float32)
    pose = np.asarray(start_pose, np.float32).copy()
    pfx = np.asarray(prefix, np.float32).copy()
    recs, seg_worlds = [], []
    with torch.no_grad():
        for txt, goal in zip(texts, wps):
            if reorient:
                pose = _reorient_pose(pose, goal)  # face-travel fix (matches the demo/§13)
            feat = cmodel.encode_text(clip.tokenize([txt], truncate=True).to(DEV)).float()
            extra = build_cond(ns["cond_mode"], np.asarray(goal, float), pose, pfx,
                               occ, extent, cmean, cstd)
            cond = torch.cat([feat, torch.from_numpy(extra).unsqueeze(0).to(DEV)], -1)

            # guided_seg: candidate 0 = greedy argmax, rest stochastic -> keep argmin(ge + w*coll).
            cands = []
            for j in range(max(1, n_cand)):
                tok = safe_sample(trans, cond, if_categorial=(j > 0))
                if tok.numel() == 0:
                    continue
                world, local = decode_place(net, tok, pose, mean, std)
                ge = float(np.linalg.norm(world[-1, 0, :2] - np.asarray(goal, float)))
                cl = path_collision(world[:, 0, :2], tall, extent)
                cands.append((ge + coll_weight * cl, ge, cl, tok, world, local))
            if not cands:
                break
            greedy_cl = cands[0][2]  # candidate 0 is always greedy argmax
            _, ge, cl, tok, world, local = min(cands, key=lambda t: t[0])

            toks = tok.detach().cpu().numpy().ravel().astype(np.int64)
            if toks.size == 0 or int(toks.max()) >= 512:
                # forward_decoder worked, so tokens are pure code indices; guard anyway.
                pose, pfx = _handoff(world, local)
                seg_worlds.append(world)
                continue
            yaw = float(np.arctan2(pose[2], pose[3]))
            recs.append({
                "tokens": toks,
                "text": txt,
                "goal": np.asarray(goal, np.float32),          # world
                "start": pose.copy().astype(np.float32),        # world (x,y,sin,cos), post-reorient
                "prefix_pose": pfx.copy().astype(np.float32),   # 66, root-relative heading-canon
                "occ_crop": occ_crop_fn(occ, extent, pose[:2], yaw).astype(np.float32),  # 784
                "action": "walk",
                # metadata (NOT read by train_probe; used to filter which segments to distill):
                "seg_coll": float(cl),
                "greedy_coll": float(greedy_cl),
                "goal_err": float(ge),
            })
            seg_worlds.append(world)
            pose, pfx = _handoff(world, local)
    if not seg_worlds:
        return recs, None, None
    path = np.concatenate([w[:, 0, :2] for w in seg_worlds])
    return recs, path_collision(path, tall, extent), float(np.mean([r["goal_err"] for r in recs]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="teacher model (e.g. step10/checkpoints/goalaug)")
    ap.add_argument("--vqvae-ckpt", required=True)
    ap.add_argument("--out", required=True, help="output dir for distill.pkl + eval_scenes.json")
    ap.add_argument("--n-chains", type=int, default=200)
    ap.add_argument("--n-segments", type=int, default=6)
    ap.add_argument("--n-cand", type=int, default=8)
    ap.add_argument("--coll-weight", type=float, default=10.0)
    ap.add_argument("--min-step", type=float, default=0.6)
    ap.add_argument("--max-step", type=float, default=1.2)
    ap.add_argument("--hard-frac", type=float, default=0.7,
                    help="fraction of legs biased to have an obstacle partly in the way "
                         "(avoidance demos). 0 = the plain free-space sampler.")
    ap.add_argument("--eval-frac", type=float, default=0.2, help="fraction of scenes held out for eval")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    net = load_vqvae(ckpt_path=args.vqvae_ckpt, device=DEV); net.eval()
    meta = f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta"
    mean = np.load(f"{meta}/mean.npy").astype(np.float32)
    std = np.load(f"{meta}/std.npy").astype(np.float32)
    cmodel, _ = clip.load("ViT-B/32", device=DEV, jit=False); cmodel.eval()
    trans, ns = load_model(args.ckpt)
    assert ns["cond_mode"] == "full", f"expected cond_mode=full, got {ns['cond_mode']}"
    print(f"teacher cond_mode={ns['cond_mode']}  n_chains={args.n_chains}  n_cand={args.n_cand}  "
          f"w={args.coll_weight}  eval_frac={args.eval_frac}\n", flush=True)

    flat = build_flat_join()
    walk_idx = [i for i, p in enumerate(flat) if p["action"] == "walk"]
    rng = np.random.RandomState(args.seed); rng.shuffle(walk_idx)

    all_recs, eval_scenes, gen_scenes = [], set(), set()
    chain_colls, line_colls, done = [], [], 0
    for idx in walk_idx:
        if done >= args.n_chains:
            break
        rec = get_record(int(idx))
        if scene_is_eval(rec.scene, args.eval_frac):
            eval_scenes.add(rec.scene)
            continue  # never generate on held-out scenes
        fb, ft = os.path.join(BEV, f"{rec.scene}.npz"), os.path.join(TALL, f"{rec.scene}.npz")
        if not (os.path.exists(fb) and os.path.exists(ft)):
            continue
        zb, zt = np.load(fb), np.load(ft)
        occ, extent = zb["occ"].astype(np.float32), zb["extent"]
        tall = zt["occ"].astype(np.float32)
        cm = np.load(os.path.join(HUMANISE, "contact_motion", "motions", f"{idx:05d}.npy"))
        try:
            d0, *_ = mf.humanise_positions_to_263(cm)
        except Exception:
            continue
        if d0.shape[0] < 8:
            continue
        _, xy, _, sincos = compute_track2(rec)
        start_pose = np.array([xy[0, 0], xy[0, 1], sincos[0, 0], sincos[0, 1]], np.float32)
        prefix = mf.local_joint_positions(d0.astype(np.float32))[0].ravel()
        if args.hard_frac > 0:
            wps = sample_hard_waypoints(occ, tall, extent, start_pose[:2], args.n_segments,
                                        args.min_step, args.max_step, rng, hard_frac=args.hard_frac)
        else:
            wps = sample_waypoints(occ, extent, start_pose[:2], args.n_segments, args.min_step,
                                   rng, max_step=args.max_step)
        if wps is None:
            continue
        texts = ["walk to the target"] * args.n_segments
        line = straight_line_collision(start_pose[:2], wps, tall, extent)
        recs, ccoll, cgoal = gen_chain_records(trans, net, cmodel, mean, std, ns, texts, wps,
                                               start_pose, prefix, occ, extent, tall,
                                               args.n_cand, args.coll_weight)
        if not recs or ccoll is None:
            continue
        all_recs.extend(recs)
        gen_scenes.add(rec.scene)
        chain_colls.append(ccoll); line_colls.append(line); done += 1
        if done % 10 == 0 or done == args.n_chains:
            print(f"  [{done}/{args.n_chains}] {rec.scene}  segs={len(all_recs)}  "
                  f"chain_coll={ccoll*100:.1f}%  line={line*100:.1f}%  goal={cgoal:.2f}m", flush=True)

    out_pkl = os.path.join(args.out, "distill.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(all_recs, f)
    with open(os.path.join(args.out, "eval_scenes.json"), "w") as f:
        json.dump({"eval_scenes": sorted(eval_scenes), "gen_scenes": sorted(gen_scenes),
                   "eval_frac": args.eval_frac, "n_chains": done, "n_segments": args.n_segments,
                   "n_cand": args.n_cand, "coll_weight": args.coll_weight, "seed": args.seed}, f)
    n_helped = sum(1 for r in all_recs if r["seg_coll"] < r["greedy_coll"] - 1e-9)
    print(f"\n=== {done} chains -> {len(all_recs)} distilled segments ===")
    print(f"  guided chain collision {np.mean(chain_colls)*100:.2f}%  vs line {np.mean(line_colls)*100:.2f}%")
    print(f"  segments where guided beat greedy on collision: {n_helped}/{len(all_recs)} "
          f"({100*n_helped/max(len(all_recs),1):.0f}%)")
    print(f"  gen scenes {len(gen_scenes)}  held-out eval scenes {len(eval_scenes)}")
    print(f"\nsaved {out_pkl}\nsaved {os.path.join(args.out, 'eval_scenes.json')}")


if __name__ == "__main__":
    main()
