#!/usr/bin/env python3
"""Composed cross-action rollout: walk -> sit -> stand -> walk (build-order step 11).

This is the FIRST rollout in the project that chains DIFFERENT actions, and the
setting done-criteria 4 and 5 actually require -- every prior chained rollout
(demo_rollout.py, render_chain_video.py) is walk-only with goals on free floor,
which CLAUDE.md section 1 defines as a failure.

WHY IT IS SEEDED FROM A REAL SIT CLIP. A composed chain has no ground truth, so
there is no oracle for the whole chain (CLAUDE.md risk #4 forbids reading a model
number off an un-oracled pipeline). What a real sit clip gives us instead is a
scene that is KNOWN to contain sit-able furniture and the WORLD LOCATION of the
sit target (the seated pelvis at the clip's last frame). We reuse only that
geometry; the motion is entirely generated.

THE FOUR SEGMENTS, and where each goal/text comes from:
  1 walk  goal = a standing spot ~0.5 m in front of the furniture; text "walk to <obj>"
  2 sit   goal = the real sit target (seated pelvis xy);            text = clip utterance
  3 stand goal = back to the standing spot in front;                text "stand up from <obj>"
  4 walk  goal = a free-floor waypoint away from the furniture;     text "walk to the door"
Goals are world coordinates; rollout() expresses each in its own segment's start
frame (RESULTS section 4) and hands the DECODED ending pose forward as the next
segment's prefix (RESULTS section 7). The seam INTO segment 2 is a standing prefix
(like the N=1 sit case that beat null); the seam INTO segment 3 is the GENERATED
seated pose -- the one thing eval_accumulation's N=1 stand-up test could not cover,
since it used a real seated prefix.

VERIFICATION WITHOUT AN ORACLE. We cannot oracle the composed chain, so we check
STRUCTURE instead: pelvis height must DROP during the sit (standing ~0.9 m ->
seated ~0.5 m) and RISE back during the stand. A sit segment whose pelvis never
drops did not sit, whatever the goal error says. Printed per segment; the pass
flag is `sat` (seg-2 end pelvis below --sit-z) AND `stood` (seg-3 end pelvis back
above --stand-z).
"""
import argparse
import os
import sys

import clip
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
import torch

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
for p in ["src", "scripts/track1", "scripts/chaining"]:
    sys.path.insert(0, os.path.join(REPO_ROOT, p))

import motion_features as mf  # noqa: E402
from humanise_join import build_flat_join, get_record, compute_track2, J_PELVIS  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402
from rollout import rollout, load_model, blend_seam  # noqa: E402
from demo_rollout import sample_waypoints  # noqa: E402
from render_chain_video import CHAIN, SEGC, FPS  # noqa: E402

T2M = os.environ.get("WANDER_T2M_GPT_ROOT")
HUMANISE = os.environ.get("WANDER_HUMANISE_ROOT")
BEV = os.path.expanduser("~/wander_data/bev_cache")
TALL = os.path.expanduser("~/wander_data/bev_tall_cache")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def object_phrase(utterance):
    """'sit on the coffee table' -> 'the coffee table'. Falls back to 'the seat'."""
    u = utterance.strip().lower()
    if " on " in u:
        return u.split(" on ", 1)[1].strip()
    if " in " in u:
        return u.split(" in ", 1)[1].strip()
    return "the seat"


MAX_HOP = 1.1  # per-segment walk range; the model, trained on 0.63 m mean displacement,
               # covers ~1.3 m before stopping (RESULTS section 9). Longer goals undershoot.


def compose_goals_texts(start_xy, sit_xy, occ, extent, rng, front=0.3, stand=0.5, away=1.3):
    """Build the goals for a walk-up -> sit -> stand -> walk-away chain.

    The walk-up from start_xy to `front` metres in front of the furniture is split into
    as many in-range hops (<= MAX_HOP) as it takes, because a single long walk segment
    UNDERSHOOTS (the model walks ~1.3 m and stops) and then hands the sit segment a goal
    that is still metres away -- and a long sit goal makes the model WALK, not sit
    (sit clips barely move, so action=sit is entangled with a ~0.1-0.5 m goal). So the
    walk-up must actually DELIVER the body to the furniture; only then is the sit goal
    short enough to be in distribution. Returns (goals, n_walk_in).

    front: how far in front of the sit target the walk-up ends (short, so the sit goal is
      short). stand: where the stand-up ends. away: the walk-away distance.
    """
    start_xy = np.asarray(start_xy, float)
    sit_xy = np.asarray(sit_xy, float)
    d = start_xy - sit_xy
    n = np.linalg.norm(d)
    u = d / (n if n > 1e-6 else 1e-6)              # unit vector furniture -> start
    approach_pt = sit_xy + front * u               # end of the walk-up, just off the furniture
    span = np.linalg.norm(approach_pt - start_xy)
    n_walk_in = max(1, int(np.ceil(span / MAX_HOP)))
    walk_goals = [start_xy + (approach_pt - start_xy) * ((k + 1) / n_walk_in)
                  for k in range(n_walk_in)]
    stand_pt = sit_xy + stand * u
    wp = sample_waypoints(occ, extent, stand_pt, 1, min_step=max(0.6, away - 0.4),
                          rng=rng, max_step=away + 0.4)
    away_pt = wp[0] if wp else stand_pt + away * u
    goals = walk_goals + [sit_xy, stand_pt, away_pt]
    return goals, n_walk_in


def pelvis_z(seg):
    return seg["world"][:, J_PELVIS, 2]


def collision(path, tall, extent):
    xmin, xmax, ymin, ymax = extent
    H, W = tall.shape
    c = np.clip(((path[:, 0] - xmin) / (xmax - xmin) * W).astype(int), 0, W - 1)
    r = np.clip(((ymax - path[:, 1]) / (ymax - ymin) * H).astype(int), 0, H - 1)
    return float((tall[r, c] > .5).mean())


def render(segs, texts, occ, extent, tall, wps, sit_xy, out_path, title, blend_n):
    worlds, prev = [], None
    for s in segs:
        w = blend_seam(prev, s["world"], n=blend_n)
        worlds.append(w); prev = w
    allw = np.concatenate(worlds)
    seg_of = np.concatenate([[i] * len(w) for i, w in enumerate(worlds)])
    path = allw[:, 0, :2]
    xmin, xmax, ymin, ymax = extent
    H, W = tall.shape
    pc = np.clip(((path[:, 0] - xmin) / (xmax - xmin) * W).astype(int), 0, W - 1)
    pr = np.clip(((ymax - path[:, 1]) / (ymax - ymin) * H).astype(int), 0, H - 1)
    coll = collision(path, tall, extent)
    dist = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))

    def w2px(p):
        return (np.clip(int((p[0] - xmin) / (xmax - xmin) * W), 0, W - 1),
                np.clip(int((ymax - p[1]) / (ymax - ymin) * H), 0, H - 1))

    fig = plt.figure(figsize=(14, 6.2))
    axm = fig.add_subplot(1, 2, 1)
    axm.imshow(occ, cmap="gray_r")
    for i in range(len(worlds)):
        m = seg_of == i
        axm.plot(pc[m], pr[m], "-", lw=2.4, color=SEGC(i / max(1, len(worlds) - 1)))
    for i, wp in enumerate(wps):
        gx, gy = w2px(wp)
        axm.plot([gx], [gy], "*", color="#d62728", ms=12, ls="none")
    sx, sy = w2px(sit_xy)
    axm.plot([sx], [sy], "s", color="#1f77b4", ms=11, ls="none", label="sit target")
    axm.plot(pc[0], pr[0], "o", color="#2ca02c", ms=11, label="start")
    dot, = axm.plot([], [], "o", color="#111", ms=9)
    axm.set_title(f"{title} · path {dist:.1f} m · collision {coll*100:.1f}%", fontsize=9)
    axm.axis("off"); axm.legend(fontsize=7, loc="lower right")

    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    a, b, c3 = allw[..., 0], allw[..., 1], allw[..., 2]
    cx, cy = (a.min() + a.max()) / 2, (b.min() + b.max()) / 2
    r = max(a.ptp(), b.ptp(), 1e-6) / 2 + 0.4
    ax3.set_xlim(cx - r, cx + r); ax3.set_ylim(cy - r, cy + r)
    ax3.set_zlim(min(c3.min(), 0) - .05, max(c3.max(), 1.8))
    ax3.set_box_aspect([1, 1, .8]); ax3.tick_params(labelsize=5)
    ax3.plot(path[:, 0], path[:, 1], np.zeros(len(path)), "-", color="#888", lw=1)
    lines = [ax3.plot([], [], [], lw=2.4, marker="o", ms=3)[0] for _ in CHAIN]
    fig.suptitle("composed rollout — walk to furniture, sit, stand, walk away", fontsize=11)
    fig.tight_layout()

    def update(f):
        p = allw[f]
        col = SEGC(seg_of[f] / max(1, len(worlds) - 1))
        for ln, ch in zip(lines, CHAIN):
            ln.set_data(p[ch, 0], p[ch, 1]); ln.set_3d_properties(p[ch, 2])
            ln.set_color(col)
        dot.set_data([pc[f]], [pr[f]])
        ax3.set_title(f"segment {seg_of[f]+1}/{len(worlds)}: {texts[seg_of[f]]!r}", fontsize=9)
        return []

    FuncAnimation(fig, update, frames=len(allw), interval=1000 / FPS).save(
        out_path, writer=FFMpegWriter(fps=FPS, bitrate=2600))
    plt.close(fig)
    return coll, dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vqvae-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed-action", default="sit", choices=["sit", "lie"],
                    help="which interaction clips to seed scenes/targets from")
    ap.add_argument("--n-demos", type=int, default=6)
    ap.add_argument("--min-approach", type=float, default=2.5,
                    help="require the sit target at least this far from the clip start, so "
                         "segment 1 can walk AND segment 2 still has a ~--front approach to sit over")
    ap.add_argument("--front", type=float, default=1.5,
                    help="distance in front of the sit target where segment 1 stops and "
                         "segment 2's approach+sit begins. Short values (0.5) do NOT sit -- the "
                         "sit motion needs its approach inside the same segment (see compose_goals_texts)")
    ap.add_argument("--stand", type=float, default=0.5,
                    help="distance in front of the sit target where the stand-up ends")
    ap.add_argument("--away", type=float, default=1.3, help="segment-4 walk-away distance")
    ap.add_argument("--reorient", action="store_true",
                    help="rotate each walk segment's start to face its goal so the body faces "
                         "where it walks (inference heading fix; keeps clean seams)")
    ap.add_argument("--start-dist", type=float, default=0.0,
                    help="if >0, synthesize the chain start this far from the furniture on free "
                         "floor (facing it) so segment 1 is a real walk-up, instead of using the "
                         "clip's own short approach. Bypasses --min-approach.")
    ap.add_argument("--sit-z", type=float, default=0.7,
                    help="pelvis height (m) below which segment 2 counts as having sat")
    ap.add_argument("--stand-z", type=float, default=0.8,
                    help="pelvis height (m) above which segment 3 counts as having stood")
    ap.add_argument("--blend-n", type=int, default=6,
                    help="display-only crossfade frames at each seam. Interaction seams run "
                         "~2x walk seams (Step 11a), so the default is longer than the walk "
                         "demo's 4.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    net = load_vqvae(ckpt_path=args.vqvae_ckpt, device=DEV); net.eval()
    mean = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy").astype(np.float32)
    std = np.load(f"{T2M}/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy").astype(np.float32)
    cmodel, _ = clip.load("ViT-B/32", device=DEV, jit=False); cmodel.eval()
    trans, ns = load_model(args.ckpt)

    flat = build_flat_join()
    seed_idx = [i for i, p in enumerate(flat) if p["action"] == args.seed_action]
    rng = np.random.RandomState(args.seed)
    rng.shuffle(seed_idx)

    print(f"model cond_mode={ns['cond_mode']}  seeding from {len(seed_idx)} "
          f"'{args.seed_action}' clips\n")
    made, rows = 0, []
    for idx in seed_idx:
        if made >= args.n_demos:
            break
        rec = get_record(int(idx))
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
        sit_xy = xy[-1].astype(np.float32)
        prefix = mf.local_joint_positions(d0.astype(np.float32))[0].ravel()
        if args.start_dist > 0:
            # Synthesize a chain start ~start_dist from the furniture on free floor, facing it,
            # so segment 1 is a REAL walk-up. HUMANISE sit clips only carry ~1.3 m of approach
            # (median 0.11 m), too short to show; the fixed model follows arbitrary walk goals,
            # so we place the start ourselves. The prefix stays the clip's standing frame-0 pose.
            sp = sample_waypoints(occ, extent, sit_xy, 1, min_step=max(0.6, args.start_dist - 0.5),
                                  rng=rng, max_step=args.start_dist + 0.5)
            if not sp:
                continue
            s_xy = np.asarray(sp[0], float)
            d = sit_xy - s_xy
            yaw = float(np.arctan2(d[1], d[0]))          # face the furniture
            start_pose = np.array([s_xy[0], s_xy[1], np.sin(yaw), np.cos(yaw)], np.float32)
        else:
            start_pose = np.array([xy[0, 0], xy[0, 1], sincos[0, 0], sincos[0, 1]], np.float32)
            if np.linalg.norm(start_pose[:2] - sit_xy) < args.min_approach:
                continue

        goals, nwi = compose_goals_texts(start_pose[:2], sit_xy, occ, extent, rng,
                                         front=args.front, stand=args.stand, away=args.away)
        obj = object_phrase(rec.utterance)
        stand_txt = f"stand up from {obj}" if args.seed_action == "sit" else f"get up from {obj}"
        # nwi walk-up segments deliver the body to the furniture, then sit / stand / walk away.
        texts = [f"walk to {obj}"] * nwi + [rec.utterance, stand_txt, "walk to the door"]
        actions = ["walk"] * nwi + [args.seed_action, "stand up", "walk"]

        segs = rollout(trans, net, cmodel, clip, mean, std, ns, texts, goals,
                       start_pose, prefix, occ, extent,
                       actions=actions if ns["cond_mode"] in ("full_action", "full_action_head") else None,
                       reorient=args.reorient)
        if len(segs) < len(goals):
            continue

        zs = [pelvis_z(s) for s in segs]
        sat = float(zs[nwi][-1]) < args.sit_z          # the sit segment is index nwi
        stood = float(zs[nwi + 1][-1]) > args.stand_z  # stand-up is the next one
        goal_errs = [s["goal_err"] for s in segs]
        seam_errs = [s["seam_err"] for s in segs[1:]]
        title = f"{rec.scene} · {obj}"
        out_path = os.path.join(args.out, f"demo_{made:02d}_{rec.scene}_{idx}.mp4")
        coll, dist = render(segs, texts, occ, extent, tall, goals, sit_xy, out_path,
                            title, args.blend_n)

        rows.append(dict(idx=idx, scene=rec.scene, obj=obj, sat=sat, stood=stood,
                         coll=coll, dist=dist, goal_errs=goal_errs, seam_errs=seam_errs, zs=zs))
        made += 1
        print(f"[{made}/{args.n_demos}] {rec.scene}  {obj}")
        print(f"    pelvis-z end/min per seg: "
              + "  ".join(f"s{i+1}={z[-1]:.2f}/{z.min():.2f}" for i, z in enumerate(zs)))
        print(f"    SAT={sat}  STOOD={stood}  "
              f"goal_err={[round(g,2) for g in goal_errs]}  "
              f"seam(mm)={[round(s*1000) for s in seam_errs]}  "
              f"coll={coll*100:.1f}%  path={dist:.1f}m", flush=True)

    if not rows:
        print("no demos produced"); return
    n_sat = sum(r["sat"] for r in rows)
    n_stood = sum(r["sat"] and r["stood"] for r in rows)
    print(f"\n=== {len(rows)} composed chains: {n_sat} sat, {n_stood} sat AND stood ===")
    print("A demo where SAT and STOOD are both true is a watchable walk->sit->stand->walk "
          "(done-criteria 4 and 5). Structure check only -- there is no oracle for a "
          "composed chain (CLAUDE.md risk #4).")


if __name__ == "__main__":
    main()
