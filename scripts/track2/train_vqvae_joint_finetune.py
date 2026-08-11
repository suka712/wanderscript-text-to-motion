#!/usr/bin/env python3
"""
Track 2 -- joint VQ-VAE finetune on HumanML3D + HUMANISE (diagnostic; see
docs/track_2/001_tokenizer_finetune.md). Unfreezes the frozen T2M-GPT VQ-VAE
(encoder + codebook + decoder) and finetunes it jointly on both datasets with
balanced sampling, to test whether HUMANISE-lie reconstruction (139.8mm on
the frozen model, 3.09x the H3D baseline) improves toward the H3D-lie control
(~90mm) without regressing general motion (H3D baseline ~45mm, HUMANISE-walk
~48mm).

Architecture / loss recipe mirrors T2M-GPT's own train_vq.py exactly (same
ReConsLoss, same commit/loss_vel weights as the ORIGINAL from-scratch run --
see pretrained/VQVAE/run.log: commit=0.02, loss_vel=0.5, recons_loss=
l1_smooth) so this is a genuine finetune of that recipe, not a different
training regime. What's different from train_vq.py:
  1. Joint balanced dataloader (src/joint_vqvae_dataset.py) instead of
     T2M-GPT's single-dataset dataset_VQ.DATALoader.
  2. Warm-started from net_best_fid.pth (resume_pth), not from scratch.
  3. Much lower LR (default 2e-5, vs 2e-4 for from-scratch training) --
     warm-starting a converged model at the from-scratch LR risks wrecking
     the codebook before it has a chance to adapt (CLAUDE.md failure point
     3: "may forget general motion").
  4. Held-out per-category MPJPE eval (scripts/track2/eval_per_category_mpjpe.py)
     instead of T2M-GPT's own FID-based eval_trans.evaluation_vqvae -- this
     project's decision gate is stated in MPJPE (CLAUDE.md 2b, 001_tokenizer_
     finetune.md), not FID.
  5. Fine-grained heartbeat logging to a dedicated file, independent of the
     main console/run log -- learned the hard way (see docs/old_docs_aug8/
     STEP2_baseline_calibration.md) that unattended multi-hour jobs need
     progress evidence finer-grained than "still running" so a stall is
     distinguishable from a slow-but-healthy job without multi-hour forensics.

Normalization: evaluator-consistent (frozen checkpoint's own meta mean/std),
same choice as eval_per_category_mpjpe.py and used for BOTH datasets during
training -- see that script's docstring for the explicit reasoning. This is
the "keep using the frozen checkpoint's original norm stats" branch of the
decision the pitfall note in 001_tokenizer_finetune.md asks for; the warm-
started encoder/decoder/codebook is already calibrated to this distribution,
and recomputing joint-dataset stats mid-project would silently break
comparability with every existing per-category number and require the
frozen model to be treated as out-of-distribution against its own eval.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import joint_vqvae_dataset as jd  # noqa: E402
from vqvae_loader import load_vqvae  # noqa: E402
import eval_per_category_mpjpe as evalcat  # noqa: E402

T2M_GPT_ROOT = os.environ.get("WANDER_T2M_GPT_ROOT", "/home/user/Khiem/T2M-GPT")
if T2M_GPT_ROOT not in sys.path:
    sys.path.insert(0, T2M_GPT_ROOT)
import utils.losses as losses  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-name", default="track2_joint_finetune")
    ap.add_argument("--out-dir", default="/media/user/2tb/motion_data/track2_checkpoints")
    ap.add_argument("--resume-pth", default=f"{T2M_GPT_ROOT}/pretrained/VQVAE/net_best_fid.pth")
    ap.add_argument("--resume-optimizer", default=None,
                     help="optional optimizer state to resume alongside --resume-pth")
    ap.add_argument("--start-iter", type=int, default=0)

    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--window-size", type=int, default=64)
    ap.add_argument("--h3d-frac", type=float, default=0.5)
    ap.add_argument("--num-workers", type=int, default=4)

    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warm-up-iter", type=int, default=200)
    ap.add_argument("--lr-scheduler", type=int, nargs="+", default=[15000])
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--weight-decay", type=float, default=0.0)

    ap.add_argument("--commit", type=float, default=0.02)
    ap.add_argument("--loss-vel", type=float, default=0.5)
    ap.add_argument("--recons-loss", default="l1_smooth")

    ap.add_argument("--total-iter", type=int, default=30000)
    ap.add_argument("--eval-iter", type=int, default=3000)
    ap.add_argument("--print-iter", type=int, default=100)
    ap.add_argument("--heartbeat-sec", type=float, default=15.0)
    ap.add_argument("--eval-n-clips", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    return ap.parse_args()


def prepare_quantizer_for_finetune(net):
    """CRITICAL, easy-to-miss bug this function prevents: QuantizeEMAReset
    (models/quantize_cnn.py) tracks its "has the codebook been initialized"
    state in a plain Python bool (`self.init`, set False in __init__ via
    reset_codebook()) plus two plain-attribute EMA accumulators
    (`code_sum`/`code_count`, both None until init_codebook() runs) -- NONE
    of which are registered buffers, so `net.load_state_dict(ckpt, strict=
    True)` restores the pretrained `codebook` tensor itself but leaves
    `init=False`. On the very first net.train() forward pass, the quantizer
    then runs `if self.training and not self.init: self.init_codebook(x)`,
    which OVERWRITES THE ENTIRE PRETRAINED CODEBOOK with samples tiled from
    that one batch's encoder output -- silently discarding the whole
    finetune's warm start in iteration 1. Verified empirically (see track2
    session notes / RESULTS.md): without this fix, 512/512 codebook rows are
    replaced immediately; with it, ~85-95% of rows get the intended small
    EMA nudge per step and only genuinely rare/dead codes get reset (the
    designed "Reset" behavior of QuantizeEMAReset, unavoidable and correct
    with or without loading a checkpoint).

    The fix: manually put the quantizer into the state it would be in had it
    been trained continuously and just happened to pause here -- init=True,
    code_sum seeded to the loaded codebook itself (equivalent to "every code
    already has an EMA history equal to its current, converged value"),
    code_count seeded to 1.0 per code (matches what init_codebook() itself
    would set on a from-scratch run). This exactly reproduces what
    init_codebook(codebook) would do if called with the codebook as its own
    input (since nb_code_x == nb_code, _tile is a no-op).
    """
    q = net.vqvae.quantizer
    if q.init:
        return  # already fine (e.g. resuming one of OUR mid-finetune checkpoints)
    q.init = True
    q.code_sum = q.codebook.clone()
    q.code_count = torch.ones(q.nb_code, device=q.codebook.device)


def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):
    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for pg in optimizer.param_groups:
        pg["lr"] = current_lr
    return current_lr


class Heartbeat:
    """Independent, timestamped, flushed-every-write progress log -- so a
    dead/hung process is distinguishable from a slow one without forensic
    replay (see module docstring)."""

    def __init__(self, path, every_sec):
        self.path = path
        self.every_sec = every_sec
        self.last = 0.0
        self.f = open(path, "a", buffering=1)

    def maybe(self, nb_iter, extra=""):
        now = time.time()
        if now - self.last >= self.every_sec:
            self.last = now
            line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  iter={nb_iter}  {extra}"
            self.f.write(line + "\n")
            self.f.flush()

    def write(self, msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
        self.f.write(line + "\n")
        self.f.flush()


def main():
    args = build_args()
    torch.manual_seed(args.seed)
    out_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    hb = Heartbeat(os.path.join(out_dir, "heartbeat.log"), args.heartbeat_sec)
    hb.write(f"START exp={args.exp_name} args={vars(args)}")

    mean = np.load(evalcat.EVAL_MEAN_PATH).astype(np.float32)
    std = np.load(evalcat.EVAL_STD_PATH).astype(np.float32)

    print("Loading H3D train split...")
    h3d_motions, h3d_stats = jd.load_h3d_split("train", window_size=args.window_size)
    print("Loading HUMANISE train split (from precomputed 263 cache)...")
    hum_motions, hum_stats = jd.load_humanise_split("train", window_size=args.window_size)
    hb.write(f"data loaded h3d={h3d_stats} humanise={hum_stats}")

    h3d_ds = jd.WindowMotionDataset(h3d_motions, mean, std, window_size=args.window_size)
    hum_ds = jd.WindowMotionDataset(hum_motions, mean, std, window_size=args.window_size)
    joint_loader = jd.BalancedJointLoader(
        h3d_ds, hum_ds, batch_size=args.batch_size, h3d_frac=args.h3d_frac,
        num_workers=args.num_workers, seed=args.seed,
    )
    loader_iter = iter(joint_loader)

    net = load_vqvae(ckpt_path=args.resume_pth, device=DEVICE)
    prepare_quantizer_for_finetune(net)
    net.train()
    print(f"VQ-VAE loaded from {args.resume_pth}, unfrozen, on {DEVICE}")
    hb.write(f"model loaded from {args.resume_pth}, quantizer.init patched for finetune-resume")

    optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99),
                             weight_decay=args.weight_decay)
    if args.resume_optimizer:
        optimizer.load_state_dict(torch.load(args.resume_optimizer, map_location=DEVICE))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_scheduler, gamma=args.gamma
    )

    Loss = losses.ReConsLoss(args.recons_loss, 22)

    def save_ckpt(name, nb_iter):
        path = os.path.join(out_dir, name)
        torch.save({"net": net.state_dict(), "iter": nb_iter}, path)
        torch.save(optimizer.state_dict(), os.path.join(out_dir, f"optim_{name}"))
        return path

    def run_eval(nb_iter):
        net.eval()
        t0 = time.time()
        results = evalcat.run_full_eval(net, mean=mean, std=std, n_clips=args.eval_n_clips,
                                         seed=0, verbose=True)
        dt = time.time() - t0
        net.train()
        out_json = os.path.join(out_dir, f"eval_iter{nb_iter:06d}.json")
        with open(out_json, "w") as f:
            json.dump({"iter": nb_iter, "results": results, "eval_seconds": dt}, f, indent=2)
        hb.write(f"EVAL iter={nb_iter} took {dt:.1f}s -> {out_json}")
        summary = " ".join(
            f"{k}={v['mean_mm']:.1f}mm" for k, v in results.items()
        )
        hb.write(f"EVAL SUMMARY iter={nb_iter} {summary}")
        return results

    # Eval BEFORE any finetuning (iter 0) -- held-out baseline on the exact
    # same methodology used for every later checkpoint, so "before vs after"
    # in RESULTS.md is apples-to-apples (see eval_per_category_mpjpe.py
    # docstring for why this differs slightly from check14/20's numbers).
    print("Running iter-0 (pre-finetune) held-out eval...")
    run_eval(0)
    save_ckpt("net_iter000000.pth", 0)

    print("Warmup phase...")
    avg_recons = avg_perplexity = avg_commit = 0.0
    t_start = time.time()
    for nb_iter in range(1, args.warm_up_iter + 1):
        current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)
        gt_motion = next(loader_iter).to(DEVICE).float()
        pred_motion, loss_commit, perplexity = net(gt_motion)
        loss_motion = Loss(pred_motion, gt_motion)
        loss_vel = Loss.forward_vel(pred_motion, gt_motion)
        loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_recons += loss_motion.item()
        avg_perplexity += perplexity.item()
        avg_commit += loss_commit.item()
        hb.maybe(nb_iter, extra=f"phase=warmup lr={current_lr:.2e} loss={loss.item():.5f}")

    avg_recons /= args.warm_up_iter
    avg_perplexity /= args.warm_up_iter
    avg_commit /= args.warm_up_iter
    print(f"Warmup done. Recons {avg_recons:.5f} PPL {avg_perplexity:.2f} Commit {avg_commit:.5f}")
    hb.write(f"WARMUP DONE recons={avg_recons:.5f} ppl={avg_perplexity:.2f} commit={avg_commit:.5f}")

    print(f"Main training: {args.total_iter} iters, eval every {args.eval_iter}...")
    avg_recons = avg_perplexity = avg_commit = 0.0
    for nb_iter in range(1, args.total_iter + 1):
        gt_motion = next(loader_iter).to(DEVICE).float()
        pred_motion, loss_commit, perplexity = net(gt_motion)
        loss_motion = Loss(pred_motion, gt_motion)
        loss_vel = Loss.forward_vel(pred_motion, gt_motion)
        loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        avg_recons += loss_motion.item()
        avg_perplexity += perplexity.item()
        avg_commit += loss_commit.item()

        cur_lr = optimizer.param_groups[0]["lr"]
        hb.maybe(nb_iter, extra=f"phase=train lr={cur_lr:.2e} loss={loss.item():.5f} "
                                 f"iters/sec={nb_iter / (time.time() - t_start):.2f}")

        if nb_iter % args.print_iter == 0:
            avg_recons /= args.print_iter
            avg_perplexity /= args.print_iter
            avg_commit /= args.print_iter
            elapsed = time.time() - t_start
            print(f"Iter {nb_iter}/{args.total_iter}  lr {cur_lr:.2e}  "
                  f"Recons {avg_recons:.5f}  PPL {avg_perplexity:.2f}  Commit {avg_commit:.5f}  "
                  f"({nb_iter / elapsed:.2f} it/s, {elapsed:.0f}s elapsed)")
            avg_recons = avg_perplexity = avg_commit = 0.0

        if nb_iter % args.eval_iter == 0 or nb_iter == args.total_iter:
            save_ckpt("net_last.pth", nb_iter)
            save_ckpt(f"net_iter{nb_iter:06d}.pth", nb_iter)
            run_eval(nb_iter)

    hb.write("TRAINING COMPLETE")
    print("Training complete.")


if __name__ == "__main__":
    main()
