#!/usr/bin/env python3
"""
Grounding probe training (docs/track_1/001_grounding_probe.md, "Training" +
"REQUIRED baseline").

Finetunes T2M-GPT's pretrained transformer (frozen VQ-VAE tokens as targets,
frozen CLIP ViT-B/32 for text) with --conditioned to add start-pose + goal
conditioning, or without it to train the required no-goal-conditioning
baseline. The transformer architecture itself (models.t2m_trans.Text2Motion_
Transformer, reused unmodified from T2M-GPT) is IDENTICAL across runs -- the
only difference is the width of the vector fed into its existing cond_emb
projection. This matches CLAUDE.md 2b's "learned embeddings concatenated to
its input" literally, without any model-architecture surgery: the
concatenation happens in feature space, before the existing cond_emb Linear,
so the pretrained transformer blocks/head need no changes at all.

--cond-mode selects WHICH FRAME the goal is expressed in, which turned out to
matter more than anything else in this probe:

  abs (the original run, kept for reproducibility) -- clip_dim=518: CLIP text
    [512] + start (x, y, sin, cos) [4] + goal (x, y) [2], all in ABSOLUTE
    ScanNet world coordinates, globally normalized. This is what the first
    probe measured, and it is a badly posed task: the model generates in a
    canonicalized frame whose origin and heading are the start pose, so to use
    this conditioning it must compute R(yaw0)^T (goal - start) -- a BILINEAR
    function of its own inputs. It is handed that job through a single Linear,
    which cannot represent a rotation, via 6 raw dims out of 518 that
    warm-start near zero. The null result under this mode says little about
    whether grounding is possible.

  rel (default) -- clip_dim=514: CLIP text [512] + goal expressed in the
    model's OWN generation frame [2], i.e. start-relative and heading-aligned
    (se2_utils.world_to_local_xy). This is the exact quantity the canonicalized
    representation needs, precomputed instead of demanded of a linear layer.
    The start pose carries no extra information in this frame -- it is (0, 0)
    at heading 0 by construction -- so it is dropped from the conditioning
    vector rather than fed as a constant; it still enters at inference, as the
    SE(2) placement.

The unconditioned baseline (clip_dim=512) is unaffected by --cond-mode and
does not need retraining when the mode changes.

Pretrained-checkpoint compatibility: the conditioned run's cond_emb has a
different input width than the pretrained checkpoint's (512 vs 518), so its
weight can't be strict-loaded. We warm-start it anyway: copy the pretrained
(embed_dim, 512) weight into the first 512 columns of the new (embed_dim,
518) weight (the extra 6 columns keep their fresh N(0, 0.02) init from
Text2Motion_Transformer's own _init_weights), and copy the bias directly
(unaffected, still (embed_dim,)). This preserves the pretrained text-
conditioning pathway instead of forcing the whole cond_emb to relearn from
scratch, while giving fresh capacity for the new start/goal signal.
"""
import argparse
import json
import os
import pickle
import sys

import clip
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
T2M_GPT_ROOT = os.environ.get("WANDER_T2M_GPT_ROOT", "/home/user/Khiem-ssh/T2M-GPT")
if T2M_GPT_ROOT not in sys.path:
    sys.path.insert(0, T2M_GPT_ROOT)

from se2_utils import world_to_local_xy  # noqa: E402

import models.t2m_trans as trans  # noqa: E402
import utils.utils_model as utils_model  # noqa: E402

MOTION_DATA_ROOT = os.environ.get("WANDER_MOTION_DATA_ROOT", "/media/user/2tb/motion_data")
OUT_DIR = os.environ.get("WANDER_TRACK1_PROBE_ROOT", os.path.join(os.path.dirname(MOTION_DATA_ROOT), "track1_probe"))
TOKENS_DIR = os.path.join(OUT_DIR, "tokens")
PRETRAINED_TRANS = os.path.join(T2M_GPT_ROOT, "pretrained", "VQTransformer_corruption05", "net_best_fid.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Exact architecture of the pretrained checkpoint (from its own run.log),
# NOT option_transformer.py's generic argparse defaults.
NB_CODE = 512
EMBED_DIM_GPT = 1024
CLIP_DIM_BASE = 512
BLOCK_SIZE = 51
NUM_LAYERS = 9
N_HEAD_GPT = 16
DROP_OUT_RATE = 0.1
FF_RATE = 4
PKEEP = 0.5  # matches the pretrained recipe ("VQTransformer_corruption05")
MAX_MOTION_LENGTH = BLOCK_SIZE  # 51, matches T2M-GPT's own convention for unit_length=4

MOT_END_IDX = NB_CODE
MOT_PAD_IDX = NB_CODE + 1

COND_EXTRA_DIMS = {"abs": 6, "rel": 2, "rel_prefix": 2 + 66,
                   "full": 2 + 66 + 28 * 28}


def cond_extra_raw(d, cond_mode):
    """Unnormalized conditioning vector appended to the CLIP text feature.
    See the module docstring for why `rel` exists and what `abs` got wrong."""
    if cond_mode == "abs":
        return np.concatenate([d["start"], d["goal"]]).astype(np.float32)
    if cond_mode == "rel":
        return world_to_local_xy(d["goal"], d["start"]).astype(np.float32)
    if cond_mode == "rel_prefix":
        # relative goal (2) + the previous segment's ending body configuration
        # (66 = 22 joints x 3, root-relative and heading-canonicalized, hence
        # frame-independent -- see scripts/continuation/prepare_continuation_data.py)
        return np.concatenate([
            world_to_local_xy(d["goal"], d["start"]),
            d["prefix_pose"],
        ]).astype(np.float32)
    if cond_mode == "full":
        # everything settled by the probes: relative goal (04), seam pose (07),
        # occupancy crop in the agent's frame (06). Raw occupancy goes straight
        # into the existing cond_emb Linear -- that IS the linear readout the
        # scene probe validated, so no new module is introduced.
        return np.concatenate([
            world_to_local_xy(d["goal"], d["start"]),
            d["prefix_pose"],
            d["occ_crop"],
        ]).astype(np.float32)
    raise ValueError(cond_mode)


def compute_cond_norm(manifest, cond_mode):
    """Per-dimension mean/std over the TRAIN split. For `abs`, the sin/cos
    columns are left untouched (already unit-scale), matching the original run."""
    raw = np.stack([cond_extra_raw(d, cond_mode) for d in manifest])
    mean = raw.mean(0).astype(np.float32)
    std = (raw.std(0) + 1e-6).astype(np.float32)
    if cond_mode == "abs":
        mean[2:4] = 0.0  # start sin, cos
        std[2:4] = 1.0
    return mean, std


def cond_extra(d, cond_mode, mean, std):
    return ((cond_extra_raw(d, cond_mode) - mean) / std).astype(np.float32)


class ProbeMotionDataset(Dataset):
    def __init__(self, manifest, cond_mode, cond_mean, cond_std, goal_aug=0.0, rng_seed=0):
        self.data = manifest
        self.cond_mode = cond_mode
        self.cond_mean = cond_mean
        self.cond_std = cond_std
        # goal_aug: probability of TRUNCATION augmentation. Cut the token
        # sequence at a random length L and make the goal the world position at
        # frame 4L-1, so the same clip supplies many (goal, motion) pairs at
        # varied distance and direction. Addresses RESULTS §9: the model only
        # ever saw the goal it was already reaching, so it learned to produce
        # plausible motion rather than to GO somewhere.
        # NOTE the target motion is truncated WITH the goal. Randomising the goal
        # while keeping the motion would teach the model to ignore it.
        self.goal_aug = goal_aug
        self.rng = np.random.RandomState(rng_seed)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        if self.goal_aug > 0 and "xy_traj" in d and self.rng.rand() < self.goal_aug:
            n_tok = len(d["tokens"])
            traj = d["xy_traj"]
            if n_tok >= 3 and len(traj) >= 8:
                L = self.rng.randint(2, n_tok + 1)
                fi = min(4 * L - 1, len(traj) - 1)
                d = dict(d)
                d["tokens"] = d["tokens"][:L]
                d["goal"] = traj[fi].astype(np.float32)
        m_tokens = d["tokens"].astype(np.int64)
        m_tokens_len = m_tokens.shape[0]
        if m_tokens_len + 1 < MAX_MOTION_LENGTH:
            m_tokens = np.concatenate([
                m_tokens,
                np.array([MOT_END_IDX], dtype=np.int64),
                np.full(MAX_MOTION_LENGTH - 1 - m_tokens_len, MOT_PAD_IDX, dtype=np.int64),
            ])
        else:
            m_tokens = np.concatenate([m_tokens[:MAX_MOTION_LENGTH - 1], np.array([MOT_END_IDX], dtype=np.int64)])
            m_tokens_len = MAX_MOTION_LENGTH - 1

        extra = cond_extra(d, self.cond_mode, self.cond_mean, self.cond_std)
        return d["text"], m_tokens, m_tokens_len, extra


def build_transformer(clip_dim):
    return trans.Text2Motion_Transformer(
        num_vq=NB_CODE, embed_dim=EMBED_DIM_GPT, clip_dim=clip_dim, block_size=BLOCK_SIZE,
        num_layers=NUM_LAYERS, n_head=N_HEAD_GPT, drop_out_rate=DROP_OUT_RATE, fc_rate=FF_RATE,
    )


def load_pretrained(trans_encoder, conditioned, cond_dim=0):
    ckpt = torch.load(PRETRAINED_TRANS, map_location="cpu")["trans"]
    if not conditioned:
        trans_encoder.load_state_dict(ckpt, strict=True)
        print("loaded pretrained transformer, strict=True (unconditioned baseline)")
        return
    old_w = ckpt.pop("trans_base.cond_emb.weight")  # (embed_dim, 512)
    old_b = ckpt.pop("trans_base.cond_emb.bias")    # (embed_dim,)
    missing, unexpected = trans_encoder.load_state_dict(ckpt, strict=False)
    assert set(missing) == {"trans_base.cond_emb.weight", "trans_base.cond_emb.bias"}, missing
    assert not unexpected, unexpected
    with torch.no_grad():
        trans_encoder.trans_base.cond_emb.weight[:, :CLIP_DIM_BASE].copy_(old_w)
        trans_encoder.trans_base.cond_emb.bias.copy_(old_b)
    print(f"loaded pretrained transformer, warm-started cond_emb "
          f"(512->{CLIP_DIM_BASE + cond_dim} cols, rest strict)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditioned", action="store_true")
    ap.add_argument("--cond-mode", choices=["rel", "abs", "rel_prefix", "full"], default="rel",
                    help="frame the goal is expressed in; see module docstring. "
                         "Ignored when --conditioned is not set.")
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--print-iter", type=int, default=100)
    ap.add_argument("--goal-aug", type=float, default=0.0,
                    help="probability of truncation augmentation (see ProbeMotionDataset). "
                         "0.5 is the first thing to try for the RESULTS section 9 problem.")
    ap.add_argument("--save-every", type=int, default=0,
                    help="also write net_iter<N>.pth every N iters. Worth setting for long "
                         "runs -- a crash at iter 19000 of 20000 otherwise loses everything.")
    ap.add_argument("--out-name", type=str, default=None)
    ap.add_argument("--tokens-dir", type=str, default=TOKENS_DIR,
                    help="token dir from prepare_probe_data.py. MUST match the tokenizer "
                         "the model will be decoded with -- tokens are codebook-specific.")
    args = ap.parse_args()

    tag = f"conditioned-{args.cond_mode}" if args.conditioned else "unconditioned"
    out_name = args.out_name or tag
    ckpt_dir = os.path.join(OUT_DIR, "checkpoints", out_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(os.path.join(args.tokens_dir, "train.pkl"), "rb") as f:
        train_manifest = pickle.load(f)
    print(f"train manifest: {len(train_manifest)} clips")

    cond_mean, cond_std = compute_cond_norm(train_manifest, args.cond_mode)
    print(f"cond_mode={args.cond_mode} mean={cond_mean} std={cond_std}")

    dataset = ProbeMotionDataset(train_manifest, args.cond_mode, cond_mean, cond_std,
                                 goal_aug=args.goal_aug)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    def cycle(it):
        while True:
            for x in it:
                yield x

    loader_iter = cycle(loader)

    clip_model, _ = clip.load("ViT-B/32", device=DEVICE, jit=False)
    clip.model.convert_weights(clip_model)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    cond_dim = COND_EXTRA_DIMS[args.cond_mode] if args.conditioned else 0
    clip_dim = CLIP_DIM_BASE + cond_dim
    trans_encoder = build_transformer(clip_dim)
    load_pretrained(trans_encoder, args.conditioned, cond_dim)
    trans_encoder.train()
    trans_encoder.to(DEVICE)

    optimizer = utils_model.initial_optim("all", args.lr, 1e-6, trans_encoder, "adamw")
    loss_ce = nn.CrossEntropyLoss()

    avg_loss, right_num, nb_sample = 0.0, 0, 0
    for nb_iter in range(1, args.iters + 1):
        text, m_tokens, m_tokens_len, extra = next(loader_iter)
        m_tokens = m_tokens.to(DEVICE)
        target = m_tokens
        input_index = target[:, :-1]

        mask = torch.bernoulli(PKEEP * torch.ones(input_index.shape, device=DEVICE))
        mask = mask.round().to(dtype=torch.int64)
        r_indices = torch.randint_like(input_index, NB_CODE)
        a_indices = mask * input_index + (1 - mask) * r_indices

        text_tok = clip.tokenize(list(text), truncate=True).to(DEVICE)
        with torch.no_grad():
            feat_clip_text = clip_model.encode_text(text_tok).float()

        if args.conditioned:
            cond = torch.cat([feat_clip_text, extra.to(DEVICE).float()], dim=-1)
        else:
            cond = feat_clip_text

        cls_pred = trans_encoder(a_indices, cond).contiguous()

        bs = m_tokens.shape[0]
        loss_cls = 0.0
        for i in range(bs):
            L = m_tokens_len[i].item() + 1
            loss_cls = loss_cls + loss_ce(cls_pred[i][:L], target[i][:L]) / bs
            probs = torch.softmax(cls_pred[i][:L], dim=-1)
            pred_idx = probs.argmax(dim=-1)
            right_num += (pred_idx == target[i][:L]).sum().item()
        nb_sample += (m_tokens_len + 1).sum().item()

        optimizer.zero_grad()
        loss_cls.backward()
        optimizer.step()

        avg_loss += loss_cls.item()
        if args.save_every and nb_iter % args.save_every == 0:
            torch.save({"trans": trans_encoder.state_dict()},
                       os.path.join(ckpt_dir, f"net_iter{nb_iter:06d}.pth"))
        if nb_iter % args.print_iter == 0:
            acc = right_num * 100 / max(nb_sample, 1)
            print(f"[{tag}] iter {nb_iter}/{args.iters} loss {avg_loss / args.print_iter:.4f} acc {acc:.2f}%", flush=True)
            avg_loss, right_num, nb_sample = 0.0, 0, 0

    ckpt_path = os.path.join(ckpt_dir, "net_final.pth")
    torch.save({"trans": trans_encoder.state_dict()}, ckpt_path)
    with open(os.path.join(ckpt_dir, "norm_stats.json"), "w") as f:
        json.dump({"cond_mean": cond_mean.tolist(), "cond_std": cond_std.tolist(),
                   "cond_mode": args.cond_mode, "conditioned": args.conditioned,
                   "clip_dim": clip_dim, "tokens_dir": args.tokens_dir,
                   "goal_aug": args.goal_aug}, f)
    print(f"saved {ckpt_path}")


if __name__ == "__main__":
    main()
