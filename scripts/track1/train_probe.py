#!/usr/bin/env python3
"""
Grounding probe training (docs/track_1/001_grounding_probe.md, "Training" +
"REQUIRED baseline").

Finetunes T2M-GPT's pretrained transformer (frozen VQ-VAE tokens as targets,
frozen CLIP ViT-B/32 for text) with --conditioned to add start-pose + goal
conditioning, or without it to train the required no-goal-conditioning
baseline. The transformer architecture itself (models.t2m_trans.Text2Motion_
Transformer, reused unmodified from T2M-GPT) is IDENTICAL between the two
runs -- the only difference is the width of the vector fed into its existing
cond_emb projection: clip_dim=512 (CLIP text only) for the baseline, or
clip_dim=518 (CLIP text [512] + start x,y,sin,cos [4] + goal x,y [2]) for the
conditioned run. This matches CLAUDE.md 2b's "learned embeddings concatenated
to its input" literally, without any model-architecture surgery: the
concatenation happens in feature space, before the existing cond_emb Linear,
so the pretrained transformer blocks/head need no changes at all.

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
T2M_GPT_ROOT = os.environ.get("WANDER_T2M_GPT_ROOT", "/home/user/Khiem-ssh/T2M-GPT")
if T2M_GPT_ROOT not in sys.path:
    sys.path.insert(0, T2M_GPT_ROOT)

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


class ProbeMotionDataset(Dataset):
    def __init__(self, manifest, xy_mean, xy_std):
        self.data = manifest
        self.xy_mean = xy_mean
        self.xy_std = xy_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
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

        start = d["start"].copy()
        start[:2] = (start[:2] - self.xy_mean) / self.xy_std
        goal = (d["goal"] - self.xy_mean) / self.xy_std

        return d["text"], m_tokens, m_tokens_len, start.astype(np.float32), goal.astype(np.float32)


def compute_xy_norm(manifest):
    xy = np.concatenate([
        np.stack([d["start"][:2] for d in manifest]),
        np.stack([d["goal"] for d in manifest]),
    ])
    return xy.mean(0).astype(np.float32), (xy.std(0) + 1e-6).astype(np.float32)


def build_transformer(clip_dim):
    return trans.Text2Motion_Transformer(
        num_vq=NB_CODE, embed_dim=EMBED_DIM_GPT, clip_dim=clip_dim, block_size=BLOCK_SIZE,
        num_layers=NUM_LAYERS, n_head=N_HEAD_GPT, drop_out_rate=DROP_OUT_RATE, fc_rate=FF_RATE,
    )


def load_pretrained(trans_encoder, conditioned):
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
    print("loaded pretrained transformer, warm-started cond_emb (512->518 cols, rest strict)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditioned", action="store_true")
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--print-iter", type=int, default=100)
    ap.add_argument("--out-name", type=str, default=None)
    args = ap.parse_args()

    tag = "conditioned" if args.conditioned else "unconditioned"
    out_name = args.out_name or tag
    ckpt_dir = os.path.join(OUT_DIR, "checkpoints", out_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(os.path.join(TOKENS_DIR, "train.pkl"), "rb") as f:
        train_manifest = pickle.load(f)
    print(f"train manifest: {len(train_manifest)} clips")

    xy_mean, xy_std = compute_xy_norm(train_manifest)
    print("xy_mean", xy_mean, "xy_std", xy_std)

    dataset = ProbeMotionDataset(train_manifest, xy_mean, xy_std)
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

    clip_dim = CLIP_DIM_BASE + 6 if args.conditioned else CLIP_DIM_BASE
    trans_encoder = build_transformer(clip_dim)
    load_pretrained(trans_encoder, args.conditioned)
    trans_encoder.train()
    trans_encoder.to(DEVICE)

    optimizer = utils_model.initial_optim("all", args.lr, 1e-6, trans_encoder, "adamw")
    loss_ce = nn.CrossEntropyLoss()

    avg_loss, right_num, nb_sample = 0.0, 0, 0
    for nb_iter in range(1, args.iters + 1):
        text, m_tokens, m_tokens_len, start, goal = next(loader_iter)
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
            cond = torch.cat([feat_clip_text, start.to(DEVICE).float(), goal.to(DEVICE).float()], dim=-1)
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
        if nb_iter % args.print_iter == 0:
            acc = right_num * 100 / max(nb_sample, 1)
            print(f"[{tag}] iter {nb_iter}/{args.iters} loss {avg_loss / args.print_iter:.4f} acc {acc:.2f}%", flush=True)
            avg_loss, right_num, nb_sample = 0.0, 0, 0

    ckpt_path = os.path.join(ckpt_dir, "net_final.pth")
    torch.save({"trans": trans_encoder.state_dict()}, ckpt_path)
    with open(os.path.join(ckpt_dir, "norm_stats.json"), "w") as f:
        json.dump({"xy_mean": xy_mean.tolist(), "xy_std": xy_std.tolist(),
                    "conditioned": args.conditioned, "clip_dim": clip_dim}, f)
    print(f"saved {ckpt_path}")


if __name__ == "__main__":
    main()
