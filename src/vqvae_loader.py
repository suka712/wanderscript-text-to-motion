"""Loads the frozen, officially pretrained T2M-GPT VQ-VAE (net_best_fid.pth)
for inference-only use in the reconstruction canary (Task 2). No model code is
modified; args reproduce exactly the config recorded in the checkpoint's own
pretrained/VQVAE/run.log (dataname=t2m, quantizer=bailando at train time, but
the released weights' buffer names match the 'ema_reset' quantizer class in
the current repo -- both implementations expose the same `codebook` buffer /
encode-decode interface, so this is an inference-only compatibility choice,
not a retrain).

Checkpoint provenance:
  Source:  official T2M-GPT repo download script
           /home/user/Khiem-ssh/T2M-GPT/dataset/prepare/download_model.sh
           -> gdown 1LaOvwypF-jM2Axnq5dc-Iuvv3w_G-WDE -> VQTrans_pretrained.zip
  Repo commit: b1446f1 (Mael-zys/T2M-GPT, "Update t2m extractor")
  Local path: /home/user/Khiem-ssh/T2M-GPT/pretrained/VQVAE/net_best_fid.pth
"""
import sys
import types

import torch

T2M_GPT_ROOT = "/home/user/Khiem-ssh/T2M-GPT"
if T2M_GPT_ROOT not in sys.path:
    sys.path.insert(0, T2M_GPT_ROOT)

import models.vqvae as vqvae  # noqa: E402

DEFAULT_CKPT = f"{T2M_GPT_ROOT}/pretrained/VQVAE/net_best_fid.pth"


def load_vqvae(ckpt_path: str = DEFAULT_CKPT, device: str = "cuda"):
    args = types.SimpleNamespace(
        dataname="t2m",
        quantizer="ema_reset",
        nb_code=512,
        code_dim=512,
        output_emb_width=512,
        down_t=2,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        vq_act="relu",
        vq_norm=None,
        mu=0.99,
    )
    net = vqvae.HumanVQVAE(
        args, args.nb_code, args.code_dim, args.output_emb_width,
        args.down_t, args.stride_t, args.width, args.depth,
        args.dilation_growth_rate, args.vq_act, args.vq_norm,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    net.load_state_dict(ckpt["net"], strict=True)
    net.eval()
    net.to(device)
    return net
