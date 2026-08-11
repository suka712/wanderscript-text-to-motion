# Track 1 — RESULTS

Branch: `track1-grounding` · Box: 3090 (fresh clone) · Updated: 2026-08-11

Progress logged incrementally per the setup/probe specs, not just a final summary.

---

## Setup (000_setup_3090.md) — PASS

### Data paths (on the 3090)
Data was rsynced from the 4090 (`ntx`, reachable via Tailscale at a direct
peer-to-peer link — both boxes turned out to share the same LAN subnet/gateway,
confirmed via `tailscale ping`: 1ms RTT, direct, not DERP-relayed). Sustained
~10.5-11 MB/s, no resume issues, zero rsync errors in the transfer log.

All under `WANDER_MOTION_DATA_ROOT=/home/dsp52026/wander_data/motion_data/`:
| var | path | size |
|---|---|---|
| `WANDER_H3D_ROOT` | `.../motion_data/H3D` | 32G |
| `WANDER_HUMANISE_ROOT` | `.../motion_data/HUMANISE` | 23G |
| `WANDER_SCANNET_ROOT` | `.../motion_data/scannet/scans` | 3.7G |
| (bonus) `HUMANISE_263_cache` | `.../motion_data/HUMANISE_263_cache` | 1.3G |

Env vars are set in `~/.wander_env` (sourced explicitly per-command; not sourced
automatically by non-interactive shells — see gotcha below) and appended to
`~/.bashrc` (works fine there for interactive/AnyDesk sessions).

### Repos + checkpoints
Rather than `git clone` from GitHub (see network gotcha below — this was
initially attempted and was extremely slow), both sibling repos were rsynced
directly from existing checkouts already present on the 4090, over the same
fast LAN path:
- `WANDER_T2M_GPT_ROOT=/home/dsp52026/Khiem/T2M-GPT` (4.7G, from
  `user@ntx:/home/user/Khiem/T2M-GPT/`) — **includes the pretrained checkpoints**,
  so this also satisfies setup doc section 2 (Checkpoints) as a side effect:
  - VQ-VAE: `pretrained/VQVAE/net_best_fid.pth` (78,826,058 bytes, dated 2023-01-05,
    matches the original T2M-GPT release — loads with `strict=True` via
    `src/vqvae_loader.py`).
  - Transformer: `pretrained/VQTransformer_corruption05/net_best_fid.pth`
    (913,971,193 bytes, same provenance).
- `WANDER_MDM_ROOT=/home/dsp52026/Khiem/motion-diffusion-model` (2.1G, from
  `user@ntx:/home/user/jered/T2M_test/motion-diffusion-model/` — despite the
  path name this directory is actually owned by the `user` account we have
  access to, not a different person's data).

First rsync attempt at T2M-GPT partially failed mid-transfer (destination
directory disappeared under it — `mkdir`/`mkstemp` errors, "No such file or
directory" for basic top-level dirs). Root cause not fully pinned down but
plausibly a concurrent local operation on the same path; retried cleanly with
zero errors on the second attempt. Final directory verified complete (4.7G,
all expected subdirs present, checkpoint file sizes match).

### Environment
Used the existing `afford` conda env **copied directly from the 4090 over LAN**
(`/home/user/anaconda3/envs/afford` → `/home/dsp52026/anaconda3/envs/afford`,
6.5G, zero rsync errors) instead of a fresh `conda create` + pip install, to
avoid the network gotcha below entirely (a fresh torch install alone is
1-2GB). Modern conda packages are largely relocatable (self-contained RPATH),
and this worked:
- Python 3.8.20, torch 1.12.0 (CUDA 11.3 build), CUDA available.
- Verified with an actual GPU matmul (not just `torch.cuda.is_available()`):
  ran clean on `NVIDIA GeForce RTX 3090`.
- All `requirements.txt` versions match exactly: numpy 1.24.3, scipy 1.10.1,
  matplotlib 3.7.1, trimesh 3.21.7, smplx 0.1.28, ftfy 6.2.3, regex 2024.11.6,
  tensorboard 2.12.1, torchvision 0.13.0, torchaudio 0.12.0, clip 1.0.
  (`pytorch3d` 0.7.8 also present, as documented — unused, harmless leftover.)
- **Gotcha**: the env's `pip`/`conda`-installed console-script shebangs still
  point at the original `/home/user/anaconda3/envs/afford/bin/python` path
  (from the raw copy), so bare `pip` is broken on this box. Use
  `python -m pip ...` or invoke `python` directly — that's all this project's
  scripts need anyway.

GPU: RTX 3090, 24GB, driver 535.309.01, CUDA 12.2 (cu113 wheels work via
minor-version compatibility, as the setup doc predicted).

### Network gotcha (new, worth recording for the next box)
This box's general internet egress has a **per-TCP-flow rate limit** (~10-30
KB/s per connection), confirmed via a dedicated diagnostic pass:
- Single-connection `curl`/`git`/`wget` to GitHub, an OVH speedtest mirror, and
  a UK ISP mirror were all capped to the same order of magnitude, over both
  HTTP and HTTPS — not GitHub-specific, not a TLS-fingerprint block.
- `aria2c -x8 -s8` (8 parallel connections) got ~10x aggregate throughput on a
  Range-capable target — the smoking gun for a per-flow cap, not a flat
  bandwidth cap.
- BBR vs cubic congestion control made no difference (ruled out bufferbloat).
- Nothing local is misconfigured (checked `tc`, iptables/nftables, proxy env
  vars, NetworkManager, cgroups — all clean); the 4090 (`ntx`) shows the
  identical cap and shares the same subnet/gateway as this box, so it's
  enforced upstream (campus/lab network policy), not fixable from either
  host.
- This is why browsing/streaming (many parallel/QUIC connections) feels
  unthrottled while `git clone`/`curl` crawl — confirmed directly by the user
  via a plain browser download over AnyDesk (~8-15 KB/s on a single-file
  download, matching the CLI numbers exactly).
- **Practical takeaway**: for anything large, check whether a LAN-reachable
  copy already exists on the 4090 first (as done above for the repos and the
  conda env) rather than fighting this cap. `aria2c` is installed and helps
  only for Range-capable single large files, which GitHub's zip/clone
  endpoints are not.
- This project's own repo (`wanderscript-text-to-motion`) is tiny (324K
  `.git`, largest blob ~28KB) since data/checkpoints are gitignored, so normal
  `git pull`/`push` against `origin` (GitHub) is a minor, not a blocking,
  tax — unaffected in practice despite being technically subject to the same
  cap.

### Smoke test — PASS
`scripts/verify/check14_mpjpe_evaluator_norm.py`, evaluator-consistent
normalization, `afford` env, `WANDER_*` env vars set:

| category | 3090 (this run) | expected (docs/REPORT_aug9.md) |
|---|---|---|
| H3D baseline | 45.22mm | ~45.3mm |
| walk | 47.85mm (1.06x) | 47.9mm (1.06x) |
| stand up | 67.60mm (1.49x) | 67.6mm (1.49x) |
| sit | 72.67mm (1.61x) | 72.6mm (1.60x) |
| lie | 139.77mm (3.09x) | 139.8mm (3.09x) |

Matches within RNG/rounding noise across the board.

**Verdict: PASS. The 3090 is at parity with the 4090.** Proceeding to
`001_grounding_probe.md`.

---

## Grounding probe (001_grounding_probe.md)

Not started yet — will log data-prep, smoke numbers, and per-model
goal-error/start-error here incrementally as each stage completes.
