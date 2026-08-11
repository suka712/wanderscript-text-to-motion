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

### Data prep — done
- Reused HUMANISE's own official `train.txt`/`test.txt` split rather than
  inventing a new one: **16523 train / 3125 test** clips, 0 skipped (all
  clips had >=8 raw frames).
- Per clip: tokens from the FROZEN VQ-VAE (`net.encode`), start pose
  `(x, y, sin(yaw), cos(yaw))` at frame 0 and goal `(x, y)` at the last frame
  actually covered by the tokenized target (see frame-alignment note in
  `scripts/track1/prepare_probe_data.py` docstring — `process_file` drops
  the last raw frame when computing velocities, so goal is read at the
  263-dim sequence's own last index, not the raw clip's), both from
  `humanise_join.compute_track2`'s world-frame track. Text = HUMANISE's own
  per-clip utterance (e.g. "sit on the coffee table that is between the
  bench and the end table").
- Token length distribution: 7-32 tokens/clip, comfortably inside the
  pretrained transformer's 50-motion-token budget (`block_size=51`) — no
  clips needed truncation.
- Output: `wander_data/track1_probe/tokens/{train,test}.pkl`.

### Model design
Per CLAUDE.md 2b: "start pose + goal coordinate, as learned embeddings
concatenated to its input." Implemented as literally as possible with zero
architecture surgery: T2M-GPT's `Text2Motion_Transformer` (imported
unmodified) already prepends one condition token, built by projecting a
`clip_dim`-wide vector through `cond_emb: Linear(clip_dim, embed_dim)`.
- **Unconditioned baseline** (required by the spec): `clip_dim=512`, exactly
  vanilla — CLIP ViT-B/32 text feature only. Loads the pretrained transformer
  checkpoint `strict=True`, full warm start.
- **Conditioned**: `clip_dim=518` = 512 (CLIP text) + 4 (start x,y,sin,cos)
  + 2 (goal x,y). The concatenation happens in feature space, before the
  existing `cond_emb` — no new modules, no sequence-length/block_size
  changes. Start/goal (x,y) are normalized using train-set mean/std (sin/cos
  left as-is, already unit-scale) before concatenation.
- Pretrained-checkpoint compatibility: conditioned `cond_emb`'s input width
  (518) doesn't match the pretrained checkpoint's (512), so it can't be
  strict-loaded. Warm-started instead: pretrained `cond_emb` weight copied
  into the first 512 columns of the wider layer (extra 6 columns keep their
  fresh N(0, 0.02) init); bias copied directly. Everything else in the
  transformer (9 blocks, both trans_base and trans_head, embeddings, output
  head) loads `strict=True`/unchanged in both variants.
- Deliberate deviation from CLAUDE.md 2c's mention of T5 text conditioning:
  used CLIP ViT-B/32 instead, matching what's already integrated and
  pretrained in this exact T2M-GPT codebase/checkpoint. The probe's question
  (does start/goal conditioning work) is orthogonal to text-encoder choice,
  and swapping to T5 would mean losing the pretrained cond_emb warm-start
  and downloading a new frozen encoder over this box's throttled network
  for no benefit to what's being tested.
- Exact architecture (from the pretrained checkpoint's own run.log, not
  generic argparse defaults): `nb_code=512, embed_dim_gpt=1024, block_size=51,
  num_layers=9, n_head_gpt=16, down_t=2`. Training corruption rate `pkeep=0.5`
  carried over unchanged from the pretrained recipe
  ("VQTransformer_corruption05").

### SE(2) placement (probe eval)
New utility (`scripts/track1/se2_utils.py`), reusing validated code wherever
possible: `motion_features.recover_positions` (already-validated inverse
263->joint-positions) gives the generated clip's LOCAL root trajectory,
which starts at local origin at frame 0 by construction of the
representation. `se2_utils` adds only the small missing glue — undoing the
Y-up axis relabel and composing the local (x,y) trajectory onto the fed
world start pose via a rigid rotate+translate.

**Sanity check confirmed during smoke testing**: `start_err = 0.0` exactly,
for both models, on every clip. This is expected and correct, not a result —
placement composes frame 0 onto the fed start pose by construction, so it
can't be anything else regardless of what the model learned. It confirms the
placement code itself is right (a bug here would show up as nonzero
start-error). The real signal is goal-error, compared between the two
models below.

### Training — in progress
Both models finetune from the pretrained transformer checkpoint, 4000
iterations, batch size 64, lr 1e-4, on the full 16523-clip train split.
Launched sequentially (unconditioned first, required-baseline priority, then
conditioned) in the background; this section will be filled in with final
train loss/accuracy as each finishes.

- **Unconditioned: DONE.** 4000/4000 iters, final loss 0.70, token-accuracy
  76.1% (climbed from a loss ~8.9 cold-start on `cond_emb`, as expected for
  finetuning an otherwise-converged pretrained model with one
  freshly-initialized-ish input layer). ~28 min wall-clock on the 3090.
- Conditioned: running now (sequential after unconditioned, same box).

### Bug found + fixed: normalization pitfall, again
First full eval run gave nonsensical goal-errors (~93m mean, both models,
room-scale space is a few meters) — the exact "normalization pitfall" this
project has hit before (check14's docstring, CLAUDE.md 3), just in new code
this time: the frozen VQ-VAE operates on 263-dim vectors normalized by its
OWN checkpoint mean/std, and both `prepare_probe_data.py` (encode) and
`eval_probe.py` (decode) were feeding/reading it raw, un-normalized data.

Isolated with a ground-truth-token round trip (real extracted tokens,
decoded + SE(2)-placed, checked against the same clip's own fed goal — this
should be near-zero if the pipeline is correct, since it's not testing the
transformer at all): un-normalized end-to-end gave ~105m mean error;
denormalizing only the decode side (leaving the still-buggy un-normalized
encode) already dropped it to 0.46m, confirming the decode-side omission was
the dominant bug and pinning down exactly where to fix. Interesting
secondary finding: the encoder is apparently fairly robust to
out-of-distribution input scale (still landed on tokens that decode
close to the true trajectory) — likely because per-frame reconstruction
error (what MPJPE-style metrics measure) is far more forgiving than
cumulative trajectory error (what goal-reaching needs), so a "small-ish"
per-frame bias that looked tolerable in the wrong-normalization H3D MPJPE
canary before (137mm vs 45mm, ~3x) compounds into meters of drift over even
a few dozen frames once integrated. Fixed both sides properly rather than
relying on that robustness. This invalidates the tokens, both trained
checkpoints, and the eval numbers above them — redoing data prep, both
training runs, and eval with the fix in place. (Same class of bug, same
project; adding to the pattern already noted in CLAUDE.md 3.)

### Probe eval — redoing after the fix above
Re-extracting tokens with correct normalization, then retraining both
models from the pretrained checkpoint again (same hyperparameters), then
re-evaluating. Will report final numbers here once done.
