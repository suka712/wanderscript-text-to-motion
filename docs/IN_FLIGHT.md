# In flight — read this if you are picking up cold

Volatile state not in RESULTS.md: what is running, where things live, next action.
**Update or delete when its work lands.**

Last updated: 2026-09-12.

---

## Current: from-scratch scene-grounded VQ-VAE (ntx 4090)

Training a heightmap-conditioned VQ-VAE from the BASE T2M-GPT checkpoint (not the finetuned
one) on three balanced sources. This breaks the **redundancy trap**: when the codebook is
pre-trained without a heightmap, it encodes contact height, and adding the heightmap later
(frozen decoder, unfrozen encoder, or transformer-side) is always ignored at generation because
the token already carries the answer. Training from scratch means the codebook learns WITH the
heightmap and never encodes absolute contact height.

**Config:** `~/wander_data/scene_tokenizer_v2/checkpoints/scene_vqvae_scratch/`
- 30k iters, batch 192, consist_weight 0.5, height_aug 0.6, seed 42
- Three-source balanced: H3D 34% / HUMANISE 33% / TRUMANS 33%
- Base VQ-VAE: `pretrained/VQVAE/net_best_fid.pth`
- Script: `train_scene_vqvae.py --from-scratch`
- Loader: `BalancedThreeSourceHMLoader` (`src/scene_joint_dataset.py`)

**Monitor:**
```bash
pgrep -af "[t]rain_scene_vqvae"
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader
tail -5 ~/wander_data/scene_tokenizer_v2/checkpoints/scene_vqvae_scratch/heartbeat.log
```

**What to watch:** shift-invariance should exceed 0.72 (§12's plateau at consist_weight=0.25).
If seat-height correlation drops at convergence, the codebook is still encoding height.

**Wait-loop gotcha:** `while pgrep -f foo.py; do sleep 60; done` never exits — pgrep matches
its own command line. Use `pgrep -f "[f]oo.py"` or poll a log sentinel.

---

## Cascade after VQ-VAE converges

1. **Re-extract tokens**: `reextract_tokens.py --scene-vqvae <best_ckpt> --base-vqvae <base>`
2. **Retrain transformer**: `train_probe.py --cond-mode full_action` on new tokens
3. **Demo**: `render_mesh_demo.py --mode interaction`

---

## Key paths

| path | what |
|---|---|
| `~/wander_data/step11/checkpoints/action` | Best interaction model (`full_action`) |
| `~/wander_data/step10/checkpoints/goalaug` | Best navigation model |
| `~/wander_data/pathb/checkpoints/pv_2000` | Best scene-aware navigation (`full`, greedy avoids obstacles) |
| `/media/user/2tb/motion_data/track2_checkpoints/.../net_iter020000.pth` | Finetuned VQ-VAE |
| `~/wander_data/trumans_combined_tokens/train.pkl` | Combined manifest (21,832 clips) |
| `/media/user/2tb/motion_data/TRUMANS_processed/trumans_263_cache` | TRUMANS 263 cache |
| `~/wander_data/trumans_heightmap_cache/` | TRUMANS heightmap cache (6200 clips) |
| `~/wander_data/motion_data/HUMANISE_heightmap_cache/` | HUMANISE heightmap cache (19,648 clips) |
| `~/wander_data/bev_cache/` | BEV renders (643 scenes) |
| `~/wander_data/bev_tall_cache/` | 0.9m tall-obstacle rasters |

---

## Step-11 reproduce (interaction model)

```bash
train_probe.py --conditioned --cond-mode full_action --iters 20000 --lr 1e-4 \
  --goal-aug 0.5 --walk-prefix-aug 0.5 \
  --tokens-dir ~/wander_data/step10/tokens --out-name action
```

Eval: `eval_sit_capability.py --ckpts .../action --vqvae-ckpt <ft-vqvae> --n 60 --actions sit "stand up" --prefix-mode walk`

---

## Not worth redoing

- **Target-instance exclusion** — blocked (no ScanNet instance seg); connected-component proxy failed (331/400 merged with walls).
- **Scene ablation on single HUMANISE segments** — saturated; only chained rollouts resolve it.
- **Randomising the goal while keeping motion** — teaches the model to ignore the goal. Use truncation augmentation.
- **Heading conditioning** (`full_action_head`) — heading target is redundant with goal+prefix on GT, gets no gradient, ignored at convergence. Use `reorient=True` at inference instead.
- **Sit orientation** — a DATA limitation (HUMANISE approach≈sit-facing, median 5°). Must come from the PLANNER, not the motion model.

---

## Known gaps

1. No comparison to published work (generation FID unreproduced after 5 attempts)
2. Single seed everywhere — point estimates, no variance
3. Composed-chain interaction yield ~50%, single-seed
4. Sit orientation unresolved for general furniture (works for curated freestanding pieces)
