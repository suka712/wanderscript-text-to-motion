# In flight — read this if you are picking up cold

Volatile state not in RESULTS.md: what is running, where things live, next action.
**Update or delete when its work lands.**

Last updated: 2026-09-12.

---

## From-scratch scene-grounded VQ-VAE — DONE (2026-09-12)

Trained a heightmap-conditioned VQ-VAE from the BASE T2M-GPT checkpoint on three balanced
sources (H3D/HUMANISE/TRUMANS). Breaks the **redundancy trap**: codebook learned WITH the
heightmap from the start, so it never encodes absolute contact height.

**Results:** follow_ratio 0.98 (sit) / 0.92 (lie) at convergence — the decoder reads the
heightmap. MPJPE: h3d 63, walk 42, sit 71, lie 99 mm. Full cascade completed:

| Step | Checkpoint |
|---|---|
| VQ-VAE (30k iters) | `scene_tokenizer_v2/checkpoints/scene_vqvae_scratch/net_iter030000.pth` |
| Re-extracted tokens (21,832 clips) | `scene_tokenizer_v2/tokens/{train,test}.pkl` |
| Transformer (20k iters, 90.8% acc) | `scene_tokenizer_v2/checkpoints/action_scratch/net_final.pth` |
| Demo (scene0380, SAT&STOOD) | `scene_tokenizer_v2/demo/mesh_interaction_scene0380_00_0.mp4` |

**Next:** evaluate whether sit height now tracks furniture at generation (the whole point).
Run `eval_seat_height.py` or `eval_contact_demo.py` on the new model to measure
corr(seat_height, pelvis_height) — should exceed the §12 plateau if the trap is truly broken.

**Reproduce cascade:**
```bash
# 1. VQ-VAE
train_scene_vqvae.py --from-scratch --base-vqvae <base> --consist-weight 0.5 --height-aug 0.6 --total-iter 30000 --batch-size 192
# 2. Re-extract
reextract_tokens.py --scene-vqvae <ckpt> --base-vqvae <base> --src-tokens ~/wander_data/trumans_combined_tokens --out <out>
# 3. Transformer
train_probe.py --conditioned --cond-mode full_action --iters 20000 --goal-aug 0.5 --walk-prefix-aug 0.5 --tokens-dir <tokens> --out-name action_scratch
# 4. Demo
render_mesh_demo.py --mode interaction --ckpt <transformer_dir> --vqvae-ckpt <base> --scene-vqvae <scene_vqvae_ckpt> --out <dir>
```

**Wait-loop gotcha:** `while pgrep -f foo.py; do sleep 60; done` never exits — pgrep matches
its own command line. Use `pgrep -f "[f]oo.py"` or poll a log sentinel.

---

## Key paths

| path | what |
|---|---|
| `~/wander_data/step11/checkpoints/action` | Best interaction model (`full_action`) |
| `~/wander_data/scene_tokenizer_v2/checkpoints/action_scratch` | Scene-grounded interaction model |
| `~/wander_data/scene_tokenizer_v2/checkpoints/scene_vqvae_scratch/net_iter030000.pth` | From-scratch SceneVQVAE |
| `~/wander_data/scene_tokenizer_v2/tokens/` | Tokens from from-scratch SceneVQVAE |
| `~/wander_data/step10/checkpoints/goalaug` | Best navigation model |
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
