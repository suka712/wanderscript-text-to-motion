# CLAUDE.md — Scene-Aware Text-to-Motion

Ground truth for goals, architecture, and conventions. Numbers are in `docs/RESULTS.md`.
Running state is in `docs/IN_FLIGHT.md` — read it first for current work.

**Status 2026-09-12.** Steps 1–13 done. From-scratch scene-grounded VQ-VAE training in progress
(breaks the redundancy trap — see IN_FLIGHT). Best interaction model:
`~/wander_data/step11/checkpoints/action` (`cond_mode=full_action`). Next after tokenizer:
re-extract tokens → retrain transformer → benchmark comparison + FID.

Treat the design as correct but not sacred. Components marked **SWAPPABLE** can change with
evidence. Components marked **LOAD-BEARING** cannot change without re-deciding the project —
STOP and surface it. When a result says a LOAD-BEARING component fails, verify the measurement
before believing it (a 90° eval bug already invalidated one track's conclusion).

---

## 1. Goal

Scene-aware text-to-motion for indoor scenes — **general full-body motion INCLUDING interaction
(sit, lie, reach-toward), not navigation only.** 22-joint skeleton; hand/finger manipulation out
of scope for V1. Explicit start position, explicit goal per segment, indefinite chaining, local
MLLM planner, open source.

---

## 2. Architecture

### 2a. Motion representation (LOAD-BEARING)
- **22-joint HumanML3D**, 263-dim canonicalized feature vector (position-invariant: global
  translation/orientation removed, root motion as local per-frame velocities).
- Every clip stored as TWO tracks: (1) canonical 263-dim, (2) world-frame (x, y, yaw) Z-up.
- Yaw is always `(sin, cos)`, never scalar.
- Collision scored against the **0.9m TALL-obstacle raster** (0.12m counts sitting as collision).
- **Two yaw conventions, 90° apart (LOAD-BEARING).** Canonical frame 0 faces +Z in Y-up =
  −Y in Z-up = yaw −π/2; `compute_track2` defines yaw 0 = +X. Placement rotates by
  `yaw0 + π/2` (`se2_utils.se2_place`). Getting this wrong rotates everything 90° and is
  invisible to start-error checks.
- **Give the model geometry in ITS frame** — start-relative, heading-aligned. Absolute world
  coords fail (0.515 m vs 0.164 m relative).

### 2b. Training cascade (STRICT ORDER)
1. **VQ-VAE** — encodes motion → discrete codebook tokens → decodes back. Finetuned jointly
   on HumanML3D + HUMANISE (interaction reconstruction improved, RESULTS §3).
2. **Re-extract tokens** — changing the VQ-VAE invalidates all tokens. Training the transformer
   on stale tokens produces silent garbage. Always: VQ-VAE → re-extract → transformer.
3. **Transformer** — AR model predicting token sequences, conditioned on:
   - Text (CLIP)
   - Goal coordinate in segment's start frame (explicit, never text-inferred)
   - Continuation prefix: tail body config of previous segment (66-d joint positions,
     root-relative, heading-canonicalized — must be ON-MANIFOLD, never blended/smoothed)
   - 4-way action one-hot: walk/sit/stand/lie (`cond_mode=full_action`, LOAD-BEARING for
     interaction — the (x,y) goal can't distinguish "sit" from "walk to the seat")
   - Scene occupancy crop (in agent's frame; binary raster, not DINOv2)

### 2c. Inference
1. **Qwen3-VL** (local MLLM) → per-segment `{action, goal_coord}` plan
2. **Transformer** generates tokens (optionally collision-guided: per-segment best-of-N)
3. **VQ-VAE** decodes to canonical motion
4. **SE(2) placement** onto segment start pose → world-frame motion
5. **Chaining**: end pose of segment k → start of k+1 (continuous body configuration)

### 2d. Chaining rules (LOAD-BEARING)
- Condition on ending **body configuration** (66-d joints), NOT tokens (wrong frame).
- Prefix must be **on-manifold** (VQ-VAE-reconstructed). Off-manifold noise collapses it.
- Seam blend is **not cosmetic** — ~70mm residual is VQ-VAE reconstruction error; the blend hides it.
- `rollout(..., reorient=True)` rotates walk segments to face their goal (fixes moonwalking).

---

## 3. Components

**LOAD-BEARING:** 22-joint HumanML3D two-track · VQ-VAE finetuned jointly · Transformer with
explicit goal + continuation + action conditioning · Strict VQ-VAE → tokens → transformer order.

**SWAPPABLE:** Collision-guided decoding (`collision_guided.py`) · Global path planning
(`grid_planner.py`) · MLLM choice (Qwen3-VL 8B) · Foot-contact deskating
(`foot_contact.deskate`) · Scene representation details.

---

## 4. Open risks
1. **Measurement validity** — highest-frequency failure mode. Five silent convention bugs so far.
   **MANDATORY: every pipeline gets an oracle control** before reading model numbers.
2. **Shared GPU** — the 4090 is shared. Check `nvidia-smi --query-compute-apps` first.
3. **Single seed** — all results are point estimates, no run-to-run variance measured.
4. **No comparison to published work** — generation FID unreproduced after 5 attempts.

---

## 5. Environment
- **ntx (4090, 25 GB, shared)**: Data `/media/user/2tb/motion_data/`, T2M-GPT `/home/user/Khiem/T2M-GPT`.
- **dsx (3090, 24 GB)**: Data `~/wander_data/motion_data/`, T2M-GPT `~/Khiem/T2M-GPT`.
  Conda env `afford` was raw-copied — use `~/anaconda3/envs/afford/bin/python -m pip`.
  Source `~/.wander_env` explicitly (non-interactive shells miss it).
- Both on Tailscale: `train-4090` / `train-3090`.
- T2M-GPT base code: `/home/user/Khiem-ssh/T2M-GPT/` — separate, not forked.
- **Network**: per-TCP-flow rate limit. Use `aria2c -x8 -s8` for large files.
- **Benchmarks**: PSMo + AffordMotion (HUMANISE). SceMoS (CVPR 2026, TRUMANS) — related work,
  not directly comparable. SceMoS architecture inspired our scene-grounded tokenizer work.
