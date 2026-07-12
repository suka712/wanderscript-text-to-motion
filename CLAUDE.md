# CLAUDE.md — Scene-Aware Text-to-Motion (T2M) Research Project

Read this every session. It is the durable context for the repo. Specs for individual
steps are separate and disposable; this file is the ground truth for goals, stack,
paths, and the decisions that must not be silently violated.

---

## Goal
Scene-aware text-to-motion for humanoid robot navigation in indoor scenes.
- Explicit **start position** (x, y, yaw) as a direct user input.
- **Indefinite chaining** of motion segments via token continuation.
- **Local MLLM** (no cloud API).
- Open source.

## One-line architecture claim
An MLLM produces a spatial multi-goal plan; a **frozen** tokenizer generates local
(canonicalized) motion; an SE(2) rollout with **collision-guided decoding** chains
segments indefinitely from an explicit start pose. Scene grounding happens at
**decode time**, not by retraining a scene-aware vocabulary.

Differentiated from:
- **PSMo** (ACM MM 2025): no diffusion, no RRT*, local MLLM, explicit start pos.
- **SceMoS** (CVPR 2026): explicit control + cross-segment chaining + training-free
  scene guidance, instead of their retrained geometry-grounded tokenizer.

---

## Stack (locked)
- **Qwen3-VL 8B** (local) — reads BEV image + instruction → JSON plan. Per segment:
  `{action, goal_object_id, goal_coord: [x,y], duration, trans_flag}`.
  The MLLM does spatial grounding (picks goal from BEV). The motion transformer does NOT.
- **T2M-GPT** — base. Two weights from ONE checkpoint:
  - **VQ-VAE** (motion tokenizer: encoder + codebook + decoder) — **FROZEN**.
  - **GPT / transformer** (text→token predictor) — **FINETUNED**. All new conditioning
    is added here.
- **DINOv2** (frozen) — encodes BEV RGB → patch features for scene conditioning.
- **T5** (frozen) — text encoder (already in T2M-GPT).
- **VQ-VAE frozen — hardened, not just an initial choice (confirmed STEP1b):** the
  lie/sit reconstruction break (see MVP scope below) is a codebook-coverage limit, not
  a decoder-weights problem. Decoder-only finetuning can't fix it; fixing it would need
  an encoder+codebook+decoder retrain, which discards the "reuse T2M-GPT tokens" core
  of the design. Since navigation doesn't need those motions, VQ-VAE stays frozen.
  Revisit only if V2 requires scene interaction (sitting, lying).
- **Start position (x, y, yaw)** — anchoring frame for the SE(2) rollout AND a
  conditioning input to the transformer.
- **Token chaining** — end pose of segment k → start pose of segment k+1.

## Deliberately excluded (do NOT add without explicit approval)
- RRT* — MLLM goal waypoints replace it.
- Retrieval database.
- Heightmap conditioning. (V2 only, if time allows.)

---

## THE LOAD-BEARING DECISION — read before touching anything
The frozen VQ-VAE uses the **HumanML3D 263-dim canonicalized** representation:
global translation and orientation are stripped; root motion is stored as local
velocities. This is position-invariant — good for plain T2M, but it means the motion
vocabulary has **no notion of absolute position or scene geometry**.

Therefore the pipeline keeps **two parallel tracks** for every sequence:
1. **Canonicalized 263-dim** — fed to the frozen VQ-VAE (tokenization).
2. **World-frame root trajectory** — per-frame absolute (x, y, yaw). Used for
   placement (SE(2) rollout), chaining, start-position, and collision/contact metrics.

If track (2) does not exist in preprocessing, **STOP** — nothing downstream works.
This is the single silent point of failure. Verify it before writing pipeline code.

Conventions:
- Yaw is represented **everywhere** as `(sin, cos)`, never a raw scalar (avoids ±π
  wraparound). Convert at the boundaries only.
- Start pose **anchors the world frame** of the rollout. It is not merely a soft
  embedding the transformer may ignore (it will ignore it — canonicalized targets
  carry no gradient toward absolute position).

**Axis convention (verified in STEP1b):** World frame = ScanNet **Z-up**. Yaw = rotation
about world Z. HUMANISE raw joint data is natively Z-up — no Y-up→Z-up conversion is
needed or should be reintroduced. (A `scene_translation` offset in the placement
transform was found missing during STEP1b validation and fixed — keep it.)

---

## BEV rendering — two aligned rasters
Rendered from ScanNet meshes, same camera / same extent / aligned to world frame so
rollout coordinates map directly to pixels:
1. **RGB** → DINOv2 → semantic scene conditioning.
2. **Binary walkable/occupied occupancy raster** → collision-guided decoding + metrics.
Do not ship only the RGB. The occupancy raster is required for the core contribution.

---

## Scene-guided decoding (the research contribution)
Frozen VQ-VAE is scene-blind by design. Scene-awareness is injected at AR decode time:
during generation, take top-k next tokens, decode each candidate's root displacement,
project against the BEV occupancy raster, penalize collision + reward goal progress,
re-rank. Training-free. Touches nothing frozen.

## Segment chaining / continuity
- SE(2) rollout: decoded local motion composed onto the anchoring start pose.
- Chaining: end pose of segment k becomes start pose of segment k+1.
- Seam smoothing: **4-frame overlap**, linear blend on root, slerp on joint rotations.
  (Frozen VQ-VAE will not give smooth boundaries for free.)

---

## MVP scope — locomotion only (STEP1b finding)
The frozen T2M-GPT VQ-VAE reconstructs HUMANISE motions unevenly:
- walk ~112mm (≈ H3D baseline, fine)
- stand-up ~192mm (1.4x)
- sit ~293mm (2.1x)
- lie ~703mm (5.1x, structurally broken)

This is a codebook-coverage limit: HumanML3D's discrete codes don't span supine/seated
poses. The MVP application is **indoor robot navigation** — walk / turn / stand / stop.
Sit and lie are NOT needed for navigation.

Decision: **MVP HUMANISE training is filtered to locomotion (walk/stand/turn).**
Sit/lie are a documented limitation → **V2**, alongside heightmap conditioning.

## Data — `/media/user/2tb/motion_data/`
- **HUMANISE** — scene-grounded (ScanNet scenes) + text. SMPL-X. Must be converted to
  HumanML3D 263-dim for the frozen VQ-VAE. Primary scene-aware training set. **Filtered
  to locomotion (walk/stand/turn) for MVP training — see MVP scope above.**
- **HumanML3D** — 14.6K sequences, rich text, no scene. For baseline repro.
- **ScanNet meshes** — BEV rendering (RGB + occupancy).
- Preprocessing stats already computed — verify contents, don't assume.

## Benchmarks & metrics
- Numeric comparison: **T2M-GPT** (HumanML3D FID / R-precision),
  **PSMo** + **AffordMotion** (HUMANISE contact / non-collision, reported numbers).
- **SceMoS is NOT a numeric benchmark** — it reports on TRUMANS, incomparable dataset.
  Cite as related work, contrast qualitatively only.
- **Start-position error** is an **ablation**, not a benchmark (no baseline takes
  explicit start pos, so there is nothing to compare against).

## Hardware
- Primary: 4090, 25 GB.
- 5090 only if needed.

## Timeline
- ~140 days, solo. Critical path is scene conditioning on the frozen tokenizer
  (step 3 below), NOT the MLLM wiring. Do not under-budget it.

---

## Build order
1. **Verify plumbing** (do first, ~small). See STEP1 spec.
2. **T2M-GPT baseline** on HumanML3D → match paper FID / R-precision.
3. **Scene conditioning**: start-pose + goal + DINOv2 into the transformer,
   finetune on HUMANISE. Single-segment, correct start position. (Hardest step.)
4. **SE(2) rollout + seam blending** → two segments connect cleanly.
5. **Collision-guided decoding** → non-collision metric improves.
6. **Qwen JSON** wired end-to-end → ScanNet demo mp4.

## Done criteria
1. Baseline numbers match T2M-GPT paper.
2. Single segment generates from the correct start position.
3. Multi-segment instruction → correct per-segment motion.
4. Two segments connect cleanly (no teleport / foot-skate at seam).
5. Watchable mp4 in a ScanNet room.

## Assumptions that must hold (fail any → stop and escalate, do not work around)
- World-frame root trajectory survived preprocessing.
- HUMANISE→HumanML3D 263-dim conversion exists and is faithful.
- Frozen VQ-VAE reconstructs HUMANISE motion acceptably (codebook coverage).
- BEV pipeline can emit an aligned occupancy raster, not just RGB.
