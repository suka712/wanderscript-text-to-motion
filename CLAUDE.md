# CLAUDE.md — Scene-Aware Text-to-Motion Project

Authoritative project document. Read fully every session before acting. This is the
ground truth for goals, architecture, what is validated, and where the project is likely
to fail. Step specs are separate and disposable; this file is not.

A note on epistemic status: several original assumptions turned out wrong (a frozen
tokenizer was assumed sufficient; cross-segment chaining was assumed trivial). Both were
wrong. This document reflects the corrected understanding. Treat the design as correct
but not sacred — components marked SWAPPABLE can change if evidence says so. Components
marked LOAD-BEARING cannot change without re-deciding the whole project. When a
LOAD-BEARING assumption looks false, STOP and surface it, do not work around it.

---

## 1. Goal
Scene-aware text-to-motion for indoor scenes. **General full-body motion INCLUDING scene
interaction — sit, lie, reach-toward. Not navigation only.** A demo that only walks/turns
is a failure. Body-level interaction (22-joint skeleton) is in scope; dexterous hand/finger
manipulation is NOT in scope for V1.

Deliverables:
- Explicit **start position** (x, y, yaw) as a direct user input.
- Explicit **goal** per motion segment (spatial coordinate, not inferred from text).
- **Indefinite chaining** of motion segments with continuous body pose across seams.
- **Local MLLM** planner (no cloud API).
- Open source.

---

## 2. The pipeline (read this before any code)

Two phases: offline TRAINING (produces the models), online INFERENCE (produces motion).
Understand both before touching either.

### 2a. Motion representation (LOAD-BEARING)
- Working skeleton is the **22-joint HumanML3D format**. SMPL-X inputs (HUMANISE) are
  reduced to these 22 body joints; hands and face are discarded. This is sufficient for
  sit/lie/reach at body level and matches every benchmark (HUMANISE, AffordMotion).
- Motion is encoded in the **263-dim HumanML3D feature vector**, which is CANONICALIZED:
  global translation and root orientation are removed; root motion is stored as local
  per-frame velocities. Consequence: the representation is position-invariant. A generated
  clip does not know where in the world it is. This single fact drives most of the design.
- Because of canonicalization, every clip is stored as TWO parallel tracks:
  1. Canonicalized 263-dim — the thing the model generates/reconstructs.
  2. World-frame root trajectory — per-frame absolute (x, y, yaw), Z-up ScanNet frame.
     Used for placement, chaining, goal/start conditioning, and all scene metrics.
- Yaw is always represented as `(sin, cos)`, never a raw scalar (avoids wraparound).

### 2b. Training phase (two stages, STRICT ORDER)

Stage A — **VQ-VAE (motion tokenizer) joint finetune.**
- What it is: the VQ-VAE encodes a motion clip into a sequence of discrete tokens (indices
  into a learned codebook) and decodes tokens back to motion. It is the vocabulary of
  motion the rest of the system speaks in.
- Why finetune (not freeze): the off-the-shelf T2M-GPT VQ-VAE is trained on HumanML3D only
  and reconstructs HUMANISE interaction poorly (lie ~703mm, structurally broken). A
  general-motion paper needs interaction, so the codebook must cover it.
- How: finetune jointly on **HumanML3D + HUMANISE**, balanced sampling (HumanML3D 14.6k
  vs HUMANISE 19.6k — weight so neither dominates; HUMANISE-only would cause catastrophic
  forgetting of general motion).
- Input: 263-dim canonicalized motion. Output: a finetuned codebook + encoder + decoder.
- Success = per-category reconstruction (walk/stand/sit/lie) all acceptable, sit/lie
  brought down from baseline, general motion not regressed. This number is a reportable
  result, not just a gate.

Stage B — **Transformer finetune (conditional continuation).**
- What it is: the autoregressive model (the "GPT" of T2M-GPT) that predicts a sequence of
  motion tokens. In vanilla T2M-GPT it is conditioned on TEXT ONLY.
- What changes here — two additions, both LOAD-BEARING:
  1. **Explicit spatial conditioning.** The transformer receives, as learned embeddings
     concatenated to its input: the **start pose** (x, y, yaw) and the **goal coordinate**
     (x, y). The model is TOLD where to begin and where to end. It does NOT infer location
     from text or from the scene image. (Decision locked: goal is an explicit input, never
     text-only. Inference burden is the enemy.)
  2. **Conditional continuation for chaining.** The transformer is trained to generate a
     segment CONDITIONED ON THE TAIL of the previous segment (prefix = last N tokens / last
     pose of segment k-1). This is what makes chaining actually work — see 2d. This is a
     training-time change, not an inference trick; the model must LEARN to continue from an
     arbitrary ending pose.
- Scene conditioning (DINOv2 features of a scene render) also enters here — but see the
  SWAPPABLE note in section 4; do not build deep dependence on it before it is shown to help.
- STRICT ORDERING: finetuning the VQ-VAE changes the codebook, which invalidates all
  previously extracted tokens. After Stage A you MUST re-extract tokens with the finetuned
  VQ-VAE, THEN train the transformer on those new tokens. Training the transformer on stale
  tokens produces silent garbage. Order is always: finetune VQ-VAE -> re-extract tokens ->
  finetune transformer.

### 2c. Inference phase
1. **Qwen3-VL (local MLLM)** reads the instruction + a scene image and emits a per-segment
   JSON plan: `{action, goal_object_id, goal_coord: [x,y], duration}`. The MLLM does the
   spatial grounding — it decides WHERE the goal is. The motion model never guesses location.
2. For each segment, the **transformer** generates motion tokens, conditioned on: text
   (T5) + start pose + goal coord + (scene features) + prefix from previous segment.
3. **Collision-guided decoding** (optional steering, see 2e) re-ranks candidate tokens
   against the scene occupancy map during generation.
4. **VQ-VAE decoder** turns tokens into canonicalized local motion.
5. **SE(2) rollout** places that local motion into the world by composing it onto the
   segment's start pose, producing world-frame motion.
6. **Chaining**: the end pose of segment k becomes the start pose (and prefix context) of
   segment k+1. Repeat for indefinite length.
7. Output: world-frame motion rendered to mp4 (or driving a robot).

### 2d. Chaining — why it is hard, and what actually works (LOAD-BEARING)
The naive approach (generate each segment independently, then just place segment k+1 where
segment k ended) DOES NOT WORK. Because motion is canonicalized, each segment is generated
as if starting from a neutral pose. Segment k ends mid-stride in a specific body
configuration; segment k+1 begins from canonical neutral. Gluing them makes the body
teleport at the seam — feet slide, limbs snap. Placing the root correctly with SE(2) fixes
position but not body configuration.

The working approach: the transformer generates segment k+1 CONDITIONED ON THE TAIL of
segment k (its last pose/tokens as prefix). The model continues from the actual ending
pose. Plus a short seam blend (4-frame overlap, linear on root, slerp on rotations) to
clean residual discontinuity. The continuation conditioning is the real fix; the blend is
cosmetic polish on top. This is genuine work and was underestimated in the original plan —
treat it as one of the two hardest parts of the project.

### 2e. Collision-guided decoding — what it is and its status (SWAPPABLE, NOT the hill)
At inference, in the AR loop: instead of greedily taking the top next token, take the
top-k, decode each candidate's resulting root movement, check it against the scene
occupancy map, and re-rank to prefer non-colliding, goal-approaching motion. It injects
scene-awareness at decode time without extra training.

It is the most academically novel piece AND the least proven. It can fail two ways:
(1) if the top-k candidates are all similar (e.g. all "walk forward"), there is no evasive
option to promote; (2) greedy per-token re-ranking cannot plan a turn several tokens ahead.
Fallbacks, in order: beam search over a short horizon; segment-level rejection sampling
(generate N candidate segments, keep the lowest-collision one) — rejection sampling is the
guaranteed floor and always improves non-collision.

STRATEGIC NOTE: this is NOT the load-bearing contribution. If it underwhelms, it demotes
to an ablation ("+X% non-collision") and the paper still stands on explicit control +
conditional-continuation chaining + an interaction-capable tokenizer. Do not let the
project's success depend on this component.

---

## 3. What is validated so far (do not redo, do not re-litigate)
From completed verification work (STEP1/STEP1b):
- **Data join:** ID-join across HUMANISE's three sources (pure_motion / align_data /
  contact_motion) is 19,648/19,648 = 100%, at full scale.
- **263-dim conversion:** built by REUSING HumanML3D's own feature extractor (not
  hand-written), so the layout is guaranteed to match what the tokenizer expects. All
  19,648 clips convert cleanly, 0 NaN/exceptions. (Verify the SMPL-X -> 22-joint mapping
  once more at scale — a wrong joint mapping corrupts everything silently.)
- **World-frame track:** reconstructed from pure_motion (SMPL-X translation+orientation) +
  align_data rigid transform. Validated by overlaying trajectories on scene mesh floors
  (150/150 scenes pass). This process caught and fixed a missing `scene_translation`
  offset.
- **Axis convention:** HUMANISE raw joint data is natively **Z-up** (empirically verified;
  the original Y-up assumption was WRONG). World frame = ScanNet Z-up, yaw about Z.
- **Data quality:** 2 NaN-corrupted files exist in the official HumanML3D release — skip/
  handle them.
- **Scene meshes:** all 643 HUMANISE ScanNet meshes load fine (trimesh).

Still outstanding from STEP2 (blocked on shared-GPU availability, not design):
- T2M-GPT baseline FID/R-precision reproduction (calibrates the eval harness).
- Final reconstruction-FID number on HumanML3D.
Run these when the GPU is free; the baseline confirms the harness before any numbers
downstream are trusted.

---

## 4. Components and how locked they are
LOAD-BEARING (changing these = re-deciding the project):
- 22-joint HumanML3D representation with two-track (canonical + world-frame) storage.
- VQ-VAE finetuned (not frozen), joint on HumanML3D + HUMANISE.
- Transformer with explicit spatial conditioning (start + goal) AND conditional
  continuation for chaining.
- Strict training order: VQ-VAE -> re-extract tokens -> transformer.

SWAPPABLE (change freely if evidence says so — surface it, don't agonize):
- **Scene representation.** A top-down BEV render + DINOv2 was inherited from another
  paper. DINOv2 is trained on natural images, so top-down floorplans are out of its
  distribution and may give weak features. Nothing here is locked. Treat the scene-feature
  encoder as an experiment; if the grounding probe shows it is not carrying weight, replace
  it (different view, different encoder, or drop the image and rely on explicit goal coords).
- Collision-guided decoding (see 2e) and its fallbacks.
- MLLM choice (Qwen3-VL 8B) — swappable if planning quality is poor.
- Seam-blend details.

NOT excluded by default: earlier drafts listed RRT*, retrieval DB, and heightmap as
"excluded." That was leftover cruft from a diluted context, not a real decision. Nothing
is excluded on principle. Add what helps; justify it.

---

## 5. Critical failure points — know these, back off early
Ordered by risk. For each, the plan is to test cheaply and pivot early, not to discover
failure after building everything on top.

1. **Chaining continuity (2d).** Conditional continuation must actually produce smooth
   pose transitions across segments. Hard, underestimated. If continuation training does
   not give clean seams, that is a core-mechanism problem — surface immediately.
2. **Goal grounding.** Even with explicit goal coords fed in, the model must LEARN to
   reach them — and canonicalized targets give a weak gradient toward absolute position.
   DE-RISK EARLY with a grounding probe (section 6) BEFORE building the full transformer
   stage. If it cannot reach fed coordinates at toy scale, pivot (e.g. predict root
   trajectory explicitly first, then generate motion along it).
3. **VQ-VAE joint finetune balance.** May forget general motion (HumanML3D) or underfit
   interaction (HUMANISE). It is the foundation everything else sits on — validate
   per-category reconstruction before proceeding to the transformer.
4. **Collision-guided decoding (2e).** May not steer. Not the hill — has fallbacks and
   demotes to an ablation. Low strategic risk by design.
5. **Shared GPU.** Two training stages now (~2x compute). Jobs can be killed by contention.
   Resolve scheduling and confirm the 5090 fallback; this gates the timeline more than any
   single algorithm.

Meta: the remaining hard problems (1, 2) are genuine research bets with no clean answer,
unlike the validation work so far. Expect iteration. Expect more work than the naive plan
implied.

---

## 6. Build order
1. ~~Verify plumbing~~ DONE (STEP1/1b).
2. **Baseline calibration** (STEP2): reproduce T2M-GPT FID/R-precision; get recon-FID.
   Blocked only on GPU. Confirms the eval harness before trusting any downstream number.
3. **VQ-VAE joint finetune** (balanced HumanML3D + HUMANISE). Validate per-category
   reconstruction. Re-extract tokens after.
4. **Grounding probe** (cheap, BEFORE the full transformer build): a minimal
   goal-conditioned model — does generated root trajectory actually reach a fed goal
   coordinate? Pass -> proceed. Fail -> pivot the grounding approach now, not at day 100.
5. **Transformer finetune**: explicit start+goal conditioning + conditional continuation,
   on the new tokens. Single segment reaching the goal from the correct start.
6. **Chaining**: conditional continuation across segments + SE(2) rollout + seam blend.
   Two segments connect with continuous body pose.
7. **Collision-guided decoding** (+ rejection-sampling floor). Non-collision improves.
8. **Qwen JSON** wired end-to-end -> ScanNet demo mp4 showing scene interaction.

## 7. Done criteria
1. Baseline matches T2M-GPT paper (harness trusted).
2. VQ-VAE reconstructs interaction (sit/lie) acceptably after joint finetune.
3. Single segment reaches an explicit goal from an explicit start pose.
4. Multi-segment instruction -> correct per-segment motion including interaction, with
   continuous body pose across seams (no teleport).
5. Watchable mp4 of scene interaction in a ScanNet room.

## 8. Environment / logistics
- Data: `/media/user/2tb/motion_data/` (HUMANISE, HumanML3D, ScanNet meshes). Data,
  checkpoints, renders live on disk and are gitignored — never committed.
- T2M-GPT base code: `/home/user/Khiem-ssh/T2M-GPT/` — kept SEPARATE from the working
  `wander` repo; wander imports/points at it, does not fork it in.
- Hardware: 4090 (~25 GB) primary, 5090 fallback (likely needed — two training stages).
- Benchmarks: PSMo + AffordMotion (reported HUMANISE numbers; PSMo has no public code, so
  state test-protocol differences honestly). SceMoS is related work only — it reports on
  TRUMANS, a different dataset; no numeric comparison.
