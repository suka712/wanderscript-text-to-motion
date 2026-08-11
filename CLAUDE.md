# CLAUDE.md — Scene-Aware Text-to-Motion Project

Authoritative project document. Read fully every session before acting. This is the
ground truth for goals, architecture, what is validated, and where the project is likely
to fail. Step specs are separate and disposable; this file is not.

**Status as of 2026-08-12.** Data pipeline validated; tokenizer finetune done and it
works (2b, section 3); grounding probe done and it **passes** (2f). Next action is
build-order step 5: re-extract tokens with the finetuned VQ-VAE. The only remaining
research risk is chaining. Full reconciliation of the two diagnostic tracks:
`docs/RECONCILIATION_aug12.md` — read it if you are picking this up cold.

A note on epistemic status: several original assumptions turned out wrong (a frozen
tokenizer was assumed sufficient; cross-segment chaining was assumed trivial). Both were
wrong. This document reflects the corrected understanding. Treat the design as correct
but not sacred — components marked SWAPPABLE can change if evidence says so. Components
marked LOAD-BEARING cannot change without re-deciding the whole project. When a
LOAD-BEARING assumption looks false, STOP and surface it, do not work around it.

The inverse also applies, and it has now happened once: **when a result says a
LOAD-BEARING component fails, verify the measurement before believing it.** Track 1
reported that goal grounding was impossible and recommended re-architecting the project
around it. The finding was an artifact of a 90° bug in its own eval. The cost of checking
was one afternoon; the cost of acting on it would have been the whole design. Numbers that
would change the architecture get an oracle control first — no exceptions (section 5 #4).

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
- **Two yaw conventions exist and they differ by 90°. LOAD-BEARING.** HumanML3D's
  `process_file` canonicalizes frame 0 to face **+Z in its Y-up frame**; under our
  `zup_to_yup_hml` relabel that is **−Y in the Z-up world frame, i.e. world yaw −π/2**.
  But `humanise_join.compute_track2` defines **world yaw 0 = facing +X**. So placing a
  canonicalized clip at a world start pose rotates by **`yaw0 + π/2`**, and expressing a
  world point in the model's frame uses the inverse (`se2_utils.world_to_local_xy`).
  Getting this wrong rotates every trajectory 90° and is invisible to a start-error
  check — it already invalidated one track's conclusion. See RECONCILIATION_aug12.md.
- **Corollary, and the main design lesson so far: give the model geometry in the frame it
  generates in.** Goals, and later the chaining prefix, must be start-relative and
  heading-aligned. Feeding absolute world coordinates forces the model to compute
  `R(yaw₀)ᵀ(goal − start)` — a bilinear op — and it will simply fail to. Measured:
  goal-error 0.515 m absolute-frame vs 0.164 m relative-frame, identical everything else.

### 2b. Training phase (two stages, STRICT ORDER)

Stage A — **VQ-VAE (motion tokenizer) joint finetune.**
- What it is: the VQ-VAE encodes a motion clip into a sequence of discrete tokens (indices
  into a learned codebook) and decodes tokens back to motion. It is the vocabulary of
  motion the rest of the system speaks in.
- Why finetune (not freeze): the off-the-shelf T2M-GPT VQ-VAE is trained on HumanML3D only
  and reconstructs HUMANISE interaction worse than locomotion. **Corrected number** (the
  original canary used a normalization mismatched to this checkpoint — same class of bug
  found and fixed in the Step 2 FID work; re-run with the correct evaluator-consistent
  normalization): H3D baseline MPJPE 45.3mm; HUMANISE walk 47.9mm (1.06x), stand-up 67.6mm
  (1.49x), sit 72.6mm (1.60x), **lie 139.8mm (3.09x)**. Lie is still the worst category by
  a real margin, and a 10-clip visual spot-check found no catastrophic outliers (65-246mm,
  uniform moderate degradation, not a broken subset) — but "3.09x, moderately worse" is a
  materially weaker claim than the original "5.1x, structurally broken." A general-motion
  paper still needs interaction, so the codebook still needs to cover it, but the magnitude
  of the original justification should not be quoted as-is. See docs/STEP1_plumbing.md
  Check 3 for the full before/after and the visual evidence.
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
     concatenated to its input, the **goal coordinate expressed in the segment's own start
     frame** (start-relative, heading-aligned). The model is TOLD where to end. It does NOT
     infer location from text or from the scene image. (Decision locked: goal is an
     explicit input, never text-only. Inference burden is the enemy.)
     **VALIDATED — see 2f.** Note what is NOT fed: the absolute start pose. In the start
     frame it is (0,0) at heading 0 by construction, so it carries no information; it
     enters at inference only, as the SE(2) placement. Feeding absolute world coordinates
     instead does not work — measured, see 2a and RECONCILIATION_aug12.md.
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
   + **goal coord expressed in that segment's start frame** (see 2b.1 and 2f) + (scene
   features) + prefix from previous segment. The start pose is not a model input; it is
   applied in step 5.
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

### 2f. Grounding — VALIDATED (2026-08-12), was the #2 project risk
The grounding probe (build-order step 4) has been run and **passed**. Frozen tokenizer,
single segment, relative-frame goal conditioning, 200 held-out HUMANISE clips:

| | goal-error | corr(commanded, achieved) |
|---|---|---|
| NULL (stay at start) | 0.627 m | — |
| unconditioned | 0.490 m | +0.07, +0.62 |
| conditioned, absolute frame | 0.515 m | −0.12, +0.60 |
| **conditioned, relative frame** | **0.164 m** | **+0.90, +0.97** |
| ORACLE (ground-truth tokens) | 0.124 m | — |

The model lands within 4 cm of the tokenizer's own reconstruction floor. Holds across
displacement scale (0.07 m error at <0.25 m commanded, 0.51 m at >2 m) with the
model/oracle ratio flat at 1.1–1.4 — **what limits goal accuracy is tokenizer
reconstruction, not goal-following.** That is the argument for taking Stage A's finetune.

Do NOT overclaim this. It is single-segment, no chaining, no scene features, and every
fed goal is in-distribution (displacement std ~0.35/0.73 m). It retires risk #2 in
section 5. It says nothing about risk #1 (chaining).

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

- **Reconstruction-FID calibration (STEP2 Task 2):** done. FID 0.066 ± .001 vs paper's
  0.070 ± .001, R@1-3 all within variance — harness and checkpoint confirmed trustworthy.
- **Reconstruction canary, corrected normalization:** the frozen VQ-VAE's own checkpoint
  has a specific expected mean/std (`checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/`)
  that differs from the raw H3D data-prep `Mean.npy`/`Std.npy` used in the original Step 1
  canary. Using the wrong one inflates error substantially for BOTH MPJPE and FID (H3D
  baseline MPJPE alone dropped 137.3mm -> 45.3mm when corrected). See the Stage A note in
  section 2b above for the corrected per-category numbers.

From the two diagnostic tracks (`docs/RECONCILIATION_aug12.md`, 2026-08-12):
- **Tokenizer joint finetune works.** HUMANISE lie 136.9 -> 96.3 mm, sit 69.6 -> 47.6,
  stand 67.0 -> 53.0, walk 50.2 -> 34.1; H3D held-out flat 56.11 -> 56.2. The sit/lie gap
  was real codebook coverage, NOT the upstream SMPL-X -> 22-joint step. Final checkpoint:
  `track2_checkpoints/track2_joint_finetune_run1/net_iter020000.pth` on the 4090.
- **Grounding works** with relative-frame goal conditioning — see 2f.
- **CORRECTION — the H3D baseline of 45.3 mm was LEAKED and must not be quoted.** check14
  sampled from all of `new_joint_vecs`, including clips the frozen model trained on. The
  honest held-out figure is **56.11 mm**; the H3D-lie control is **117.5 mm** (n=11, high
  variance), not 90.1. Any "did general motion regress" check compares against 56.11.
- **`QuantizeEMAReset` does not survive `load_state_dict`.** It keeps `init`, `code_sum`,
  `code_count` as plain Python attributes, not registered buffers, so a `strict=True` load
  restores the codebook tensor but leaves `init=False` — the first `train()` forward then
  overwrites all 512 codes from one batch. Anyone resuming/finetuning this VQ-VAE must
  seed the EMA accumulators from the loaded codebook first
  (`prepare_quantizer_for_finetune`).

Bug ledger — four silent frame/normalization bugs so far, all the same shape (a convention
mismatch that degrades a number without crashing anything). Read this before adding a new
measurement; see failure point #4 in section 5:
1. Y-up vs Z-up for HUMANISE raw joints (STEP1) — caught.
2. Wrong mean/std in the MPJPE canary; 137 mm read as real, actually 45 mm (STEP2) — caught.
3. The same normalization error again, in the Track 1 probe's encode/decode — caught by
   that track, but only after it had produced a full set of published numbers.
4. **90° SE(2) yaw-convention error** (2a) — NOT caught by that track; it inverted the
   track's conclusion and was found in orchestrator review.

Still outstanding from STEP2 (not blocking anything):
- T2M-GPT generation FID/R-precision reproduction (Task 1) — root-caused to a self-inflicted
  timeout, not GPU contention; see `docs/old_docs_aug8/STEP2_baseline_calibration.md`
  before re-attempting. Reconstruction FID is calibrated and that is what the tracks needed.

---

## 4. Components and how locked they are
LOAD-BEARING (changing these = re-deciding the project):
- 22-joint HumanML3D representation with two-track (canonical + world-frame) storage.
- VQ-VAE finetuned (not frozen), joint on HumanML3D + HUMANISE.
- Transformer with explicit spatial conditioning (goal in the start frame — see 2a, 2f)
  AND conditional continuation for chaining.
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
2. ~~**Goal grounding.**~~ **RETIRED 2026-08-12 — the probe passed, see 2f.** Goal-error
   0.164 m vs a 0.124 m tokenizer floor and a 0.490 m unconditioned baseline. The original
   framing below was right about the mechanism but the fix is cheap: express the goal in
   the start frame. Kept for the record; do not re-litigate, do not pivot to
   trajectory-first.
   <details><summary>original text</summary>
   Even with explicit goal coords fed in, the model must LEARN to
   reach them — and canonicalized targets give a weak gradient toward absolute position.
   DE-RISK EARLY with a grounding probe (section 6) BEFORE building the full transformer
   stage. If it cannot reach fed coordinates at toy scale, pivot (e.g. predict root
   trajectory explicitly first, then generate motion along it).
   </details>
3. ~~**VQ-VAE joint finetune balance.**~~ **RETIRED 2026-08-12 — Track 2 passed.** Joint
   finetune at 1:1 sampling, lr 2e-5: HUMANISE-lie 136.9 -> 96.3 mm, H3D held-out flat,
   no category regressed. No forgetting observed at this balance.
4. **NEW — measurement validity.** This project has now shipped four silent frame/
   normalization bugs (see section 3), two of which were only caught after they had
   changed a conclusion. Each was missed because the check used could not detect it —
   start-error cannot see a rotation; per-frame MPJPE cannot see cumulative drift.
   **MANDATORY from here: every pipeline that emits a number gets an oracle control** —
   push ground truth through the identical path and confirm near-perfect output BEFORE
   reading any model number off it. If the oracle is not small compared to the effect you
   are measuring, you have no measurement. This is now the highest-frequency failure mode
   in the project, ahead of any algorithmic risk.
5. **Collision-guided decoding (2e).** May not steer. Not the hill — has fallbacks and
   demotes to an ablation. Low strategic risk by design.
6. **Shared GPU.** The 4090 is shared; the 3090 (added Aug 2026) is not. Two training
   stages (~2x compute). See section 8.

Meta: with grounding retired, **chaining (#1) is now the only genuine research bet left**,
and everything else is engineering plus discipline about measurement (#4). That is a much
better position than the Aug-9 plan assumed. Expect iteration on chaining specifically.

---

## 6. Build order
1. ~~Verify plumbing~~ DONE (STEP1/1b).
2. ~~Baseline calibration~~ DONE for reconstruction (FID 0.066 vs paper 0.070, harness
   trusted). Generation-FID reproduction is still open but blocks nothing — see section 3.
3. ~~**VQ-VAE joint finetune**~~ DONE (Track 2, merged). Per-category reconstruction
   validated held-out. **Tokens have NOT been re-extracted yet — that is step 4.**
4. ~~**Grounding probe**~~ DONE and PASSED (Track 1, corrected — see 2f and
   `docs/RECONCILIATION_aug12.md`). Ran with the FROZEN tokenizer, which was the right
   call: it isolated grounding from tokenizer quality.

**-> YOU ARE HERE. Next: step 5.**

5. **Re-extract tokens with the finetuned VQ-VAE.** Mandatory before any transformer
   training (2b strict ordering) — the finetuned codebook invalidates every token in
   `track1_probe/tokens/`. Cheap; do not skip; do not train on the probe's tokens.
6. **Transformer finetune** on the new tokens: relative-frame goal conditioning (proven,
   2f) + conditional continuation (unproven, the real work). Single segment reaching the
   goal from the correct start. Sanity check: rerun the probe eval on the new tokenizer
   and confirm goal-error tracks the new (lower) oracle floor.
7. **Chaining**: conditional continuation across segments + SE(2) rollout + seam blend.
   Two segments connect with continuous body pose. **This is the last real research risk.**
   Express the prefix in segment k+1's own start frame — same lesson as 2f.
8. **Collision-guided decoding** (+ rejection-sampling floor). Non-collision improves.
9. **Qwen JSON** wired end-to-end -> ScanNet demo mp4 showing scene interaction.

## 7. Done criteria
1. ~~Baseline matches T2M-GPT paper~~ DONE for reconstruction; harness trusted.
2. ~~VQ-VAE reconstructs interaction (sit/lie) acceptably after joint finetune~~ DONE.
3. ~~Single segment reaches an explicit goal from an explicit start pose~~ DONE at probe
   scale with the frozen tokenizer (2f). Re-confirm on the finetuned tokenizer after
   token re-extraction.
4. Multi-segment instruction -> correct per-segment motion including interaction, with
   continuous body pose across seams (no teleport).
5. Watchable mp4 of scene interaction in a ScanNet room.

## 8. Environment / logistics
- Data: `/media/user/2tb/motion_data/` (HUMANISE, HumanML3D, ScanNet meshes). Data,
  checkpoints, renders live on disk and are gitignored — never committed.
- T2M-GPT base code: `/home/user/Khiem-ssh/T2M-GPT/` — kept SEPARATE from the working
  `wander` repo; wander imports/points at it, does not fork it in.
- Hardware: two boxes, both on Tailscale, both reachable from the Mac via ssh
  (`train-4090` = `ntx`, `train-3090` = `dsx`).
  - **4090** (`ntx`, 25 GB) — **shared with another user's job**, confirmed directly (an
    eval stalled for hours at 100% util with zero progress, traced to a foreign ~16-19 GB
    process). Check `nvidia-smi --query-compute-apps` before blaming your code. Light jobs
    (VQ-VAE eval) get cycles under contention; autoregressive generation may not. Ran
    Track 2. Paths: `/media/user/2tb/motion_data`, `/home/user/Khiem/T2M-GPT`.
  - **3090** (`dsx`, 24 GB, added Aug 2026) — **not shared**. Ran Track 1. Env vars in
    `~/.wander_env` (source explicitly; non-interactive shells do not pick it up). Conda
    env `afford` was raw-copied from the 4090, so its console-script shebangs point at the
    4090's paths — bare `pip` is broken, use `~/anaconda3/envs/afford/bin/python -m pip`.
    Paths: `~/wander_data/motion_data`, `~/Khiem/T2M-GPT`, repo at
    `~/Khiem/wanderscript-text-to-motion`.
  - **Network**: both boxes sit behind an upstream **per-TCP-flow rate limit (~10-30 KB/s
    per connection)** — not fixable locally, confirmed on both. `git clone`/`curl` crawl
    while browsers feel fine (many parallel connections). For anything large, rsync from
    the other box over the LAN/Tailscale direct path (~10 MB/s) instead of downloading.
    This repo is tiny, so `git pull`/`push` is unaffected in practice.
- Benchmarks: PSMo + AffordMotion (reported HUMANISE numbers; PSMo has no public code, so
  state test-protocol differences honestly). SceMoS is related work only — it reports on
  TRUMANS, a different dataset; no numeric comparison.
