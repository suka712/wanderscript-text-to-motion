# STEP 1b — Extend Plumbing (build the missing pieces before Step 2)

Read CLAUDE.md and STEP1_REPORT.md first. Step 1 correctly hit its STOP condition.
Nothing in that report threatens the architecture — every blocker is "build the
necessary thing," not "the design is wrong." This step builds those pieces. It is still
Step 1: **do not begin Step 2 (T2M-GPT baseline) until the HUMANISE reconstruction canary
passes.**

The two open decisions from the report are NOT two tasks — they are outputs of one
preprocessor. Build them together; they share the same axis fix and the same ID-join.

---

## LOCKED DECISION — axis convention (resolve this first, everything depends on it)
World frame = **ScanNet Z-up**. Yaw = rotation about world Z.
Rationale: collision/contact metrics run against the ScanNet mesh, so the mesh frame is
the source of truth. SMPL-X mocap is Y-up; convert Y-up → Z-up exactly once, when the
`align_data` rigid transform is applied, inside the preprocessor. No axis ambiguity is
allowed to survive downstream.

---

## Task 0 — Unblock Check 3 (fetch, not a decision)
- Install torch in the working environment.
- Download the pretrained **T2M-GPT VQ-VAE checkpoint** from the official release into a
  known path. Record the path and source URL/commit in the report.
- No model changes. Just make Check 3 runnable.

## Task 1 — Unified HUMANISE preprocessor (produces BOTH tracks)
Build ONE module that ingests HUMANISE raw pkls and emits both tracks per clip.

**Track 2 — world-frame root (load-bearing, build first):**
- Source: `pure_motion/*.pkl` SMPL-X per-frame translation + orientation →
  apply `align_data_release/*.pkl` rigid transform (yaw + translation) → convert Y-up→Z-up
  → per-frame absolute `(x, y, yaw)` in ScanNet world frame.
- Yaw stored as `(sin, cos)` per CLAUDE.md convention, not a scalar.
- **ID-join at full scale:** join across the three data sources (contact_motion /
  pure_motion / align_data) for ALL clips. Log coverage count and enumerate every miss.
  The single spot-check from Step 1 is not sufficient.

**Track 1 — canonicalized 263-dim (for the frozen VQ-VAE):**
- Do NOT hand-write the 263 feature math. **Reuse HumanML3D's own `motion_representation`
  extraction pipeline.** HUMANISE's `(T,22,3)` joint positions are the input that
  pipeline expects.
- Write only a thin **input adapter**: joint-order reindex (verify HUMANISE's 22-joint
  order matches HumanML3D's SMPL joint order — likely but CONFIRM against HumanML3D's
  joint spec), unit scaling, up-axis fix. Then run HumanML3D's extractor unmodified.
- Writing the 263 layout from scratch risks a silent layout mismatch that poisons every
  token. Reuse eliminates that risk. If the adapter's output shape/layout does not match
  what the VQ-VAE expects, STOP and report the mismatch — do not patch the 263 math.

**Two traps — handle explicitly:**
- Track 1 canonicalizes (strips global position) — that is correct, that is its job.
  Track 2 holds the global frame. Do NOT try to make the two frame-match per-frame; they
  reconcile only at the start pose, via SE(2), at rollout time (later step).
- HUMANISE's separate **8192-point contact data is scene-contact** for the later contact
  metric. It is NOT the foot-contact bits inside the 263 vector (those are derived from a
  velocity threshold by HumanML3D's extractor). Keep them separate. Do not write
  scene-contact into the 263 foot-contact slots.

## Task 2 — Reconstruction canary (two stages, order matters)
Run through the FROZEN VQ-VAE:
1. **Original H3D first** — baseline. Proves the checkpoint + eval harness work end to
   end. If this is bad, the problem is setup, not the data bet.
2. **Converted HUMANISE second** — the actual test of the frozen-codebook bet.
- Report MPJPE and/or reconstruction FID for both, plus one rendered sample each.

**Watch item (the one contingency that could change the architecture):** if converted
HUMANISE reconstructs *badly* (sitting / complex scene-contact motions may fall outside
HumanML3D's codebook distribution), the frozen-decoder assumption breaks and we would
consider finetuning the VQ-VAE decoder — which changes CLAUDE.md's "VQ-VAE frozen."
Do NOT preempt this. Report the numbers and STOP for a design decision if HUMANISE
reconstruction is poor. Do not silently start finetuning.

## Task 3 — Validate the world-frame track (this is also the axis-fix test)
- Overlay reconstructed Track-2 trajectories on each scene's ScanNet mesh floor bounds.
  Confirm trajectories land ON the floor — not floating, not sunk.
- This overlay IS the Y-up/Z-up correctness test. If Z-up conversion is wrong,
  trajectories won't sit on the mesh. Report pass/fail with a saved overlay image on a
  few scenes.

---

## Deliverable
Updated report covering Tasks 0–3, scripts committed under `scripts/verify/` and the
preprocessor under an appropriate module path, and this git repo actually initialized +
committed (Step 1 committed nothing). Respect the `.gitignore` — no data, no checkpoints,
no renders in git.

**Green / unlocked verdict requires ALL of:**
- World-frame track built, ID-join run at scale with logged coverage, floor overlay passes.
- 263 converter reuses HumanML3D extraction, layout confirmed to match the VQ-VAE.
- Reconstruction canary: H3D passes AND converted HUMANISE reconstructs acceptably.

If HUMANISE reconstruction is poor → STOP, report numbers, await decision on VQ-VAE
decoder finetuning. Otherwise, Step 2 (T2M-GPT baseline repro) is unlocked.
