# STEP 2 — T2M-GPT Baseline Repro on HumanML3D (+ harness calibration)

Read CLAUDE.md (with the STEP1b patch applied) first. Step 1 plumbing is green for
locomotion. This step reproduces the T2M-GPT baseline on HumanML3D and — critically —
calibrates the evaluation harness so all later numbers are trustworthy.

This is largely running existing code, not building. The real deliverable is a **trusted,
paper-matching eval harness**, not new architecture.

Done-criterion for the project's step 1 ("numbers match T2M-GPT paper") lives here.

---

## Why calibration is the point
STEP1b reported H3D reconstruction MPJPE = 137.3mm (root-relative, local). That is
higher than a correctly-working T2M-GPT VQ-VAE should give. Suspect a harness convention
issue (normalization, joint subset, global-vs-local scale) rather than true recon quality.
Until the harness is calibrated against the paper, no MPJPE number — including the
sit/lie figures — can be taken at face value. MPJPE conventions vary; **FID is what the
paper reports and is convention-independent.** Calibrate on FID first.

## Task 1 — Reproduce the paper's reported metrics
Using the official T2M-GPT repo + pretrained checkpoints already on disk
(`pretrained/VQVAE/net_best_fid.pth`, plus the GPT/transformer checkpoint — download it
if not present):
- Run T2M-GPT's own evaluation on HumanML3D test set.
- Report **FID** and **R-precision (Top-1/2/3)** for text-to-motion generation.
- Compare against the T2M-GPT paper's reported numbers. State the tolerance/delta.

**Match target:** within reasonable reproduction variance of the paper. If FID and
R-precision land near the paper, the harness is trusted. If they don't, the harness or
checkpoints are wrong — fix before proceeding; do not build on an untrusted harness.

## Task 2 — Reconstruction FID calibration (resolves the 137mm question)
- Compute **reconstruction FID on HumanML3D** (encode→decode through the frozen VQ-VAE,
  FID vs. ground-truth test motions).
- Compare against T2M-GPT's reported reconstruction FID.
- Also render 2–3 H3D walk reconstructions visually.

Interpretation:
- Recon FID matches paper + visuals look good → the 137mm MPJPE was a metric-convention
  artifact. Retroactively, the STEP1b sit (293mm) figure is likely inflated by the same
  factor and sit may be more usable than it looked. **Lie stays broken regardless**
  (structural visual break, not a metric artifact).
- Recon FID also off → the harness/checkpoint is genuinely wrong; fix it. This would mean
  the STEP1b canary ran on a miscalibrated harness and those numbers need re-reading.

Report the recon FID, the paper's number, and a one-line verdict on what it implies for
the STEP1b sit/lie figures.

## Task 3 — Add the locomotion filter to HUMANISE preprocessing
Per the CLAUDE.md patch (MVP = locomotion):
- Add a label-based filter to the HUMANISE preprocessor output: keep walk / stand / turn,
  drop sit / lie.
- Log surviving clip count and the per-category breakdown.
- This does not affect Step 2 (H3D-only) but prepares Step 3's training set. Do it now
  while the preprocessor is fresh.

---

## Deliverable
- Report: paper vs. reproduced FID + R-precision (Task 1); recon FID + calibration verdict
  (Task 2); locomotion-filtered HUMANISE clip counts (Task 3).
- Scripts committed. Respect `.gitignore`.

**Verdict gates:**
- Task 1 metrics match paper (within variance) → project step-1 done-criterion met, harness
  trusted.
- Task 2 tells us whether STEP1b's MPJPE numbers were inflated → informs whether sit is
  recoverable for a possible V1.5, but does NOT change the locomotion-MVP decision.
- On green, Step 3 (scene conditioning: start-pose + goal + DINOv2 into the transformer,
  finetune on locomotion-filtered HUMANISE) unlocks — that spec comes next.

Do not start Step 3 until the harness is calibrated and trusted. A wrong harness silently
corrupts every downstream number.
