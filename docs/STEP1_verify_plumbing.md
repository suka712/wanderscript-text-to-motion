# STEP 1 — Verify Plumbing (do this before any pipeline code)

Read CLAUDE.md first. This step writes NO model code. It is four data checks. Their job
is to confirm the load-bearing assumptions before we build on them. If any check fails,
STOP and report — do not work around it.

Data root: `/media/user/2tb/motion_data/`

---

## Check 1 — World-frame root trajectory exists
The most important check. The pipeline needs per-frame absolute `(x, y, yaw)` for every
sequence, separate from the HumanML3D 263-dim canonicalized tensor.

- Inspect the preprocessed HUMANISE outputs and the precomputed preprocessing stats.
- Determine whether an absolute world-frame root trajectory is stored anywhere.
- If only canonicalized 263-dim tensors exist: the world frame was discarded.
  **STOP and report.** Do not attempt to reconstruct it silently — the canonicalized
  repr integrates local velocities and drifts; recovering a faithful absolute frame is
  a design decision, not a quick fix.

Report: where the world-frame root lives (path + tensor shape + coordinate convention:
units, up-axis, yaw zero-direction), or that it is absent.

## Check 2 — HUMANISE → HumanML3D 263-dim conversion
- Confirm HUMANISE motion has been converted to the exact 263-dim HumanML3D format the
  T2M-GPT VQ-VAE expects (not raw SMPL-X).
- Verify feature layout matches HumanML3D (root velocities, rotations, joint positions,
  velocities, foot contacts — same ordering/dims as T2M-GPT expects).

Report: conversion location, output shape, and a confirmation the layout matches
T2M-GPT's expected input (or the specific mismatch).

## Check 3 — Reconstruction canary (frozen VQ-VAE on HUMANISE)
- Load the T2M-GPT VQ-VAE checkpoint (frozen).
- Take a few converted HUMANISE sequences (mix of walk / sit / stand / turn).
- Run encode → decode. Compute reconstruction error (MPJPE and/or reconstruction FID
  if the eval harness is available). Also spot-render one for visual sanity.

Interpretation:
- Clean reconstruction → frozen-codebook bet holds. Proceed.
- Poor reconstruction → HUMANISE has motions outside HumanML3D's codebook coverage.
  **Report before proceeding** — this may force finetuning the VQ-VAE decoder, which
  changes the architecture.

Report: reconstruction metric numbers + subjective note on the rendered sample.

## Check 4 — BEV occupancy raster is achievable
- Confirm ScanNet meshes for the HUMANISE scenes are present and loadable.
- Prototype rendering ONE scene to two aligned outputs:
  - RGB BEV (for DINOv2).
  - Binary walkable/occupied raster.
- Confirm both share the same camera/extent and that a world-frame `(x, y)` maps to a
  known pixel (state the mapping explicitly).

Report: renderer used, the world→pixel mapping, and a saved example of both rasters.

---

## Deliverable for Step 1
A short written report answering all four checks, plus any scripts used committed under
`scripts/verify/`. No pipeline/model code yet. End with a clear verdict:

- ALL PASS → ready for Step 2 (T2M-GPT baseline repro).
- Any FAIL → name which assumption broke and stop for a design decision.

Do not begin Step 2 until Check 1 passes. Checks 3 and 4 failing are recoverable but
must be surfaced, not silently patched.
