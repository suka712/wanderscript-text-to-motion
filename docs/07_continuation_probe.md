# 07 — Conditional continuation probe

**Status: DONE. The mechanism works.** Conditioning a segment on the previous segment's
ending body configuration halves seam error and lands at the tokenizer's own floor. This
was the project's last untested load-bearing mechanism (CLAUDE.md 2d, risk #1).

It also produced a design constraint for step 8 that was not obvious in advance: **the
prefix pose must be an on-manifold pose.** See "Robustness" below.

## Result

One seam, ground-truth previous segment, 300 held-out clips. Seam error = mean per-joint
distance between the previous segment's ending local pose and the generated segment's first
local pose. Root and heading are excluded — SE(2) placement matches those by construction,
so they cannot reveal teleporting.

| | seam err | goal err | token acc |
|---|---|---|---|
| no-prefix (the naive approach 2d predicts fails) | 155.8 mm | 0.0707 m | 81.7% |
| **continuation, exact prefix** | **71.2 mm** | 0.0573 m | 98.6% |
| **continuation, reconstructed prefix** (realistic) | **86.0 mm** | 0.1073 m | — |
| ORACLE (ground-truth tokens) | 70.9 mm | 0.0544 m | — |
| canonicalization floor (no VQ-VAE) | 21.6 mm | — | — |

With an exact prefix the model lands **within 0.3 mm of the oracle** — the conditioning is
doing everything available to it, and nothing is left on the table. Goal error is unaffected,
so continuation did not cost the grounding from `04_grounding.md`.

**The remaining seam error is the tokenizer, not the conditioning.** The oracle sits at
70.9 mm against a 21.6 mm canonicalization floor, so ~70 mm of discontinuity is VQ-VAE
reconstruction of the first frame and is not addressable by better conditioning. This is
what the 4-frame seam blend in 2d exists to hide, and it is now a measured quantity rather
than an assumption — the blend is load-bearing for visual quality, not cosmetic.

## Robustness — the constraint this probe found

In real chaining the prefix pose is not exact: segment *k*'s ending pose reaches *k+1*
through the decoder, carrying reconstruction error. Two ways of injecting that error give
opposite answers, and the difference is the finding:

| prefix perturbation | seam err | goal err |
|---|---|---|
| none | 71.2 mm | 0.057 m |
| VQ-VAE round trip (structured, ~70 mm, anatomically plausible) | 86.0 mm | 0.107 m |
| iid Gaussian, σ = 25 mm | 331.3 mm | 0.473 m |
| iid Gaussian, σ = 50 / 100 / 200 mm | 330.0 / 329.3 / 329.0 mm | ~0.47 m |

Realistic error costs 15 mm of seam and doubles goal error — degradation, but the mechanism
still beats the naive baseline by 1.8×. **iid noise at a fifth the magnitude is
catastrophic**, and it is flat from 25 mm to 200 mm, which is the signature of falling off
the data manifold rather than degrading: iid per-coordinate noise breaks bone lengths and
produces an anatomically impossible pose, and the model has never seen one.

**Design constraint for step 8: whatever pose is handed to segment *k+1* must be a real
decoded pose.** A blended, interpolated, or smoothed seam pose is off-manifold and will
behave like the noise rows, not the reconstruction row. Concretely: if a seam blend is
applied for visual continuity, the *blended* pose must not be fed back as the next segment's
conditioning — feed the decoded one.

## Method notes

- **What is conditioned on, and in which frame.** Motion is canonicalized per segment, so
  A's tokens and B's tokens describe different frames and conditioning on A's raw tokens is
  ill-posed. The body configuration at the seam — root-relative, heading-canonicalized joint
  positions (66-d) — is frame-independent, and that is the signal. It is expressed in **B's**
  frame, not A's. This is not oracle knowledge: at inference, segment *k+1*'s canonical frame
  is *defined by* the seam pose, so canonicalizing segment *k*'s world-frame ending pose into
  *k+1*'s frame yields exactly this quantity. Conditioning on A's view would hand the model a
  signal in the wrong frame — the mistake that cost this project the original grounding
  verdict.
- **Off-by-one, found by the prep-time check.** `process_file` drops the last frame when
  differencing for velocities, so `A = cm[:t+1]` ends at raw frame *t−1* while B starts at
  *t* — a full frame of motion, ~4 mm of the original residual. A is now `cm[:t+2]`, and is
  deliberately not cropped to a multiple of 4 (it is never tokenized; cropping would move the
  seam).
- **The canonicalization residual is real, not a bug.** Even with the off-by-one fixed, A's
  and B's views of the same physical pose differ by 21.6 mm, because `process_file`
  normalizes per segment: `uniform_skeleton` rescales from each segment's own first frame and
  the floor offset is each segment's own minimum height. For step 8 the floor term is
  chicken-and-egg — B's floor is a minimum over frames that do not exist until B is
  generated. Anything requiring seams below ~20 mm needs the representation changed, not the
  model improved.
- 15,632 train / 2,962 test segment pairs, finetuned tokenizer, `--cond-mode rel_prefix`
  (clip_dim 580 = 512 CLIP + 2 relative goal + 66 seam pose).

## What this does not establish

**This is one seam with a ground-truth previous segment.** It isolates the mechanism, which
is what a probe should do, but it is not chaining:

- **No error accumulation.** Segment *k* is real motion here, not generated. The
  reconstructed-prefix row is the best available proxy, and it degrades gracefully, but
  accumulation across many segments is untested and is the main open question for step 8.
- **Two segments, not indefinite.** Nothing here speaks to drift over a long chain.
- **No seam blend applied.** The ~70 mm tokenizer residual is measured, not yet mitigated.
- **Split at the midpoint** of each clip, so segment boundaries fall mid-motion rather than
  at semantic action boundaries, which is where a real plan would cut.

## Code

`scripts/continuation/prepare_continuation_data.py`, `scripts/continuation/eval_continuation.py`,
and `--cond-mode rel_prefix` in `scripts/track1/train_probe.py`.

```
python scripts/continuation/prepare_continuation_data.py \
    --ckpt <track2_checkpoints>/net_iter020000.pth --out-dir <cont>/tokens
WANDER_TRACK1_PROBE_ROOT=<cont> python scripts/track1/train_probe.py \
    --conditioned --cond-mode rel_prefix --tokens-dir <cont>/tokens --out-name continuation
python scripts/continuation/eval_continuation.py --ckpt-name noprefix continuation \
    --ckpt-root <cont>/checkpoints --tokens-dir <cont>/tokens \
    --vqvae-ckpt <track2_checkpoints>/net_iter020000.pth [--prefix-source recon]
```
