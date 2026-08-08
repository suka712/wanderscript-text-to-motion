# grounding_probe — does generated motion reach a fed goal?
Updated: 2025-08-09 · Branch: track1-grounding · Requires 000_setup_3090.md PASS

## Question
With the tokenizer FROZEN, can a transformer conditioned on start pose + goal generate
motion that reaches the goal? Decides whether the tokenizer is even the bottleneck.
A PROBE, not a product: smallest setup that answers it. Single segment. No chaining,
no collision decoding, no scene-image features.

## Constraints
- VQ-VAE FROZEN.
- Add ONLY start-pose + goal conditioning to the transformer (no DINOv2/scene features).
- Single segment only. Chaining is a separate later step; do not build it here.

## Data prep
- Extract HUMANISE tokens with the FROZEN VQ-VAE.
- From the world-frame track, per clip: start pose = (x,y,yaw) at first frame (yaw as sin,cos);
  goal = (x,y) at last frame. Hold out a test split.

## Training
Finetune the transformer to predict the clip's tokens, conditioned on text (frozen T5) +
start pose + goal (learned embeddings concatenated to existing conditioning). Frozen tokens
are the targets.

## Probe (held-out)
Input: text + start + goal, no ground-truth motion. Generate tokens → frozen decoder →
canonicalized motion → SE(2) place at start pose → world-frame motion. Measure per clip:
- goal-reaching error = distance(generated end, fed goal)
- start-position error = distance(generated start, fed start)

## REQUIRED baseline — do not skip
HUMANISE clips can be short (start≈end), so a model may "reach" by luck. Also train a
NO-GOAL-conditioning transformer. The result that matters is the COMPARISON: goal-conditioned
goal-error must be meaningfully lower than unconditioned. If conditioning doesn't measurably
reduce goal-error, the probe FAILS even if absolute numbers look fine.

## Decision gate (report explicitly)
- Conditioning reduces goal-error → grounding works with the frozen tokenizer → tokenizer
  finetune likely unnecessary.
- It doesn't → grounding is the real bottleneck. Pivot candidate: predict root trajectory
  first, then generate motion along it. Report before building further.

## Watch item
Canonicalized targets give weak signal toward absolute position — model may produce plausible
motion that ignores the goal. That's what the baseline catches. Include a few generated-vs-goal
trajectory renders for eyeballing.

## Deliverable
track1_grounding/RESULTS.md: goal-error + start-error, conditioned vs unconditioned, a few
renders, clear verdict on the gate.