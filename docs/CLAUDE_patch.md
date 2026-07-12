# CLAUDE.md — patch (apply these edits)

Apply to the existing CLAUDE.md at repo root. Three changes, from the STEP1b findings.

## 1. Axis convention — correction
Previous text implied HUMANISE is Y-up. **This was wrong.** HUMANISE raw joint data is
natively **Z-up** (verified empirically via anatomical joint-height ordering + HUMANISE's
own placement code). World frame remains ScanNet Z-up. No Y-up→Z-up conversion is needed
for HUMANISE joints; do not reintroduce one. (A `scene_translation` offset in the
placement transform was found missing and fixed — keep it.)

## 2. MVP scope — locomotion only
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

## 3. VQ-VAE frozen — confirmed and reinforced
The lie/sit break is expressiveness of the discrete codebook, not decoder weights.
Decoder-only finetuning can't fix it; fixing it needs encoder+codebook+decoder retrain,
which discards the "reuse T2M-GPT tokens" core of the design. Since navigation doesn't
need those motions, **VQ-VAE stays frozen.** Revisit only if V2 requires scene
interaction. This is now a hardened decision, not just an initial choice.
