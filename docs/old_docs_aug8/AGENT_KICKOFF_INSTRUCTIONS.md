# Agent kickoff instructions — Track 1 (3090) / Track 2 (4090)

Paste these verbatim into a fresh session on each box. Both tracks are defined in
`docs/track_1/` and `docs/track_2/`; see `docs/REPORT_aug9.md` for the reconciliation
plan once both report results.

---

## 3090 agent (Track 1 — grounding probe)

```
You're working on the wander project (scene-aware text-to-motion). Fresh clone this repo.

1. Read CLAUDE.md and docs/REPORT_aug9.md first for full context — you have no prior
   conversation history, these are the source of truth.
2. Branch off `master` (NOT `main` — origin/main on this remote is an unrelated old repo,
   see docs/track_1/000_setup_3090.md for why): git checkout -b track1-grounding
3. Follow docs/track_1/000_setup_3090.md, then docs/track_1/001_grounding_probe.md, in
   order. Do not start the probe until the setup doc's smoke test passes.
4. Watch for the normalization pitfall called out in 000_setup_3090.md — if your smoke
   test number comes out ~2-3x higher than expected, that's almost certainly it, not a
   real environment problem.
5. Report progress incrementally in docs/track_1/RESULTS.md as you go — don't wait until
   the end. Write the smoke-test result as soon as you have it, then goal-error/
   start-error numbers as each model (conditioned vs unconditioned baseline) finishes
   training, not just a final summary. If a step is taking a long time, note what's
   running and roughly how far along, so anyone checking in isn't reading silence.
6. Do not merge this branch. Stop and report clearly once you hit the decision gate at
   the end of 001_grounding_probe.md.
```

---

## 4090 agent (Track 2 — tokenizer finetune)

```
You're working on the wander project (scene-aware text-to-motion), same repo you're
already on. This box already has the data and checkpoints — no setup needed.

1. Read CLAUDE.md and docs/REPORT_aug9.md first for full context — you have no prior
   conversation history, these are the source of truth.
2. Branch off `master` (NOT `main` — origin/main on this remote is an unrelated old repo):
   git checkout -b track2-tokenizer
3. Follow docs/track_2/001_tokenizer_finetune.md.
4. This exact GPU has caused multiple silent multi-hour failures in earlier sessions
   (documented in docs/old_docs_aug8/STEP2_baseline_calibration.md) — before assuming a
   slow run is broken, check `nvidia-smi --query-compute-apps` for contention, and use a
   generous timeout (or none) rather than an artificial bound — a tight self-imposed
   timeout was the actual cause of three of those past failures, not the GPU.
5. Report progress incrementally in docs/track_2/RESULTS.md as you go — per-category
   MPJPE (walk/stand/sit/lie + H3D overall/lie) after each eval checkpoint, not just a
   final number. State explicitly which mean/std normalization you used (see the
   pitfall note in 001_tokenizer_finetune.md) so results are comparable to the reference
   numbers.
6. Do not re-extract transformer tokens or touch the transformer. Do not merge this
   branch. Stop and report clearly once you hit the decision gate at the end of
   001_tokenizer_finetune.md.
```
