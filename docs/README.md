# Docs

One file per completed build-order stage, numbered to match CLAUDE.md §6. Each records what
was established and how to reproduce it. Read in order; each assumes the ones before it.

| | stage | status |
|---|---|---|
| [01](01_data_pipeline.md) | Data pipeline — join, 263-dim conversion, world-frame track | DONE |
| [02](02_baseline_calibration.md) | Baseline calibration — harness trusted, frozen tokenizer characterized | DONE (reconstruction) |
| [03](03_tokenizer_finetune.md) | Tokenizer joint finetune (Stage A) | DONE |
| [04](04_grounding.md) | Grounding probe — explicit goal conditioning | DONE |
| [05](05_token_reextraction.md) | Re-extract tokens with the finetuned VQ-VAE | DONE |
| [06](06_scene_probe.md) | Scene representation probe — DINOv2 vs occupancy | DONE |
| [07](07_continuation_probe.md) | Conditional continuation probe — the chaining mechanism | DONE |
| [08](08_transformer_finetune.md) | Transformer finetune — all conditioning together | DONE (scene arm unevaluated) |
| 09 | Chaining — multi-segment rollout + seam blend | **NEXT** |
| 10 | Collision-guided decoding | |
| 11 | Qwen JSON end-to-end → ScanNet demo | |
| 12 | Benchmark comparison + generation FID | |

Steps 04, 06 and 07 are **probes** — each isolates one question at a small budget. They are
not the system. Step 08 is where the system gets built.

Architecture, locked decisions and risks live in `CLAUDE.md`, not here.

**[IN_FLIGHT.md](IN_FLIGHT.md)** holds volatile state — running jobs, on-disk paths, the
next command. Read it before starting anything; update or delete it when its work lands.

`archive/` is superseded material kept for provenance — dated progress reports, the
original step specs, and the two parallel-track write-ups that 03 and 04 replace. Nothing
there is authoritative; if it disagrees with a numbered doc, the numbered doc wins.

## Conventions for adding a stage

- One file per stage, `NN_name.md`, added only when the stage is **done**.
- Result first, then method, then what it does not establish.
- Bug fixes belong inside the stage they affect, as a method note, and only when they
  change how a number should be read. They never get their own file and never restructure
  the sequence.
- Every number is held-out and comes with the control that makes it meaningful.
