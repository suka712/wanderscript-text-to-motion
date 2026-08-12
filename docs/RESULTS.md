# Results — everything established, one file

One section per build-order stage. Result first, then only the method details that change how
a number should be read. Reproduce commands live in the script docstrings, not here.
Superseded per-stage docs are in `archive/`.

Every number is held-out. Every generative number is quoted against an **oracle** (ground
truth pushed through the identical path) — a model number without its oracle is not
interpretable, and this project has twice published one that turned out to be measuring a bug.

---

## 1 · Data pipeline — DONE

| check | result |
|---|---|
| HUMANISE 3-source ID join | 19,648 / 19,648 |
| 22-joint → 263-dim vs HumanML3D's own vectors | **0.80 mm** MPJPE, foot contact bit-exact |
| HUMANISE → 263 at scale | 19,648 / 19,648, 0 NaN |
| world-frame track vs scene-mesh floors | 150 / 150 scenes |
| ScanNet meshes load / BEV world→pixel error | 643 / 643 · 0.67 px |

Two-track storage (CLAUDE.md 2a): canonicalized 263-dim for the tokenizer, plus a world-frame
`(x, y, yaw)` root track, because the 263-dim is position-invariant and cannot know where it
is. The converter wraps HumanML3D's own extractor rather than reimplementing it.

**Caveat:** the upstream SMPL-X → 22-joint reduction is HUMANISE's own and has never been
independently verified. It underlies every HUMANISE number here. No longer the prime suspect
for anything (see §3), but unchecked. Also: 2 NaN-corrupted files exist in HumanML3D.

## 2 · Baseline calibration — DONE (reconstruction only)

Reconstruction FID **0.066** vs paper 0.070; R@1–3 within variance. Harness trusted.

Frozen tokenizer, held-out MPJPE: H3D 56.11 · H3D-lie 117.46 (n=11) · HUMANISE walk 50.20 ·
stand-up 66.98 · sit 69.55 · **lie 136.91**.

**Two traps, both of which have already cost this project real time:**
- **Normalization.** The VQ-VAE expects its *own* checkpoint mean/std
  (`checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/`), never H3D's `Mean.npy`/`Std.npy`.
  Wrong pair inflates everything ~2-3× (H3D reads 137 mm instead of 45). Suspect this first
  when a reconstruction number looks absurd.
- **The old 45.3 mm H3D baseline was LEAKED** — sampled from all of `new_joint_vecs`,
  including training clips. Honest held-out figure is **56.11**; H3D-lie is **117.46**, not
  90.1. Do not quote 45.3 or 90.1.

**Still open:** generation FID / R-precision (autoregressive, not encode/decode) — unreproduced
after 5 attempts, root-caused to a self-inflicted timeout. Nothing downstream needed it, but it
means **no number here is comparable to published work.**

## 3 · Tokenizer joint finetune — DONE, works

HumanML3D + HUMANISE, 1:1 balanced, lr 2e-5, 20k iters. Held-out:

| | H3D | H3D-lie | walk | stand up | sit | lie |
|---|---|---|---|---|---|---|
| frozen | 56.11 | 117.46 | 50.20 | 66.98 | 69.55 | 136.91 |
| finetuned | 56.2 | 117.4 | **34.1** | **53.0** | **47.6** | **96.3** |

Both gates met: interaction improved, nothing regressed. Lie holds a 92–101 mm band across the
last 9 of 11 checkpoints. **The sit/lie gap was real codebook coverage, not the unverified
upstream joint reduction** — finetuning the codebook alone closed most of it.

**Landmine:** `QuantizeEMAReset` keeps `init`/`code_sum`/`code_count` as plain Python
attributes, not buffers. `load_state_dict(strict=True)` restores the codebook but leaves
`init=False`, so the first `train()` forward **overwrites all 512 codes from one batch**.
Anyone resuming or finetuning this VQ-VAE must seed the EMA accumulators from the loaded
codebook first (`prepare_quantizer_for_finetune`).

**Caveat:** the 64-frame window keeps only 38.3% of HUMANISE train clips (lie 49%, sit 50%,
stand-up 23%, walk 33%). Sit/lie survive better than walk, so it does not bias against the
categories of interest.

## 4 · Grounding — DONE, works

Frozen tokenizer, single segment, 200 held-out clips:

| | goal err | corr(commanded, achieved) |
|---|---|---|
| NULL (stay at start) | 0.627 m | — |
| unconditioned | 0.490 m | +0.07, +0.62 |
| conditioned, **absolute** frame | 0.515 m | −0.12, +0.60 |
| conditioned, **relative** frame | **0.164 m** | **+0.90, +0.97** |
| ORACLE | 0.124 m | — |

**The one decision that matters: feed the goal in the frame the model generates in** —
start-relative, heading-aligned (`se2_utils.world_to_local_xy`). Absolute world coordinates
force it to compute `R(yaw₀)ᵀ(goal − start)`, a bilinear op, through a single `Linear` that
cannot represent a rotation; it does not learn this. The start pose is **not** a model input —
in that frame it is (0,0) at heading 0 by construction, and enters only at SE(2) placement.

Holds across range (600 clips): 0.071 m error at <0.25 m commanded → 0.508 m at >2 m, with the
model/oracle ratio flat at 1.1–1.4. **Goal accuracy is limited by tokenizer reconstruction,
not by goal-following.**

**The 90° bug, kept because it explains the discipline.** The first pass reported grounding
FAILS and recommended re-architecting the project. Its `se2_place` rotated by `yaw0` instead of
`yaw0 + π/2`, rotating every trajectory 90°. Start-error is exactly 0.0 under *any* rotation,
so the only sanity check run was structurally blind to it. A ground-truth-token oracle catches
it instantly: 0.862 m at offset 0 vs 0.124 m at +π/2 — i.e. the bug alone accounted for the
entire "model" error of 0.828 m.

## 5 · Token re-extraction — DONE, verified

Stage A changed the codebook, so all prior tokens are invalid. 19,648 clips re-tokenized;
frozen tokens kept separately so §4 stays reproducible.

Verification (file sizes prove nothing — token count depends only on clip length): 512/512
codebook rows changed · only 80/3125 test clips have identical token sequences · 45.3%
per-token agreement · goals/starts unchanged.

Oracle floor **0.124 → 0.107 m**. Probe re-run on the new tokenizer: unconditioned 0.549 m,
**conditioned 0.132 m** — conditioning now cuts error 76% (was 67%), model/oracle ratio
tightening 1.32 → 1.23. **Stage A improved goal-reaching with no change to the grounding
mechanism.**

**The hazard this step prevents, quantified:** new tokens through the *old* decoder gives
0.321 m, 3× worse, with nothing raised. Stale tokens decode to something plausible rather than
crashing. Always name the tokenizer explicitly.

## 6 · Scene representation — DONE, DINOv2 dropped

4-way action classification from scene context at the goal, no text, **split by scene**, linear
head, 175 held-out scenes:

| arm | acc |
|---|---|
| prior | 26.4% |
| DINOv2 ViT-B/14 | 34.3% |
| RGB crop, **no encoder** | 36.4% |
| DINOv2 ViT-S/14 | 37.6% |
| **occupancy crop** | **63.0%** |

**DINOv2 does not beat its own unencoded input, and ViT-B scores below ViT-S** — a domain
problem, not capacity. Occupancy nearly doubles it and we already render it. Scene encoder
swapped to the **occupancy raster cropped in the agent's frame**: removes a ViT from the loop,
drops a 350 MB dependency, and reuses the raster collision needs anyway.

63% is near the ceiling for static geometry: sit and stand-up happen at the same furniture and
differ only in motion direction, so nothing static separates them (stand-up recall 0.51).

## 7 · Conditional continuation — DONE, works

The chaining mechanism, tested at one seam. 300 held-out clips:

| | seam err | goal err |
|---|---|---|
| no-prefix (naive) | 155.8 mm | 0.0707 m |
| **continuation, exact prefix** | **71.2 mm** | 0.0573 m |
| continuation, **reconstructed** prefix (realistic) | 86.0 mm | 0.1073 m |
| ORACLE | 70.9 mm | 0.0544 m |
| canonicalization floor | 21.6 mm | — |

Lands within 0.3 mm of the oracle — the conditioning extracts everything available.

**Three findings that constrain chaining:**
1. **Condition on the ending body configuration, not tokens.** Segments are canonicalized
   individually, so segment *k*'s tokens describe a different frame. Root-relative,
   heading-canonicalized joint positions (66-d) are frame-independent. Express them in *k+1*'s
   frame — computable at inference, since *k+1*'s canonical frame is *defined by* the seam pose.
2. **The prefix pose must be ON-MANIFOLD.** A VQ-VAE-reconstructed prefix (structured ~70 mm
   error) costs 15 mm. iid Gaussian noise at 25 mm — a fifth the magnitude but anatomically
   impossible — collapses it to 331 mm, *flat* out to 200 mm. So **never feed a blended,
   interpolated or smoothed pose forward**; feed the decoded one.
3. **The seam blend is not cosmetic.** ~70 mm of residual is VQ-VAE reconstruction of the first
   frame (oracle 70.9 vs a 21.6 mm floor) and no conditioning touches it.

**Canonicalization residual (21.6 mm):** `process_file` normalizes per segment —
`uniform_skeleton` rescales from each segment's first frame, floor offset is each segment's
minimum height — so two segments never agree exactly on the same physical pose. The floor term
is chicken-and-egg at inference. Seams below ~20 mm need the representation changed.

## 8 · Transformer finetune — DONE; scene arm UNTESTABLE on this data

All conditioning together (text 512 + goal 2 + seam pose 66 + occupancy 784 = 1364-d), 20k iters:

| | seam | goal |
|---|---|---|
| full (with scene) | 71.7 mm | 0.0584 m |
| noscene (ablation) | 71.4 mm | 0.0564 m |
| ORACLE | 70.9 mm | 0.0544 m |

**The combination does not degrade** — matches the 4k-iter §7 probe. A prior recorded before
the run predicted it would underperform; that prior was wrong.

**Ignore token accuracy (99.5%).** The seam pose nearly determines the early target. Never
quote it as quality.

**Why the ablation answers nothing — it is the data.** Both models sit *at* the oracle: 0.8 mm
and 4 mm of headroom. Goal error is also the wrong question for a scene arm, which is about
obstacles, not coordinates. Building a collision metric took three attempts:

1. *Exclude the target ScanNet instance* — **blocked**, no instance segmentation on either box.
2. *Connected-component proxy* — **failed**, 331/400 clips merged furniture with walls.
   Abandoned rather than tuned, since loosening the guard excludes the walls the metric exists
   to detect. Kept as `src/target_occupancy.py` with the failure documented.
3. *Tall-obstacle map* — **works.** Ground-truth collision by obstacle-height threshold:

| threshold | lie | sit | walk | raster occupied |
|---|---|---|---|---|
| 0.12 m (default) | 100.0% | 88.6% | 30.2% | 22.7% |
| **0.90 m** | **20.1%** | **3.9%** | **0.1%** | **6.1%** |
| 1.50 m | 0.0% | 2.6% | 0.0% | 1.5% |

**Collision must be scored at 0.9 m**, never 0.12 m. At 0.12 m the metric is void on an
interaction dataset — ground-truth `lie` "collides" 100% because the person is on the bed, and
being on the target *is the objective*. Cached at `~/wander_data/bev_tall_cache`.

With a correct metric the ablation *still* returns nothing: full 3.09% · noscene 3.09% ·
ORACLE 3.09% · **NULL (never move) 3.25%**. NULL landing in the same place is the tell —
collision is a property of the **start pose**, not the path. **Only 25 of 2962 clips (0.8%)
walk further than 1.5 m**; mean displacement is 0.63 m.

**This one fact explains every failed step-8 evaluation.** Goals are 0.6 m away so goal error
saturates; nothing moves so collision is start-determined; there is no navigating to do so the
scene arm has nothing to do. **It is the absence of a test, not evidence that occupancy
conditioning fails.**

Consequences: HUMANISE alone cannot demonstrate scene-aware navigation. **Chaining is the
unlock** — composing segments is what produces multi-metre paths and therefore the first
setting where a scene arm is measurable. And **check how PSMo/AffordMotion define non-collision
before any comparison**; theirs cannot be the naive definition.

---

## Conditioning inputs — evidence status

| input | status |
|---|---|
| relative-frame goal | validated (§4, §5) |
| seam pose | validated (§7) |
| occupancy scene | representation validated (§6); contribution to generation **untestable on this data** |

## Bug ledger — five silent convention bugs, same shape each time

Z-up vs Y-up · wrong mean/std in the MPJPE canary · the same normalization error again in the
grounding probe · the 90° SE(2) yaw error · the collision metric scoring the objective as
failure. The last two were caught only after producing published-looking numbers, and the 90°
one inverted a track's conclusion.

The pattern in the misses: each was checked with a metric structurally incapable of detecting
it. Start-error cannot see a rotation; per-frame MPJPE cannot see cumulative drift; a collision
rate cannot see that it is measuring the goal. **Every pipeline that emits a number gets an
oracle control before any model number is read off it.**
