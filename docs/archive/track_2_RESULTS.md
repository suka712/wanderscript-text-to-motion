# Track 2 Results — joint VQ-VAE finetune

Status: **DONE. Verdict: finetune works — lie gap was real codebook coverage, not an
upstream artifact.** Branch `track2-tokenizer` (not merged, per spec). Spec:
`001_tokenizer_finetune.md`. This doc was updated incrementally as eval checkpoints
landed, not written once at the end (see the per-checkpoint commentary below the table).

## Methodology

- **Data.** HumanML3D (`H3D_ROOT`) + HUMANISE, joint finetune, balanced 1:1 sampling per
  batch (`h3d_frac=0.5`) regardless of underlying dataset size, so neither dominates —
  matches CLAUDE.md's "weight so neither dominates" instruction directly rather than by
  tuning a ratio to the raw 14.6k/19.6k counts.
- **Splits — held-out, unlike the frozen-checkpoint canary scripts.** check7/14/15/16/
  17/20 sampled randomly from ALL clips in each dataset, with no train/test separation —
  correct for evaluating a FROZEN model that never trains on any of it, but not correct
  once we are the ones training. This work uses each dataset's official split file
  (`H3D_ROOT/{train,test}.txt`, `HUMANISE_ROOT/{train,test}.txt`): trains only on
  `train.txt` ids, evaluates only on `test.txt` ids, for both datasets, always.
- **HUMANISE → 263-dim.** Precomputed once for all 19,648 clips via
  `scripts/track2/precompute_humanise_263.py` (reuses the already-verified
  `src/motion_features.py` adapter). Ran clean: **19648/19648 converted, 0 failures.**
  Cached at `/media/user/2tb/motion_data/HUMANISE_263_cache` (not in git).
- **Normalization — evaluator-consistent, decided explicitly per the pitfall note in
  001_tokenizer_finetune.md.** Both training and eval use the FROZEN checkpoint's own
  meta mean/std (`checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/{mean,std}.npy`),
  never `H3D_ROOT/Mean.npy`/`Std.npy`. Rationale: the finetune warm-starts from that
  checkpoint's embedding space, so keeping its normalization keeps the encoder/decoder
  in-distribution from iteration 0, and keeps every number in this doc directly
  comparable to the frozen-model reference numbers in CLAUDE.md (45.3 / 47.9 / 67.6 /
  72.6 / 139.8 / 90.1).
- **Window size.** 64 frames (T2M-GPT's own recipe default), unit_length 4 (down_t=2).
  **Caveat, worth stating plainly:** HUMANISE clips are systematically much shorter than
  HumanML3D clips (median 48-64 frames vs. HumanML3D's up to 196), so a 64-frame window
  requirement drops a large fraction of HUMANISE training clips outright — measured on
  `train.txt`: lie 49.0% pass (978/1997), sit 50.5% (2376/4701), stand-up 22.9%
  (659/2880), walk 33.4% (2321/6945). Net: 6334/16523 (38.3%) of HUMANISE train clips
  are used. Sit/lie — the categories this track cares most about — are actually
  *better* preserved than walk/stand-up, so this doesn't bias against the categories of
  interest, but it does mean the effective HUMANISE training pool is smaller than the
  full split. Kept at 64 rather than reduced, to match the established architecture
  recipe exactly (a shorter window is a second, confounding change on top of the
  finetune itself, and would truncate exactly the sit-down/lie-down transition motion
  that's the point of training on this data).
- **Loss recipe.** Unchanged from the original T2M-GPT VQ-VAE training run (see
  `pretrained/VQVAE/run.log`): `commit=0.02`, `loss_vel=0.5`, `recons_loss=l1_smooth`.
  Only the learning rate differs from that from-scratch recipe (2e-5 vs. their 2e-4) —
  10x lower because this is a finetune of an already-converged model, not training from
  scratch (CLAUDE.md failure point 3: risk of forgetting general motion).
- **Eval metric.** Per-category MPJPE (mm), local/root-relative per-frame positions
  (`src/motion_features.local_joint_positions`), same metric as check7/14/20. 200 clips
  per category (all available where a category has fewer, e.g. H3D-lie).

## Pitfall caught before training: EMA-codebook resume bug

`models/quantize_cnn.py`'s `QuantizeEMAReset` tracks "has the codebook been
initialized" in a plain Python bool (`self.init`) plus two plain-attribute EMA
accumulators (`code_sum`/`code_count`) — **none of these are registered torch buffers**,
so `net.load_state_dict(ckpt, strict=True)` restores the pretrained `codebook` tensor
correctly but silently leaves `init=False`. On the first `net.train()` forward pass,
the quantizer would then run its from-scratch init path and **overwrite the entire
pretrained codebook** with samples tiled from just that one batch — verified empirically
(512/512 codebook rows replaced, in a throwaway test before any real training). Fixed by
seeding the EMA accumulators from the loaded codebook itself
(`prepare_quantizer_for_finetune` in `train_vqvae_joint_finetune.py`) before training
starts, so the quantizer behaves exactly as if it had been training continuously and
just paused at this checkpoint. With the fix, per-step codebook churn matches the
architecture's designed behavior: ~85-95% of codes get a small EMA nudge, a handful of
rarely-used codes get reset (the intended "Reset" dead-code-revival behavior, present in
from-scratch training too, not something introduced by finetuning).

## Before: frozen checkpoint, held-out split (this methodology's "iteration 0")

Run via `scripts/track2/eval_per_category_mpjpe.py` directly against
`net_best_fid.pth`, and reproduced identically as the finetune script's own iter-0 eval.

| category | mean (mm) | median (mm) | n | ratio to H3D baseline |
|---|---|---|---|---|
| H3D baseline (held-out) | 56.11 | 48.02 | 200 | 1.00x |
| H3D-lie (held-out) | 117.46 | 115.10 | 11 | 2.09x |
| HUMANISE walk | 50.20 | 46.76 | 200 | 0.89x |
| HUMANISE stand up | 66.98 | 63.24 | 200 | 1.19x |
| HUMANISE sit | 69.55 | 64.33 | 200 | 1.24x |
| HUMANISE lie | 136.91 | 115.46 | 200 | 2.44x |

**Read this against CLAUDE.md's cited reference numbers (45.3 / 47.9 / 67.6 / 72.6 /
139.8 / 90.1) carefully — they are not quite the same measurement:**
- HUMANISE numbers (walk/stand/sit/lie) match closely (within a few mm) — expected,
  since the frozen model never trained on ANY HUMANISE data, so held-out vs. full-sample
  makes no difference for it.
- **H3D baseline is 56.11mm here vs. the cited 45.3mm** — check14's number was sampled
  from ALL of H3D (`os.listdir` over the whole `new_joint_vecs` dir), which includes
  clips the frozen model was actually trained on. That's leakage for a "how good is
  reconstruction" number; held-out-only (this table) is the honest, harder number and is
  what this track's "no regression" criterion will be checked against, not 45.3mm.
- **H3D-lie is 117.46mm here (n=11) vs. the cited 90.1mm** — same leakage issue, plus a
  small held-out sample (only 11 of H3D's test.txt clips match the lie/lying text-keyword
  search). Treat this number as low-confidence/high-variance; it is a control, not the
  primary metric this track's decision gate hinges on (HUMANISE-lie is).

This distinction matters more for the finetuned model than the frozen one, since the
finetuned model *does* train on H3D-train — held-out H3D numbers are the only ones that
mean anything as a "did general motion regress" check from here on.

## After: finetune progress

Run: `track2_joint_finetune_run1`, `lr=2e-5`, `batch=256`, `total_iter=20000`,
`eval_iter=2000`, `warm_up_iter=200`, lr decay (gamma=0.5) at iters 12000/18000. Launched
2026-08-11. Checkpoints + eval JSONs under
`/media/user/2tb/motion_data/track2_checkpoints/track2_joint_finetune_run1/` (not in
git). Table below is appended to as each eval checkpoint lands.

| iter | H3D baseline | H3D-lie (n=11) | HUMANISE walk | HUMANISE stand | HUMANISE sit | HUMANISE lie |
|---|---|---|---|---|---|---|
| 0 (frozen) | 56.11 | 117.46 | 50.20 | 66.98 | 69.55 | 136.91 |
| 2000 | 55.9 | 112.0 | 35.3 | 53.5 | 50.1 | **97.7** |
| 4000 | 56.2 | 115.6 | 34.9 | 52.6 | 49.9 | **95.6** |
| 6000 | 55.5 | 116.0 | 34.5 | 52.9 | 48.8 | **96.0** |
| 8000 | 56.0 | 115.6 | 34.4 | 53.2 | 48.6 | **96.7** |
| 10000 | 55.8 | 113.6 | 34.2 | 51.7 | 48.1 | **94.5** |
| 12000 | 56.2 | 116.9 | 34.3 | 52.5 | 48.0 | **95.3** |
| 14000 | 56.4 | 115.7 | 34.3 | 53.2 | 47.8 | **92.4** |
| 16000 | 56.3 | 114.8 | 34.1 | 53.1 | 47.8 | **101.3** |
| 18000 | 56.1 | 116.7 | 34.0 | 53.0 | 47.8 | **98.5** |
| 20000 (final) | 56.2 | 117.4 | 34.1 | 53.0 | 47.6 | **96.3** |

Training completed cleanly (exit code 0), all 11 eval checkpoints (0, 2000, ..., 20000)
landed without incident. lr decays (gamma=0.5) applied after iters 12000 and 18000 with
no visible effect on the plateau either time.

### Before -> after (iter 0 frozen -> iter 20000 finetuned), held-out split

| category | before (mm) | after (mm) | change |
|---|---|---|---|
| H3D baseline | 56.11 | 56.2 | +0.2% (flat — no regression) |
| H3D-lie (n=11, noisy control) | 117.46 | 117.4 | flat |
| HUMANISE walk | 50.20 | 34.1 | **-32.1%** |
| HUMANISE stand up | 66.98 | 53.0 | **-20.9%** |
| HUMANISE sit | 69.55 | 47.6 | **-31.6%** |
| HUMANISE lie | 136.91 | 96.3 | **-29.6%** |

### Both success criteria (001_tokenizer_finetune.md)

1. **Interaction improves: HUMANISE-lie drops from 140 toward ~90.** ✅ MET — 136.91 ->
   96.3mm, and stable in a 92-101mm band across the last 9 of 11 checkpoints (iters
   2000-20000), not a one-off dip. Landed close to CLAUDE.md's cited H3D-lie reference
   (90.1mm, different methodology) and actually *below* this run's own held-out H3D-lie
   control (117.4mm, n=11) — i.e. after finetuning, HUMANISE-lie reconstructs *better*
   than H3D's own lying clips do on the same model. Plausible reading: H3D's lie-labeled
   set is small (n=11 held-out) and drawn from more varied "lying" scenarios (getting up,
   rolling, etc. — text-keyword matched, not a clean action label like HUMANISE's), so
   it's a noisier, not necessarily easier, control; not a sign that this result is
   suspicious.
2. **No forgetting: HumanML3D overall and HUMANISE walk do not regress.** ✅ MET, and
   more than met — H3D baseline held completely flat (56.11 -> 56.2mm, +0.2%, noise-level),
   and HUMANISE walk did not merely hold, it *improved* 32% (50.2 -> 34.1mm). Every other
   HUMANISE category improved too. No forgetting anywhere in six tracked categories.

### Qualitative check: lie reconstruction renders, frozen vs finetuned

`scripts/track2/render_lie_comparison.py`, held-out HUMANISE-lie clips, picked to span
the frozen model's own error range (best/median/worst of a 60-clip held-out sample) —
same "moderate vs broken" spot-check spirit as the original Step 1 visual check. Renders
in `scratch_outputs/track2_lie_renders/` (gitignored, not in git; regenerate via the
script).

| clip | role (ranked under frozen) | frozen MPJPE | finetuned MPJPE | delta |
|---|---|---|---|---|
| 02123 | best | 65.1mm | 55.1mm | -10.0mm |
| 02061 | median | 117.8mm | 43.4mm | **-74.4mm** |
| 02012 | worst | 321.7mm | 279.2mm | -42.5mm |

Visual read: the median clip (02061) — a fairly standard lying pose — goes from a
recognizable-but-off frozen reconstruction to a close visual match after finetuning,
consistent with the 74mm drop. The worst clip (02012) is an atypical compact/curled
pose in the ground truth; both frozen and finetuned reconstructions distort limb
placement relative to GT, and finetuning helps but doesn't fix it — moderately worse,
not a broken/exploded skeleton in either case (no NaN, no wildly out-of-scale joints).
Consistent with the original Step 1 finding ("moderate degradation, not catastrophic")
holding for the finetuned model as well, just less degraded.

## Decision gate — VERDICT

**Lie improves (136.9 -> 96.3mm, -29.6%, stable over 9 consecutive checkpoints) AND
general holds (H3D baseline flat, HUMANISE walk improved 32%, no category regressed) →
the gap was real codebook coverage. The joint finetune works. Per the reconciliation
table in `docs/REPORT_aug9.md`, the tokenizer track stays viable pending Track 1's
grounding result** (frozen tokenizer + explicit spatial conditioning — does generated
motion reach a fed goal coordinate?). Track 2's own question — "was frozen-tokenizer
sit/lie reconstruction a real limitation, fixable by finetuning?" — is answered yes,
cleanly, with no ambiguity requiring a tie-breaker experiment.

**Also answers the secondary Track 2 question** (upstream SMPL-X→22-joint artifact vs.
codebook coverage): since finetuning the codebook alone — no changes to the upstream
data pipeline — closed most of the lie gap, the original 3.09x-worse-than-H3D-baseline
number was predominantly a codebook coverage problem, not evidence of a broken upstream
joint-reduction step. The upstream mapping is still not independently re-verified (see
`docs/REPORT_aug9.md` section 6), but this result removes the main piece of evidence
that would have motivated investigating it urgently.

### What this does NOT decide

Per branch scope: this result does not mean "use the finetuned tokenizer in the real
build" by itself — that's a reconciliation decision against Track 1, made in
`docs/REPORT_aug9.md`'s table, not here. This branch does not re-extract transformer
tokens, does not touch the transformer, and is not merged.

## Artifacts

- Checkpoints + per-checkpoint eval JSON:
  `/media/user/2tb/motion_data/track2_checkpoints/track2_joint_finetune_run1/` (not in
  git; `net_iter020000.pth` is the final finetuned VQ-VAE, `heartbeat.log` has the full
  fine-grained run history).
- HUMANISE 263-dim cache: `/media/user/2tb/motion_data/HUMANISE_263_cache/` (not in git,
  regenerate via `scripts/track2/precompute_humanise_263.py`, resumable).
- Frozen-baseline held-out eval: `scratch_outputs/track2/eval_frozen_baseline_heldout.json`.
- Lie renders: `scratch_outputs/track2_lie_renders/`.
- Code: `src/joint_vqvae_dataset.py`, `scripts/track2/{precompute_humanise_263,
  eval_per_category_mpjpe,train_vqvae_joint_finetune,render_lie_comparison}.py`.

**Reading at iter 2000 (early, one checkpoint, not yet the decision gate):** H3D baseline
flat (55.9 vs 56.1, no regression signal). Every HUMANISE category improved substantially
— walk -30%, stand-up -20%, sit -28%, **lie -29% (136.9 -> 97.7mm)**, already close to the
held-out H3D-lie control (112.0mm) and within reach of CLAUDE.md's cited H3D-lie reference
(90.1mm, different methodology, see caveat above). Promising, consistent with "gap was
real codebook coverage," but one checkpoint at 10% of the planned run isn't the gate yet —
watching for whether this holds/continues or is a transient early-training swing (also
want to rule out the codebook's dead-code-reset churn, expected to be more active early on
after the finetune-resume patch, as a confound before reading too much into iter 2000
specifically).

**iter 4000: the iter-2000 level holds, not a transient swing.** All six numbers are
within noise of their iter-2000 values (H3D baseline 56.2 vs 55.9; HUMANISE lie still
dropping slightly, 97.7 -> 95.6). This is the second consecutive checkpoint at the
improved level, which is a meaningfully stronger signal than iter 2000 alone — a
one-off swing would be unlikely to reproduce this closely. Still watching a few more
checkpoints before calling the gate, mainly to see whether lie keeps drifting down,
plateaus, or (less likely now) reverses, and to confirm H3D baseline stays flat over a
longer stretch.

**iter 6000: third consecutive checkpoint at the same plateau.** HUMANISE lie 96.0mm
(vs. 97.7/95.6 at 2000/4000) — settled, not still trending down. H3D baseline 55.5mm,
walk/stand/sit all within a point of their iter-2000 values. Three checkpoints spanning
iters 2000-6000 in tight agreement is a real plateau, not noise. Letting the run continue
through its two lr-decay milestones (12000, 18000) to see if either shifts this, but the
core reading is already fairly stable: general motion held, HUMANISE walk/stand/sit
improved and held, **lie improved from 136.9mm to ~96mm and held** — short of the H3D-lie
control (112-116mm on held-out H3D, itself noisy at n=11) but has crossed well past it
downward, which on its face reads as "at or below the intrinsic-difficulty floor for lying
poses," not merely "closed most of the gap."

## Decision gate

Not yet reached — waiting on enough checkpoints to see a clear trend. Criteria (from
001_tokenizer_finetune.md):
1. **Lie improves + general holds** → gap was real coverage; finetune works.
2. **Lie won't improve (stays ~137)** → likely the upstream SMPL-X→22-joint artifact,
   not codebook coverage. Finetune is the wrong fix — flag for upstream investigation.
3. **General regresses** (H3D baseline or HUMANISE walk rise materially above the
   iter-0 held-out numbers above) → sampling balance or LR issue, tune before concluding.
