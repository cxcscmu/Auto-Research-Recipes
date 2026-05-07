# CIFAR-10 airbench96 task — orientation (v2)

## What the swarm is searching

The baseline `airbench96.py` (vendored from upstream
`KellerJordan/cifar10-airbench/airbench96_faster.py`) reaches ≥ **96.00 %
mean test accuracy** (n=200 upstream, ~14–18 s wallclock per seed on GPU,
27.3 s on GPU). The recipe is a wide CNN + cutout aug + EMA + ~45 epochs.

The swarm searches the **training recipe** — architecture, optimizer,
augmentation, loss, regularization, scheduling — to **MINIMIZE wallclock
training time (`train_s`)** subject to the constraint that **mean
accuracy across 10 seeds stays ≥ 0.96**.

This is the strict upstream airbench96 task formulation. Time is the
score. Accuracy is the gate.

## Hardware reality

- the scheduler allocates whole eight-GPU nodes; CIFAR uses 1 GPU. 7 sit idle per trial.
  Wasteful but accepted for paper-scale data collection.
- Each seed runs ~14 s pure training + ~30–90 s cold torch.compile warmup.
- Each `submit_trial` runs N=10 seeds → ~3–5 min wallclock per trial after
  compile cache is warm; first trial of a fresh node ~5–10 min cold.
- run_trial.sh per-seed cap: 240 s; trial-total cap: 2400 s (= 40 min).
- Per-trial cost ~$1–3 (whole-node × ~5 min incl. compile).

## Pipeline shape

3 phases per `submit_trial`:

1. **Preflight (smoke)**: SMOKE_TEST=1 RUNS=1 on seed 0, 1 epoch,
   no eval — exercises imports + dataloader + optimizer + 1-step train.
   Catches edit mistakes before paying N×real-run time. ~30–90 s on cold node.
2. **Multi-seed train (N=10)**: for SEED in 0..9, RUNS=1 SMOKE_TEST=0 →
   one full airbench96 trial each. Each writes `run_seed${SEED}.jsonl`
   with that seed's train_s + accuracy.
3. **Classify-aggregate**: `run_classify.py` reads all 10 jsonls,
   computes `{mean_acc, std_acc, mean_train_s, std_train_s, n_seeds}`,
   applies the acc≥0.96 gate, writes ONE aggregated row to
   `run_seed0.jsonl` (overwriting seed-0's per-seed jsonl).

The aggregated row is what blackboard / dashboard / lineage read; per-seed
rows + logs remain on disk for debugging.

The agent edits ONLY `airbench96.py`. Helper scripts (run_trial.sh,
run_classify.py) are managed by the harness.

## Metric & threshold gate

- **Score (primary)**: `train_s` (mean across N=10 seeds, lower better).
  **`train_s` is shell-side measured by the harness** (not by airbench96.py),
  so editing the recipe to under-report time has no effect — score is
  the wallclock between python launch and python exit per seed.
- **Threshold gate**: `mean_acc(n) ≥ 0.96` (single line, strict upstream
  airbench96). At or above → OK / keep (snapshot, chainable). Below →
  DISQUALIFIED (train_s blanked, row never wins). v2.3 removed the
  earlier BORDERLINE buffer band — was a self-imposed safety margin that
  kept actual recipes from entering the leaderboard at low N.
- **Noise floor (n=10 mean)**:
  - σ_acc ≈ 0.047 % → 2σ ≈ 0.001
  - σ_train_s ≈ 0.19 s → 2σ ≈ 0.4 s on ~14 s baseline
  - **Δ_train_s improvements need ≥ 0.5 s to be > noise**.
- **Final paper number** comes from `verify_candidate.py` at N=30 seeds
  (mean σ_acc ≈ 0.027 %, σ_train_s ≈ 0.11 s) — NOT the swarm's N=10.

## N=10 proxy semantics (read this carefully)

Our N=10 protocol is **10 INDEPENDENT COLD-PROCESS TRIALS** — each seed
spawns a fresh `python airbench96.py` invocation. Upstream airbench96's
n=200 protocol is structurally different: it runs `range(200)` inside
ONE python process, sharing the compiled torch graphs and warmup costs.

Implications:
- Our `train_s` includes per-seed cold compile / warmup × 10. Upstream
  amortizes compile across 200 runs. **Our train_s is HIGHER than what
  upstream reports for the same recipe** — typically by ~3-5 s of warmup.
- Our `train_s` is more sensitive to compile-graph changes than upstream's,
  because compile cost lands in every measurement.
- **For paper, this is disclosed as a deliberate proxy choice**: 10 cold
  trials are easier to parallelize, harder to game, and orthogonal to
  upstream's compile-cache state. Cross-recipe `Δtrain_s` should still
  reflect algorithmic change because all trials pay the same baseline
  compile overhead.
- **What we don't claim**: that our absolute `train_s` is comparable to
  upstream's leaderboard times. We claim *relative reduction* on this
  hardware vs the unedited baseline.

## Specialists

| key  | scope |
|------|-------|
| arch | CifarNet width / depth / scaling — `airbench96.py:hyp['proxy','net']` |
| opt  | Muon + SGD lr / momentum / wd / lr schedule / warmup |
| aug  | Augmentation pipeline (cutout, flip, translate, mixup) — biggest leverage |
| loss | Cross-entropy variants (label_smoothing, focal, poly-1) |
| reg  | EMA, weight decay schedule, stochastic depth, dropout |

(`meta` analyst absent in v1: core's user-message contract requires every
specialist to submit_trial, conflicting with an analyst-only role.
Reintroduce when a blackboard-write tool exists.)

## Hard limits

- Edit only `airbench96.py` (single editable seed).
- DO NOT use `download=True` or hit the internet from the recipe — GPU
  pods are network-less. CIFAR_DATA_DIR is pre-baked.
- DO NOT reduce N seeds (handled by run_trial.sh's N=10 contract). The
  multi-seed gate is the task definition; overriding it falsifies results.
- DO NOT touch the acc≥0.96 threshold — that's the strict task definition.
- run_trial.sh per-seed cap: 240 s; total trial cap: 2400 s.
- 7 of 8 GPUs sit idle per trial (whole-node allocation); DO NOT try to
  use them in parallel from one experiment.
