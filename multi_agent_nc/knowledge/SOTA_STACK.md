# Current SOTA stack — d12 miniseries baseline

Source: vendored `karpathy/nanochat` at the pinned commit (see
`vendor/.commit_pin` for the exact hash). The vendor tree is
copied **into your workdir** on iter 1 — your edits to vendor .py files
take effect via PYTHONPATH injection in experiment.py.

## Architecture (d12 ~110 M params)

- 12 transformer layers (FIXED — defines the miniseries).
- `model_dim = depth × aspect_ratio = 12 × 64 = 768` (nudged to nearest
  multiple of `head_dim=128` for clean attention shape division).
- `n_head = model_dim / head_dim = 6`.
- Sliding-window attention with `--window-pattern SSSL` (3 short-window
  layers + 1 long-window layer, tiled across the 12 layers).
- `max_seq_len=2048` baseline.
- Block / MLP / Attention defined in `vendor/nanochat/gpt.py`.

## Optimizer

Two optimizers run together; both live in `vendor/nanochat/optim.py`.
- **Muon** (matrix params): `--matrix-lr 0.02`, momentum scheduled
  0.85→0.97 (warmup) → 0.97 → 0.90 (warmdown).
- **AdamW** (embeddings, unembedding, scalars):
  `--embedding-lr 0.3`, `--unembedding-lr 0.008`, `--scalar-lr 0.5`,
  `--weight-decay 0.28` (cosine-decayed during training).

## Schedules

LR schedule (assembled in `vendor/scripts/base_train.py`):
linear warmup (`--warmup-steps 40`) → constant → linear warmdown
(`--warmdown-ratio 0.65`) → final LR fraction (`--final-lr-frac 0.05`).

Muon momentum: 3-phase trapezoid synchronous to LR.
AdamW weight decay: cosine over training.

## Training horizon

- Default: `--target-param-data-ratio 12` (Chinchilla-like for d12).
  Auto-computes `num_iterations` from data:param ratio.
- Override: `--num-iterations N` for a fixed step count.

## Data

- ClimbMix-400B shards under `$NANOCHAT_BASE_DIR/base_data_climbmix/`
  (read-only, operator-baked).
- Streaming via `vendor/nanochat/dataloader.py` — last shard is val,
  rest train.
- Tokenizer: BPE pre-baked into
  `$NANOCHAT_BASE_DIR/tokenizer/{tokenizer.pkl, token_bytes.pt}`.
  Wrapped by `vendor/nanochat/tokenizer.py`.

## Eval

- val_bpb every `--eval-every` steps (default 250) on `--eval-tokens` tokens.
- CORE metric every `--core-metric-every` steps (default 2000), on
  `--core-metric-max-per-task` examples per task.
- base_train emits the final `core_metric` and `val_bpb` to stdout —
  experiment.py regex-parses them.

## Numerics + kernels

- FP8 matmul gated by `--fp8` (default off); scaling recipe via
  `--fp8-recipe`. Implementation in `vendor/nanochat/fp8.py`.
- Flash-attention + sliding window in
  `vendor/nanochat/flash_attention.py`.

## Editable surface (per specialist)

| spec  | primary file(s)                                        | typical levers                                |
|-------|--------------------------------------------------------|-----------------------------------------------|
| arch  | `vendor/nanochat/gpt.py`                               | Block layout, MLP shape, normalization, residual structure, head wiring |
| opt   | `vendor/nanochat/optim.py`                             | Muon iterations, momentum coupling, AdamW eps/beta, parameter-group routing |
| data  | `vendor/nanochat/dataloader.py`, `tokenizer.py`        | Shuffle/replay, packing, val split shape, eval-token sampling |
| sched | `vendor/scripts/base_train.py` (schedule helpers)      | LR shape (cosine? 1cycle?), momentum trapezoid, wd cosine-vs-linear, warmup/warmdown ratios |
| sys   | `vendor/nanochat/fp8.py`, `flash_attention.py`         | FP8 recipe, kernel fusion, sliding-window pattern impl |

CLI flags in `experiment.py`'s `TRAIN_ARGS` are still legal and often
the cheapest first move — no need to rewrite a file when a flag works.

## What stays fixed

- `--depth = 12` (miniseries definition). `experiment.py` always passes
  it; if you remove the `args.depth` plumbing in `scripts/base_train.py`,
  cross-trial comparison is invalid (profile_pipeline warns).
- Tokenizer + data shards (operator-baked, read-only).
- Anything outside `experiment.py` + `vendor/nanochat/` — see INIT.md
  off-limits list.

## Mutation directions worth trying (initial)

These are starting points, not orders. Read LESSONS.md for what's
actually been tried.

- **arch** (gpt.py): residual scaling, parallel residuals, pre/post
  norm placement, FFN-shape variants (gated, SwiGLU width), head-merging
  tricks; or pure-flag: window_pattern variations, aspect_ratio.
- **opt** (optim.py): Muon Newton-Schulz polynomial degree, momentum
  decoupling, AdamW eps/beta tuning; ratio of matrix-lr vs scalar-lr
  via flags.
- **data** (dataloader.py + tokenizer.py): packing strategy, val-split
  reshuffle, sequence-packing density; or pure-flag: max-seq-len,
  device/total batch size, target-param-data-ratio.
- **sched** (base_train.py): different warmup/warmdown shapes (cosine
  vs linear vs trapezoid), wd schedule shape, momentum-LR coupling
  details; or pure-flag: warmup-steps, warmdown-ratio, final-lr-frac.
- **sys** (fp8.py + flash_attention.py): FP8 enable + recipe choice,
  kernel fusion of attention prep, alternative window-pattern impls;
  or pure-flag: --fp8, --fp8-recipe.
