# NanoChat-d12 miniseries — orientation

## What the swarm is searching

The d12 miniseries is the smallest variant of Karpathy's nanochat:
- 12-layer transformer, ~110 M parameters (GPT-2-small class)
- Pretrain only — NO SFT, NO RL, NO chat_eval
- Single trainer entrypoint: `torchrun -m scripts.base_train --depth 12 …`
- Single-process coordinator (`experiment.py` in your workdir) wraps that
  invocation, parses the train log, and writes the result JSONL.

The swarm searches the **pretrain recipe** — architecture, optimizer,
schedules, data shape, numerics — for **higher CORE metric at fixed
compute**. CORE is the composite eval that `scripts.base_train` computes
internally (no separate base_eval call for v1).

## Hardware reality

- 8 GPUs SXM, fully utilized via `torchrun --nproc_per_node=8`.
- Per-trial wallclock: ~30-90 min depending on `--num-iterations` ×
  batch_size; cold torch.compile adds ~1-3 min.
- run_trial wall caps: preflight ≤ 10 min, real run ≤ 90 min.

## Pipeline shape

Single-process coordinator:

1. Preflight (smoke): `SMOKE_TEST=1 python experiment.py` runs
   `--num-iterations 20 --device-batch-size 1 --total-batch-size 4096
   --max-seq-len 512 --core-metric-every -1 --eval-every -1`. Catches
   missing tokenizer / data shards / OOM / syntax errors / vendor edits
   that broke imports. (4096 = 1 × 512 × 8 ranks; smaller multiples
   trip base_train's divisibility assert before any training step.)
2. Real run: `SMOKE_TEST=0 python experiment.py` runs the full d12
   pretrain at the recipe's `TRAIN_ARGS`. base_train computes val_bpb +
   core_metric internally; experiment.py parses the log and writes the
   result JSONL.
3. Classify: `run_classify.py` validates / passes-through the JSONL or
   synthesizes a crash row if experiment.py exited non-zero.

## Editable surface (v2-B)

The agent edits two regions of the workdir:

1. **`experiment.py`** — recipe coordinator. Holds `TRAIN_ARGS` (the
   list of `--flag value` pairs passed to `torchrun -m
   scripts.base_train`), wallclock cap, and SMOKE_TEST overrides.
2. **`vendor/nanochat/`** — a writable copy of the vendored nanochat
   source, staged into your workdir on iter 1. PYTHONPATH is set so
   torchrun's child ranks load the modified code. Edit any .py inside
   to change real training behaviour.

`syntax_check`, `param_count`, `read_snapshot`, `diff_snapshots`, and
`rebase_to` are all multi-file aware — they walk both `experiment.py` +
`vendor/nanochat/`.

## Vendor key files (where to look)

| file (under `vendor/nanochat/`)         | what it owns                                     |
|------------------------------------------|--------------------------------------------------|
| `nanochat/gpt.py`                        | Block / MLP / Attention; norm + head structure  |
| `nanochat/optim.py`                      | Muon (newton-schulz) + AdamW                    |
| `nanochat/dataloader.py`                 | Train + val token streams                       |
| `nanochat/tokenizer.py`                  | BPE wrapper                                      |
| `nanochat/fp8.py`                        | FP8 matmul + scaling                            |
| `nanochat/flash_attention.py`            | Sliding-window + custom kernels                 |
| `scripts/base_train.py`                  | Train loop, LR/momentum/wd schedules, wd_mask   |
| `scripts/tok_train.py`                   | Tokenizer training (operator-side, ignore)      |
| `pyproject.toml`                         | Treat as off-limits (deps frozen at pin)        |
| `../.commit_pin` (at `vendor/.commit_pin`) | Provenance — bump only on intentional rebase  |

## Metrics

- **Primary**: `core_metric` (higher is better). Baseline d12 placeholder
  ~0.30; Phase-1 calibrates the actual value.
- **Secondary**: `val_bpb` (lower is better). Recorded in TSV but
  blackboard ranks by core_metric only.
- Variance across seeds at fixed config: TBD by Phase-1; assume ~0.005
  for noise-floor judgments until measured.

## Specialists

| key   | scope                                                                |
|-------|----------------------------------------------------------------------|
| arch  | d12 transformer architecture — `vendor/nanochat/gpt.py` + window/head/dim flags |
| opt   | Muon + AdamW — `vendor/nanochat/optim.py` + LR/wd flags              |
| data  | Pretrain data — `vendor/nanochat/dataloader.py` + `tokenizer.py` + batch flags |
| sched | LR / momentum / wd / warmup-warmdown — `vendor/scripts/base_train.py` schedule logic |
| sys   | Numerics + kernels — `vendor/nanochat/fp8.py` + `flash_attention.py` + `--fp8` flags |

(`meta` analyst is intentionally not in v1: core's user-message contract
requires every specialist to submit_trial, which conflicts with an
analyst-only role. Reintroduce when a blackboard-write tool exists.)

## Hard limits

See the prompt's "Hard limits" block. Most-load-bearing constraints:

- `--depth = 12` is FIXED (defines the miniseries setting). Do NOT add
  layers or remove `--depth` plumbing in `scripts/base_train.py`.
- `pyproject.toml`, `run_classify.py`, `run_trial.sh`,
  `profile_pipeline.py`, `full_eval_results/`, `ckpt/`, `logs/` are
  off-limits — they're harness, not search surface.
- `vendor/.commit_pin`: leave alone unless you intentionally bumped to
  a new upstream sha; profile_pipeline warns if it goes missing.
- 90 min real-run wall cap → `train_budget_overrun` if exceeded.
- Tokenizer + data shards are operator-baked at `$NANOCHAT_BASE_DIR`;
  experiment.py reads from there. Don't expect to retrain the tokenizer
  inside a trial.
