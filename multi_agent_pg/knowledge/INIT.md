# Parameter Golf — Multi-Agent Auto-Research

> **Purpose**: canonical rules-only context, injected at the top of every
> specialist SDK session. This file is stable. Dynamic state (leaderboard,
> research tree, tried ideas) lives on the blackboard — read
> `LEADERBOARD.md`, `KNOWLEDGE.md`, `results.tsv` for state; read this for
> rules.

---

## Challenge

Train the best language model that fits in a **≤ 16 MB artifact** (code
bytes + compressed model bytes) and trains in **≤ 600 s** on 8 GPUs SXM.
Evaluation is a separate **≤ 600 s** budget on the same 8 GPUs.

**Primary metric**: `val_bpb` — validation bits-per-byte on the FineWeb
validation set, sliding window at stride 64. Lower is better.

---

## Hard rules (non-negotiable)

1. **Artifact size** ≤ 16 000 000 bytes. Code bytes use lzma-RAW + base85
   (`pack_submission.pack_code`); model bytes are GPTQ-int6 + byte-shuffle
   brotli-11. Target ≤ 15 950 000 B to leave headroom.
2. **Training** ≤ 600 s wall-clock on 8 GPUs SXM. `torch.compile` does
   NOT count — only time inside the training loop. `train_gpt.py` self-caps
   via `MAX_WALLCLOCK_SECONDS=600`.
3. **Eval** ≤ 600 s wall-clock. GPTQ calibration + quantization + sliding
   window + TTT all count against this budget.
4. **Data legality**: GPTQ calibration, quantization, eval must not read
   training or validation tokens. Self-generated calibration is legal. TTT
   is legal only when scoring precedes training on each chunk under
   `torch.inference_mode()`.
5. **Self-contained at eval**: no network, no downloads.

---

## Dataset

Only **SP8192** is on disk (`vocab_size=8192`). Changing `vocab_size` or
the tokenizer path crashes at data-load. Flag other vocab sizes as
follow-up hypotheses; do not submit.

---

## Workflow each session

Every specialist session has the same shape:

1. Read the blackboard: `LEADERBOARD.md`, `KNOWLEDGE.md`, recent
   `results.tsv` rows. The best train_gpt.py is snapshotted under
   `snapshots/<exp_id>_<domain>/train_gpt.py`.
2. Pick a hypothesis in your domain (see domain preamble). Prefer
   non-obvious ideas over parameter sweeps — the blackboard shows what's
   already been tried.
3. Edit `train_gpt.py` in your workdir. Toolchain: optionally
   `rebase_to(exp_id, workdir)` to fork from a non-best snapshot, then
   `Read(train_gpt.py)` to see the current state, then `Edit(path,
   old_string, new_string)` for exact-string replacements. You cannot
   create new files — only mutate train_gpt.py. Keep each edit focused
   on one hypothesis.
4. Call `submit_trial(specialist, hypothesis, expected_delta, parent_exp,
   notes)`. The harness runs syntax/size preflight, dispatches to 8 GPUs
   via the scheduler, blocks until terminal, parses the result, writes a TSV row, and
   returns it.
5. Read the returned row. One submit is a complete session — the
   default is to stop here. Go back to step 3 ONLY if the row points
   to a specific next edit worth trying; a crash, an uninformative
   result, or the absence of a clear refinement means stop. Every
   `submit_trial` call writes its own TSV row — the harness does not
   cap the count.

---

## Status values and what they mean

| Status | Meaning | Action |
|---|---|---|
| `keep` | val_bpb < best + proxy_threshold | noted on leaderboard; snapshot retained |
| `discard` | val_bpb ≥ best | run was valid but not an improvement |
| `size_blocked` | artifact > 16 MB (pre- or post-run) | shrink params / compression |
| `preflight_crash` | syntax error or head-side crash | fix edit and re-propose |
| `crash` | GPU-side run crashed | read crash excerpt in notes |
| `train_budget_overrun` | train_s > 600 s | self-cap broken by edit |
| `eval_budget_overrun` | eval_s > 600 s (outer timeout) | cut GPTQ calib / TTT cost |

---

## Time budgeting inside `train_gpt.py`

Everything schedule-related is **time-fractional**, not step-based.
Step speed differs between environments and cannot be predicted.

| Control | Env var | Meaning |
|---|---|---|
| LR warmdown length | `WARMDOWN_FRACTION` | fraction of `MAX_WALLCLOCK_SECONDS` spent in warmdown (default 0.58) |
| Muon momentum ramp | `MUON_MOMENTUM_WARMUP_FRACTION` | fraction for momentum ramp (default 0.22) |
| SWA cadence | `SWA_SAMPLE_INTERVAL_FRACTION` | fraction between SWA snapshots (default 0.0072) |
| LAWA cadence | `LAWA_INTERVAL_FRACTION` | fraction between LAWA snapshots (default 0.0145) |

Do not reintroduce step-count envs. `WARMUP_STEPS=20` is the one
intentional exception (fixed-step warm-up for loss stabilization).

---

## Env vars managed by the harness (do NOT toggle via hypothesis)

The harness + `run_trial.sh` inject these per trial. Flipping an
`os.environ.get(..., "default")` literal in source does nothing.

| Env var | Value | Notes |
|---|---|---|
| `TTT_ENABLED` | `1` | leaderboard parity |
| `MAX_WALLCLOCK_SECONDS` | `600` | train self-cap |
| `PYTHONUNBUFFERED` | `1` | line-flushed logs |
| `SMOKE_TEST` | `1` for preflight only | stripped for real run |

To actually disable TTT, edit the TTT pipeline code; do not propose env
flips.

---

## Late QAT — `torch.compile` hazard

`CastedLinear._qat_enabled` starts `False`. Under `torch.compile` the
first traced graph may constant-fold the STE branch away. When the flag
flips `True` mid-training the recompiled graph must include the STE path.
Always verify (log on first activation). Keep the flag visible to
`torch.compile` as a tensor or graph break, not a Python bool.

---

## Coordination with other specialists

- Other specialists are running in parallel, each with their own workdir.
  You are not the only agent in this run.
- The blackboard's `results.tsv` is shared; read recent rows before
  proposing to avoid duplicating an in-flight idea.
- `KNOWLEDGE.md` aggregates the tree; `LEADERBOARD.md` shows current
  best. Both are regenerated on every `keep`.
- Never write to `results.tsv` directly. `submit_trial` does this under
  the filelock.
