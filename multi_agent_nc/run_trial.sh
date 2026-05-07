#!/bin/bash
# multi_agent_nc/run_trial.sh — 3-phase NanoChat-d12 miniseries trial
#
# Phases:
#   1. preflight (smoke):  SMOKE_TEST=1 → 3-iter shake-out, no eval, no CORE
#   2. real run:           SMOKE_TEST=0 → full pretrain via experiment.py
#   3. classify:           validate JSONL or synthesize crash row
#
# Output layout (PG-shape so core's pull/cleanup/snapshot work unchanged):
#   $WORKDIR/full_eval_results/$(basename $WORKDIR)/run_seed${SEED}.jsonl
#   $WORKDIR/full_eval_results/$(basename $WORKDIR)/run_seed${SEED}.log

set -euo pipefail

WORKDIR="${1:?usage: run_trial.sh <workdir>}"
WORKDIR_NAME="$(basename "$WORKDIR")"
RESULT_DIR="$WORKDIR/full_eval_results/$WORKDIR_NAME"
mkdir -p "$RESULT_DIR"

SEED="${MAGENT_SEED:-0}"
_REPO_ROOT="${MAGENT_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
NANOCHAT_BASE_SRC="${NANOCHAT_BASE_DIR:-$_REPO_ROOT/data/nanochat}"

# Default 8xH100; set NPROC_PER_NODE=1 for single-GPU dev runs.
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NC_PREFLIGHT_NPROC="${NC_PREFLIGHT_NPROC:-1}"
NC_PREFLIGHT_TIMEOUT_S="${NC_PREFLIGHT_TIMEOUT_S:-2400}"
NC_REAL_TIMEOUT_S="${NC_REAL_TIMEOUT_S:-5400}"

# Optional venv activation. Default points at <repo>/data/nanochat/venv/
# (a torch + nanochat-required wheels environment). Override via
# MAGENT_NC_VENV; set MAGENT_NC_VENV=skip to use the calling shell's
# Python directly.
VENV="${MAGENT_NC_VENV:-$_REPO_ROOT/data/nanochat/venv}"
if [[ "$VENV" != "skip" && -d "$VENV" ]]; then
    . "$VENV/bin/activate"
elif [[ "$VENV" != "skip" ]]; then
    echo "[run_trial] WARN: MAGENT_NC_VENV=$VENV not found, falling back to system python (torch may be missing)" >&2
fi

cd "$WORKDIR"
export WORKDIR

# Resolve a usable Python interpreter. NGC and many cluster images ship
# only `python3` in PATH (no `python` symlink). Probe both.
PYTHON="$(command -v python || command -v python3 || true)"
if [[ -z "$PYTHON" ]]; then
    echo "[run_trial] FATAL: neither 'python' nor 'python3' found in PATH" >&2
    exit 1
fi

# Hard-fail early if data assets are missing.
if [[ ! -d "$NANOCHAT_BASE_SRC/tokenizer" || ! -d "$NANOCHAT_BASE_SRC/base_data_climbmix" || ! -d "$NANOCHAT_BASE_SRC/eval_bundle" ]]; then
    echo "[run_trial] FATAL: $NANOCHAT_BASE_SRC missing one of {tokenizer, base_data_climbmix, eval_bundle}" >&2
    "$PYTHON" run_classify.py --preflight-status crash \
        --train-rc 1 \
        --out "$RESULT_DIR/run_seed${SEED}.jsonl"
    exit 0
fi

# /dev/shm staging is MANDATORY for the data side. Reading climbmix
# shards directly from a network filesystem is empirically insufficient
# for training-pace reads under multi-tenant contention.
#
# Only DATA assets (tokenizer / shards / eval_bundle) stage to /dev/shm.
# The vendor/ source code lives in the workdir (per-trial writable copy
# seeded by _stage_workdir), so we do NOT stage vendor to /dev/shm;
# experiment.py's PYTHONPATH points at workdir/vendor/nanochat.
NANOCHAT_BASE_SHM="/dev/shm/nanochat_base_dir"
STAGE_SENTINEL="$NANOCHAT_BASE_SHM/.stage_complete"
if [[ ! -f "$STAGE_SENTINEL" ]]; then
    echo "[run_trial] staging data assets $NANOCHAT_BASE_SRC -> $NANOCHAT_BASE_SHM (no sentinel)" >&2
    rm -rf "$NANOCHAT_BASE_SHM"
    mkdir -p "$NANOCHAT_BASE_SHM"
    # Prefer rsync; fall back to `cp -a` for images that lack rsync.
    _stage_dir() {
        if command -v rsync >/dev/null 2>&1; then
            rsync -a "$1/" "$2/"
        else
            mkdir -p "$2"
            cp -a "$1/." "$2/"
        fi
    }
    if ! command -v rsync >/dev/null 2>&1; then
        echo "[run_trial] rsync not found, using 'cp -a' fallback for staging" >&2
    fi
    # Only the read-only data assets; vendor source comes from workdir.
    for sub in base_data_climbmix tokenizer eval_bundle; do
        if [[ -d "$NANOCHAT_BASE_SRC/$sub" ]]; then
            _stage_dir "$NANOCHAT_BASE_SRC/$sub" "$NANOCHAT_BASE_SHM/$sub"
        else
            echo "[run_trial] WARN: $NANOCHAT_BASE_SRC/$sub missing" >&2
        fi
    done
    touch "$STAGE_SENTINEL"
fi
export NANOCHAT_BASE_DIR="$NANOCHAT_BASE_SHM"

# ── HuggingFace network blackout ─────────────────────────────────────────────
# nanochat/flash_attention.py uses HF Hub `kernels` package to lazy-fetch the
# FA3 kernel (`varunneal/flash-attention-3`) at import time. On offline nodes
# this turns into multi-minute HEAD-request retry storms across 8 ranks every
# rerun.
# Belt-and-suspenders: force the gate at the BASH level (before any python
# imports), not just inside experiment.py — by the time experiment.py
# setdefault runs, a transitive import in a `python -c` probe may already
# have triggered `kernels.get_kernel(...)`. HF_HUB_OFFLINE=1 also short-
# circuits any other accidental HF Hub calls (datasets, tokenizers).
# Operators who pre-cached FA3 into a shared HF_HOME and want to use it can
# `export NANOCHAT_DISABLE_REMOTE_FA3=0 HF_HUB_OFFLINE=0` before run_trial.sh.
export NANOCHAT_DISABLE_REMOTE_FA3="${NANOCHAT_DISABLE_REMOTE_FA3:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# ── Phase 1: preflight ───────────────────────────────────────────────────────
# SMOKE_TEST=1 → 3-iter single-GPU run, no eval, no CORE, no torch.compile,
# no final checkpoint save. This is a cheap correctness gate for imports,
# tokenizer/data loader, optimizer step, and result JSON wiring. The real
# path below still uses NPROC_PER_NODE (8 GPUs by default) and catches full
# DDP behavior.
#
# `--signal=TERM --kill-after=30` (PG-aligned): SIGTERM at deadline, escalate
# to SIGKILL 30s later if torchrun children ignore SIGTERM (CUDA hangs do
# this). PYTHONUNBUFFERED=1 so last few hundred lines of stdout flush even
# under SIGKILL — kill_reason extraction depends on the traceback being on
# disk.
echo "[run_trial] phase 1: preflight" >&2
# experiment.py writes BOTH the run_seed${SEED}.log (full child stdout) AND
# the run_seed${SEED}.jsonl (status row) into $RESULT_DIR. We capture its
# own stdout/stderr to a separate preflight_outer.log just for shell-level
# tracing; we DO NOT overwrite the result log with it.
t_pre=$(date +%s)
set +e
timeout --signal=TERM --kill-after=30 "$NC_PREFLIGHT_TIMEOUT_S" \
    env SMOKE_TEST=1 NPROC_PER_NODE="$NC_PREFLIGHT_NPROC" PYTHONUNBUFFERED=1 \
        "$PYTHON" experiment.py > preflight_outer.log 2>&1
preflight_rc=$?
set -e
pre_elapsed=$(( $(date +%s) - t_pre ))
echo "[run_trial] preflight rc=$preflight_rc elapsed=${pre_elapsed}s" >&2
if [[ $preflight_rc -ne 0 ]]; then
    echo "[run_trial] preflight FAILED, classify-as-crash" >&2
    # Do NOT cp preflight_outer.log over the result log — experiment.py
    # already wrote the real child stdout to $RESULT_DIR/run_seed${SEED}.log
    # AND the row JSONL with kill_reason. classifier preserves both.
    "$PYTHON" run_classify.py \
        --preflight-status crash \
        --log "$RESULT_DIR/run_seed${SEED}.log" \
        --jsonl "$RESULT_DIR/run_seed${SEED}.jsonl" \
        --train-rc $preflight_rc \
        --out "$RESULT_DIR/run_seed${SEED}.jsonl"
    exit 0
fi

# ── Phase 2: real trial ──────────────────────────────────────────────────────
# Wall cap = 90 min (5400 s). Phase-1 calibration may revise.
#
# CRITICAL: delete preflight JSONL so classify pass-through can't read the
# smoke=true row. Explicit SMOKE_TEST=0 to defend against env leak.
echo "[run_trial] phase 2: real run (after rm + SMOKE_TEST=0)" >&2
# Delete preflight artifacts so classifier can't pass-through the smoke row.
rm -f "$RESULT_DIR/run_seed${SEED}.jsonl" "$RESULT_DIR/run_seed${SEED}.log"
t_run=$(date +%s)
set +e
timeout --signal=TERM --kill-after=30 "$NC_REAL_TIMEOUT_S" \
    env SMOKE_TEST=0 PYTHONUNBUFFERED=1 \
        "$PYTHON" experiment.py > train_outer.log 2>&1
train_rc=$?
set -e
run_elapsed=$(( $(date +%s) - t_run ))
echo "[run_trial] train rc=$train_rc elapsed=${run_elapsed}s" >&2

# ── Phase 3: classify ────────────────────────────────────────────────────────
# experiment.py wrote $RESULT_DIR/run_seed${SEED}.log (real child stdout)
# AND $RESULT_DIR/run_seed${SEED}.jsonl. Classify reads from those directly.
echo "[run_trial] phase 3: classify (train_rc=$train_rc)" >&2
"$PYTHON" run_classify.py \
    --log "$RESULT_DIR/run_seed${SEED}.log" \
    --train-rc $train_rc \
    --jsonl "$RESULT_DIR/run_seed${SEED}.jsonl" \
    --out "$RESULT_DIR/run_seed${SEED}.jsonl"
