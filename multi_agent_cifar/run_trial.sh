#!/bin/bash
# multi_agent_cifar/run_trial.sh — N-seed CIFAR-10 airbench96 trial (v2).
#
# Phases:
#   1. preflight (smoke): SMOKE_TEST=1 RUNS=1 → airbench96.py 1-epoch shake-out
#   2. multi-seed train:  for SEED in 0..N-1: SMOKE_TEST=0 RUNS=1 → 1 trial
#                         each writes run_seed${SEED}.jsonl
#   3. classify-aggregate: run_classify.py reads N jsonls, computes
#                         {mean_acc, mean_train_s, std, ...}, writes single
#                         aggregated row for the blackboard at run_seed0.jsonl
#
# Output layout (PG-shape so core's pull/cleanup/snapshot work unchanged):
#   $WORKDIR/full_eval_results/$(basename $WORKDIR)/run_seed0.jsonl
#       (aggregated row — what the harness reads)
#   $WORKDIR/full_eval_results/$(basename $WORKDIR)/run_seed{0..N-1}.log
#       (per-seed train logs — debugging only)
#
# v2 metric semantics: train_s is the score (lower better); mean_acc ≥ 0.96
# is the threshold gate. See multi_agent_cifar/task_config.py for taxonomy.

set -euo pipefail

WORKDIR="${1:?usage: run_trial.sh <workdir>}"
WORKDIR_NAME="$(basename "$WORKDIR")"
RESULT_DIR="$WORKDIR/full_eval_results/$WORKDIR_NAME"
mkdir -p "$RESULT_DIR"

# v2: N seeds per trial. Default 10 (proxy for upstream's n=200). Override
# by exporting MAGENT_N_SEEDS — calibrate uses 10, verify_candidate uses
# 30 (longer + lower-noise validation pass).
N_SEEDS="${MAGENT_N_SEEDS:-10}"

# CIFAR-10 dataset. Default points under <repo>/data/cifar/data/, and
# the harness stages it into /dev/shm on first run for sub-second
# loader access (CIFAR-10 is 178 MB so the rsync is ~2 s).
_REPO_ROOT="${MAGENT_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CIFAR_DATA_SRC="${MAGENT_CIFAR_DATA_DIR:-$_REPO_ROOT/data/cifar/data}"
CIFAR_DATA_SHM="/dev/shm/cifar10-airbench-data"
STAGE_SENTINEL="$CIFAR_DATA_SHM/.stage_complete"
if [[ ! -f "$STAGE_SENTINEL" ]]; then
    echo "[run_trial] staging $CIFAR_DATA_SRC -> $CIFAR_DATA_SHM (no sentinel)" >&2
    rm -rf "$CIFAR_DATA_SHM"
    mkdir -p "$CIFAR_DATA_SHM"
    if command -v rsync >/dev/null 2>&1; then
        rsync -a "$CIFAR_DATA_SRC/" "$CIFAR_DATA_SHM/"
    else
        echo "[run_trial] rsync not found, using 'cp -a' fallback" >&2
        cp -a "$CIFAR_DATA_SRC/." "$CIFAR_DATA_SHM/"
    fi
    touch "$STAGE_SENTINEL"
fi
export CIFAR_DATA_DIR="$CIFAR_DATA_SHM"

# Single-GPU pinning. 7 GPUs idle by design (CIFAR uses 1 of 8).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Optional venv activation. Default points under <repo>/data/cifar/venv/.
# Override via MAGENT_CIFAR_VENV; "skip" disables.
VENV="${MAGENT_CIFAR_VENV:-$_REPO_ROOT/data/cifar/venv}"
if [[ "$VENV" != "skip" && -d "$VENV" ]]; then
    . "$VENV/bin/activate"
elif [[ "$VENV" != "skip" ]]; then
    echo "[run_trial] WARN: MAGENT_VENV=$VENV not found, using system python (torch may be missing)" >&2
fi

cd "$WORKDIR"
export WORKDIR

# Resolve Python interpreter (NGC images ship python3, no python symlink).
PYTHON="$(command -v python || command -v python3 || true)"
if [[ -z "$PYTHON" ]]; then
    echo "[run_trial] FATAL: neither 'python' nor 'python3' found in PATH" >&2
    exit 1
fi

# ── Phase 1: preflight ───────────────────────────────────────────────────────
# SMOKE_TEST=1 + RUNS=1 = 1 seed × 1 epoch shake-out. Catches import errors,
# shape mismatches, missing data files BEFORE we burn N×real-run time.
# v2 budget: airbench96's wider net (~30M params) + cutout + EMA recipe
# triggers a substantially longer torch.compile(max-autotune) than
# airbench94 — empirically ~3-5 min cold on GPU node. 600s wall covers
# compile + 1-epoch train; if compile is shorter the script just exits
# faster, no penalty. Override via MAGENT_PREFLIGHT_TIMEOUT_S env.
PREFLIGHT_TIMEOUT_S="${MAGENT_PREFLIGHT_TIMEOUT_S:-600}"
echo "[run_trial] phase 1: preflight (smoke), timeout=${PREFLIGHT_TIMEOUT_S}s" >&2
t_pre=$(date +%s)
set +e
timeout --signal=TERM --kill-after=30 "$PREFLIGHT_TIMEOUT_S" \
    env SMOKE_TEST=1 RUNS=1 MAGENT_SEED=0 PYTHONUNBUFFERED=1 \
        "$PYTHON" airbench96.py > preflight.log 2>&1
preflight_rc=$?
set -e
pre_elapsed=$(( $(date +%s) - t_pre ))
echo "[run_trial] preflight rc=$preflight_rc elapsed=${pre_elapsed}s" >&2
if [[ $preflight_rc -ne 0 ]]; then
    echo "[run_trial] preflight FAILED → classify as preflight_crash" >&2
    cp preflight.log "$RESULT_DIR/run_seed0.log"
    "$PYTHON" run_classify.py \
        --preflight-status crash \
        --logs preflight.log \
        --train-rc $preflight_rc \
        --n-seeds 0 \
        --out "$RESULT_DIR/run_seed0.jsonl"
    exit 0
fi

# Critical: delete the SMOKE jsonl + log so phase-3 aggregation can't pass
# them through. Real seeds will write fresh files.
rm -f "$RESULT_DIR"/run_seed*.jsonl "$RESULT_DIR"/run_seed*.log
rm -f preflight.log

# ── Phase 2: N-seed real run ────────────────────────────────────────────────
# Each SEED writes its own run_seed${SEED}.jsonl. We accumulate all rcs.
# v2 per-seed budget: airbench96 cold compile is ~3-5 min on GPU node,
# real train ~14 s. 600 s wall per seed is enough for cold compile +
# ~30s overhead margin. Cache hits on seed 1+ make the actual mean much
# lower — compile is paid per process so first seed dominates total.
SEED_TIMEOUT_S="${MAGENT_SEED_TIMEOUT_S:-600}"
echo "[run_trial] phase 2: real run, N=$N_SEEDS seeds, per-seed timeout=${SEED_TIMEOUT_S}s" >&2
t_run=$(date +%s)
all_rc=0
for ((SEED=0; SEED<N_SEEDS; SEED++)); do
    # Shell-side timing is the AUTHORITATIVE train_s — agent cannot edit it.
    # We use python's monotonic clock for sub-second precision (the train
    # itself is ~14s, integer-second resolution would alias trial gaps).
    t_seed_start_ns=$(date +%s%N)
    set +e
    timeout --signal=TERM --kill-after=30 "$SEED_TIMEOUT_S" \
        env SMOKE_TEST=0 RUNS=1 MAGENT_SEED=$SEED PYTHONUNBUFFERED=1 \
            "$PYTHON" airbench96.py > "train_seed${SEED}.log" 2>&1
    seed_rc=$?
    set -e
    t_seed_end_ns=$(date +%s%N)
    # Elapsed in seconds with 3-digit precision (millisecond level).
    seed_elapsed_s=$(awk -v a="$t_seed_start_ns" -v b="$t_seed_end_ns" 'BEGIN{printf "%.3f", (b-a)/1e9}')
    echo "[run_trial]   seed $SEED rc=$seed_rc shell_elapsed_s=$seed_elapsed_s" >&2
    cp "train_seed${SEED}.log" "$RESULT_DIR/run_seed${SEED}.log"
    # Sidecar file: harness-authoritative timing for run_classify.
    # NOT in airbench96.py's reach, NOT removable by recipe edits.
    echo "$seed_elapsed_s" > "$RESULT_DIR/run_seed${SEED}.shell_elapsed_s"
    echo "$seed_rc" > "$RESULT_DIR/run_seed${SEED}.shell_rc"
    if [[ $seed_rc -ne 0 ]]; then
        all_rc=$seed_rc
        # Continue with remaining seeds — partial success is informative.
    fi
done
run_elapsed=$(( $(date +%s) - t_run ))
echo "[run_trial] all $N_SEEDS seeds done, last_rc=$all_rc elapsed=${run_elapsed}s" >&2

# ── Phase 3: classify-aggregate ──────────────────────────────────────────────
# run_classify reads run_seed{0..N-1}.jsonl, computes mean/std + acc gate,
# writes a single aggregated row to run_seed0.jsonl (overwriting seed-0
# per-seed jsonl with the AGGREGATED row — this is what the harness reads).
echo "[run_trial] phase 3: classify-aggregate (train_rc=$all_rc, N=$N_SEEDS)" >&2
"$PYTHON" run_classify.py \
    --logs "train_seed*.log" \
    --train-rc $all_rc \
    --jsonl-glob "$RESULT_DIR/run_seed*.jsonl" \
    --n-seeds $N_SEEDS \
    --out "$RESULT_DIR/run_seed0.jsonl"
