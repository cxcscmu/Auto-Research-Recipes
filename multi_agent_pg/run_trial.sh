#!/bin/bash
# run_trial.sh: one-shot GPU-node entrypoint for a single trial.
#
# Invoked by the scheduler with one positional argument (the workdir).
# Expects the specialist's workdir to already contain:
#   train_gpt.py              the mutated (or baseline-seeded) source
#   pack_submission.py        verbatim copy from multi_agent_pg/tools/
#   run_classify.py           verbatim copy from multi_agent_pg/tools/
#
# Behaviour:
#   1. cd to $WORKDIR and activate venv.
#   2. Stage SP8192 data into /dev/shm/fineweb10B_sp8192 (one rsync per
#      host; subsequent trials on the same host are no-ops). Symlink
#      it into ./data/fineweb10B_sp8192 so train_gpt.py finds it.
#   3. Preflight (SMOKE_TEST=1, <=300s): launches the same `torchrun
#      --nproc_per_node=8` form as the real run so world_size>1
#      shape/type bugs fire here, not after burning a full train. Parses
#      `smoke_pack_bytes: total=N` out of stdout as the authoritative
#      pre-run size gate (over 16 MB -> size_blocked, exit early).
#   4. Real run (<=2200s outer timeout): MAX_WALLCLOCK_SECONDS=600
#      TTT_ENABLED=1, torchrun nproc=8. train_gpt.py self-caps train at
#      600s; run_trainer.py self-caps eval at 600s; the outer
#      `timeout 2200` backstop leaves ~1000s headroom for compile.
#   5. Pack: `python pack_submission.py` produces
#      ckpt/train_gpt_packed.py and prints `Submission size: N bytes`.
#   6. Classify: `python run_classify.py <combined.log>` emits the
#      TSV-compatible status / val_bpb / artifact_bytes / train_s / eval_s
#      line consumed by harness/tracker.parse_validate_result.
#
# Output layout under $WORKDIR/full_eval_results/<workdir-basename>/:
#     run_seed0.log              full preflight + train + pack combined log
#     run_seed0.jsonl            one-line classified result
# Plus (from the real run):
#     ckpt/final_model.int6.ptz  quantized model blob
#     ckpt/train_gpt_packed.py   self-extracting packed code
#
# Exit code is always 0 when the jsonl is written; status is carried on
# the classified line so the harness can distinguish run outcomes.
#
# Required environment (see multi_agent_pg/README.md for setup):
#   MAGENT_PG_VENV         path to a Python venv with FA3-capable torch
#   MAGENT_PG_DATA_DIR     path to the SP8192 fineweb10B dataset
#   MAGENT_PG_TOKENIZER_DIR  path to the SP8192 tokenizer assets

set -euo pipefail

WORKDIR="${1:?usage: run_trial.sh <workdir>}"

# Resolve repository root so default paths point under ./data/parameter_golf/.
_REPO_ROOT="${MAGENT_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
_DATA_BASE="$_REPO_ROOT/data/parameter_golf"

VENV="${MAGENT_PG_VENV:-$_DATA_BASE/venv}"
DATA_SRC="${MAGENT_PG_DATA_DIR:-$_DATA_BASE/fineweb10B_sp8192}"
DATA_DST="/dev/shm/fineweb10B_sp8192"
SEED="${MAGENT_SEED:-0}"
# Propagate to torch RNG only when MAGENT_SEED was explicitly set (e.g. by
# verify_candidate.py with --seeds-list). Swarm trials never set MAGENT_SEED,
# so SEED env stays unset and train_gpt.py keeps its default 42 (see
# multi_agent/train_gpt.py: seed = int(os.environ.get("SEED", 42))). This
# preserves swarm behaviour; verify gets real per-seed RNG control.
if [[ -n "${MAGENT_SEED+x}" ]]; then
    export SEED
fi

# Tunables (all mirror single_agent defaults)
PREFLIGHT_TIMEOUT_S=300
TRIAL_TIMEOUT_S=2200
SIZE_LIMIT_BYTES=16000000

OUT_DIR="$WORKDIR/full_eval_results/$(basename "$WORKDIR")"
PREFLIGHT_LOG="$OUT_DIR/preflight.log"
TRAIN_LOG="$OUT_DIR/train.log"
PACK_LOG="$OUT_DIR/pack.log"
COMBINED_LOG="$OUT_DIR/run_seed${SEED}.log"
JSONL="$OUT_DIR/run_seed${SEED}.jsonl"

echo "[run_trial] workdir=$WORKDIR venv=$VENV data_src=$DATA_SRC seed=$SEED"
mkdir -p "$OUT_DIR"
cd "$WORKDIR"

# shellcheck disable=SC1091
. "$VENV/bin/activate"

# Stage data into tmpfs on first trial per node; later trials reuse it.
# Gate on a sentinel file written *after* rsync completes, so any partial
# stage (node evicted / SIGTERM mid-rsync) is detected and healed by the
# next trial. Directory-existence alone is NOT enough — a prior trial
# observed dir+val present but train shards missing → `train_shards: 0`.
STAGE_SENTINEL="$DATA_DST/.stage_complete"
if [[ ! -f "$STAGE_SENTINEL" ]]; then
  echo "[run_trial] staging SP8192 → $DATA_DST (no sentinel, re-staging) …"
  rm -rf "$DATA_DST"
  mkdir -p "$DATA_DST"
  rsync -a "$DATA_SRC/" "$DATA_DST/"
  touch "$STAGE_SENTINEL"
fi
mkdir -p data
rm -f data/fineweb10B_sp8192
ln -s "$DATA_DST" data/fineweb10B_sp8192

# train_gpt.py defaults read DATA_DIR=./openai_parameter_golf/data for
# both the dataset files AND the tokenizer. The dataset portion we
# redirect to the tmpfs-staged copy via TRAIN_FILES/VAL_FILES; the
# tokenizer is small and stays on the source filesystem via a direct
# symlink into the expected subtree.
TOKENIZER_SRC="${MAGENT_PG_TOKENIZER_DIR:-$_DATA_BASE/tokenizers}"
mkdir -p openai_parameter_golf/data
rm -f openai_parameter_golf/data/tokenizers
ln -s "$TOKENIZER_SRC" openai_parameter_golf/data/tokenizers
export TRAIN_FILES="$DATA_DST/fineweb_train_*.bin"
export VAL_FILES="$DATA_DST/fineweb_val_*.bin"

# ── Phase 1: Preflight (SMOKE_TEST=1) ───────────────────────────────────────
echo "[run_trial] preflight (SMOKE_TEST=1, ≤${PREFLIGHT_TIMEOUT_S}s) …"
t_pre=$(date +%s)
set +e
timeout --signal=TERM --kill-after=10 "${PREFLIGHT_TIMEOUT_S}" \
    env SMOKE_TEST=1 PYTHONUNBUFFERED=1 \
        torchrun --standalone --nproc_per_node=8 \
                 --redirects 0 --tee 1 --log-dir /tmp \
                 train_gpt.py \
        > "$PREFLIGHT_LOG" 2>&1
preflight_rc=$?
set -e
pre_elapsed=$(( $(date +%s) - t_pre ))
echo "[run_trial] preflight rc=$preflight_rc elapsed=${pre_elapsed}s"

# Grep authoritative smoke pack bytes.
smoke_total=$(grep -E '^smoke_pack_bytes:' "$PREFLIGHT_LOG" | tail -n1 \
              | sed -E 's/.*total=([0-9]+).*/\1/' || true)

# Decide whether to proceed.
preflight_status="ok"
if [[ $preflight_rc -ne 0 ]]; then
  preflight_status="crash"
fi
if [[ -n "$smoke_total" && "$smoke_total" -gt $SIZE_LIMIT_BYTES ]]; then
  preflight_status="size_blocked"
fi

if [[ "$preflight_status" != "ok" ]]; then
  # Abort: emit combined log + classified jsonl, skip the real run.
  cp "$PREFLIGHT_LOG" "$COMBINED_LOG"
  python run_classify.py \
      --log "$COMBINED_LOG" \
      --preflight-status "$preflight_status" \
      --smoke-bytes "${smoke_total:-0}" \
      --pre-elapsed "$pre_elapsed" \
      --size-limit "$SIZE_LIMIT_BYTES" \
      --out "$JSONL"
  echo "[run_trial] preflight aborted ($preflight_status) — jsonl written"
  exit 0
fi

# ── Phase 2: Real run ───────────────────────────────────────────────────────
# Budgets: train ≤600s (self-capped in train_gpt.py, compile excluded via
# WARMUP_STEPS=20), eval ≤600s (enforced by run_trainer.py — polls stdout,
# SIGTERMs the torchrun process group at post-train-elapsed > 600s and emits
# `--- EVAL_TIMEOUT after 600s ---`). Outer `timeout ${TRIAL_TIMEOUT_S}` is
# the second safety net in case run_trainer.py itself hangs.
EVAL_BUDGET="${MAGENT_EVAL_BUDGET:-600}"
echo "[run_trial] launching real train (train ≤600s, eval ≤${EVAL_BUDGET}s, outer ≤${TRIAL_TIMEOUT_S}s) …"
t_run=$(date +%s)
set +e
timeout --signal=TERM --kill-after=30 "${TRIAL_TIMEOUT_S}" \
    env MAX_WALLCLOCK_SECONDS=600 TTT_ENABLED=1 PYTHONUNBUFFERED=1 \
        python run_trainer.py \
            --eval-budget "${EVAL_BUDGET}" \
            --total-timeout "${TRIAL_TIMEOUT_S}" \
            -- torchrun --standalone --nproc_per_node=8 \
                        --redirects 0 --tee 1 --log-dir /tmp \
                        train_gpt.py \
        > "$TRAIN_LOG" 2>&1
train_rc=$?
set -e
run_elapsed=$(( $(date +%s) - t_run ))
echo "[run_trial] train rc=$train_rc elapsed=${run_elapsed}s"

# Outer-timeout detection: bash `timeout` returns 124/137 on backstop kill.
# run_trainer.py also returns 124 on either its eval-cap or total-timeout
# fire — distinguish via the EVAL_TIMEOUT / OUTER_TIMEOUT markers it wrote
# into the log. run_classify.py greps for both.
outer_timeout=0
if [[ $train_rc -eq 124 || $train_rc -eq 137 ]]; then
  outer_timeout=1
  # Only append the bash-level marker if run_trainer.py did NOT already
  # emit its own EVAL_TIMEOUT/OUTER_TIMEOUT line (avoids double-marker).
  if ! grep -qE '^---[[:space:]]*(EVAL|OUTER)_TIMEOUT\b' "$TRAIN_LOG"; then
    echo "[run_trial] --- OUTER_TIMEOUT after ${TRIAL_TIMEOUT_S}s ---" >> "$TRAIN_LOG"
  fi
fi

# ── Phase 3: Pack ───────────────────────────────────────────────────────────
pack_rc=0
if [[ $train_rc -eq 0 && -f "ckpt/final_model.int6.ptz" ]]; then
  echo "[run_trial] packing submission …"
  set +e
  python pack_submission.py > "$PACK_LOG" 2>&1
  pack_rc=$?
  set -e
else
  echo "[run_trial] skipping pack (train_rc=$train_rc, model_blob_present=$([[ -f ckpt/final_model.int6.ptz ]] && echo yes || echo no))"
  : > "$PACK_LOG"
fi

# ── Phase 4: Combined log + classify ────────────────────────────────────────
{
  echo "=== PREFLIGHT (SMOKE_TEST=1, ${pre_elapsed}s) ==="
  cat "$PREFLIGHT_LOG"
  echo
  echo "=== TRAIN (rc=$train_rc, ${run_elapsed}s) ==="
  cat "$TRAIN_LOG"
  echo
  echo "=== PACK (rc=$pack_rc) ==="
  cat "$PACK_LOG"
} > "$COMBINED_LOG"

python run_classify.py \
    --log "$COMBINED_LOG" \
    --preflight-status ok \
    --smoke-bytes "${smoke_total:-0}" \
    --pre-elapsed "$pre_elapsed" \
    --train-rc "$train_rc" \
    --outer-timeout "$outer_timeout" \
    --run-elapsed "$run_elapsed" \
    --pack-rc "$pack_rc" \
    --size-limit "$SIZE_LIMIT_BYTES" \
    --out "$JSONL"

echo "[run_trial] done — jsonl=$JSONL"
