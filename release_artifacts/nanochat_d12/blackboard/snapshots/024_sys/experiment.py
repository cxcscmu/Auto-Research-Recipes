"""experiment.py — single-process coordinator for d12 miniseries pretrain.

This file is one piece of the swarm's editable surface. The other piece
is the entire vendored `nanochat` source under `workdir/vendor/nanochat/`
which is staged into your workdir on iter 1 and is **writable** — your
edits propagate to all 8 torchrun ranks via the PYTHONPATH injection
below.

Pipeline:
  1. Coordinator (this file, single Python process) reads TRAIN_ARGS,
     builds an argv for `torchrun -m scripts.base_train`, and runs it
     as a subprocess. PYTHONPATH points at workdir/vendor/nanochat/ so
     child ranks load YOUR copy of the library, not the system one.
  2. base_train is the WHOLE pipeline — it computes both `val_bpb` and
     `core_metric` internally (no separate base_eval call needed).
  3. Coordinator parses the train log for the final core_metric +
     val_bpb, writes ONE JSONL line to
     full_eval_results/<basename>/run_seed${SEED}.jsonl.

Why a single Python coordinator (not torchrun on this file):
  - We only want ONE writer of the final JSONL (rank 0 spam-fights
    otherwise).
  - We can mutate `os.environ` for child processes here without
    affecting all 8 ranks via shared module state.

NEVER run this file under torchrun directly; it would spawn 8 child
torchruns, each spawning 8 base_train ranks → cluster meltdown.

# NC-FORK note (v2-B)
You may edit experiment.py AND any .py under workdir/vendor/nanochat/
(nanochat/*.py + scripts/*.py). See knowledge/INIT.md for the
per-specialist ownership map. Do NOT touch pyproject.toml,
vendor/.commit_pin, run_classify.py, run_trial.sh, or
profile_pipeline.py — those are harness/provenance, not search surface.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


# ── Environment + paths ─────────────────────────────────────────────────────

WORKDIR     = Path(os.environ.get("WORKDIR", os.getcwd())).resolve()
WORKDIR_NM  = WORKDIR.name
SEED        = int(os.environ.get("MAGENT_SEED", "0"))
BASE_DIR    = Path(os.environ.get("NANOCHAT_BASE_DIR",
                                   str(Path.home() / "ag/nanochat_base_dir")))

# v2-B: VENDOR is the LOCAL writable copy in the workdir. Each specialist
# gets its own vendor/ tree that the agent freely edits across files
# (gpt.py, optim.py, dataloader.py, base_train.py, fp8.py, ...).
# PYTHONPATH below ensures the 8 torchrun ranks load this modified code,
# NOT the the shared filesystem / /dev/shm base_dir copy. Data assets (tokenizer, shards,
# eval_bundle) still live under NANOCHAT_BASE_DIR (read-only).
VENDOR      = WORKDIR / "vendor" / "nanochat"

RESULT_DIR  = WORKDIR / "full_eval_results" / WORKDIR_NM
SMOKE_TEST  = os.environ.get("SMOKE_TEST", "0") == "1"
NPROC       = int(os.environ.get("NPROC_PER_NODE", "8"))


# ── Tunable args (specialists edit ONLY these dicts) ────────────────────────
#
# Flag names map to upstream `scripts/base_train.py` at the pinned commit.
# If you add a flag, FIRST verify it exists upstream (Read
# $NANOCHAT_BASE_DIR/nanochat/scripts/base_train.py).
#
# Smoke vs real: SMOKE_TEST=1 shrinks the run to a 20-iter shake-out so
# the preflight catches obvious crashes (data missing, tokenizer missing,
# OOM at small scale) without paying the full pretrain cost.

TRAIN_ARGS_BASELINE: dict[str, str] = {
    # Identity (DO NOT change depth — defines the miniseries setting)
    "--depth":           "12",
    "--model-tag":       WORKDIR_NM,         # uniqueness across specialists' workdirs

    # Architecture
    "--aspect-ratio":    "64",
    "--head-dim":        "128",
    "--max-seq-len":     "2048",
    # Use full-context attention for all layers (window_pattern="L").
    # With PyTorch SDPA (no FA3 available in this environment), any custom attention
    # mask (including the sliding-window bool mask used by "S" layers) forces
    # the slow math backend. Only is_causal=True (which "L" enables) routes
    # through PyTorch's Flash SDPA kernel — ~3-5x faster per attention step.
    # SSSL had 9/12 layers on the math backend; L puts all 12 on Flash.
    # Full-context attention also improves per-token quality (more context).
    "--window-pattern":  "L",

    # Optimization
    "--device-batch-size":  "32",
    "--total-batch-size":   "-1",            # auto from data:param ratio
    "--matrix-lr":          "0.02",
    "--scalar-lr":          "0.5",
    "--embedding-lr":       "0.3",
    "--unembedding-lr":     "0.008",
    "--weight-decay":       "0.28",
    "--warmup-steps":       "40",
    "--warmdown-ratio":     "0.65",
    "--final-lr-frac":      "0.05",

    # Precision: keep BF16 (FP8 tensorwise scaling degraded quality in exp_004,
    # -0.017 core_metric vs BF16 baseline).
    "--fp8":                       False,

    # Horizon: training longer significantly improves quality for d12.
    # exp_007 (ratio=12, window=L) completed in only ~9 min on 8 GPUs,
    # leaving ~81 min of the 90 min wall budget unused.
    # Chinchilla-optimal is ratio~20; ratio=100 trains ~8x longer
    # for much better CORE metric while estimated total ~60 min (safe).
    "--num-iterations":            "-1",     # let target ratio drive
    "--target-param-data-ratio":   "130",    # push further within 90 min budget; exp_020 (ratio=100) used 61 min

    # Eval cadence (real run: every 250 steps; smoke disables both).
    # NOTE: --eval-tokens has NO -1 disable semantics upstream — passing -1
    # makes eval_steps negative, evaluate_bpb returns inf, and val_bpb
    # disappears from the TSV. Either omit (defaults to 80*524288 ≈ 42M
    # tokens) or set a finite value. We omit.
    "--eval-every":              "250",
    "--core-metric-every":       "2000",
    "--core-metric-max-per-task": "500",
    "--sample-every":            "-1",       # disable inline sampling (saves time)
    "--save-every":              "-1",       # only save at end
}

TRAIN_ARGS_SMOKE: dict[str, str] = {
    # Override a few keys for the SMOKE_TEST=1 path.
    #
    # Divisibility constraint (base_train.py:407):
    #   total_batch_size % (device_batch_size * max_seq_len * world_size) == 0
    # On 8 GPUs (NPROC=8), DBS=1, max_seq_len=512:
    #   floor = 1 * 512 * 8 = 4096 → total_batch_size must be a multiple of 4096
    # Old smoke had total_batch_size=512 → assert crash on 8 GPUs before any
    # training step ran. 4096 = 1 micro-step × 8 ranks (no grad accumulation).
    # Keep smoke genuinely cheap. It is only a launch-shape gate: imports,
    # tokenizer/data loader, DDP wiring, one optimizer step, JSONL/log path.
    # Full compile/perf/metric behavior is validated by the real run.
    "--num-iterations":          "3",
    "--target-param-data-ratio": "-1",       # disable since num_iterations is set
    "--device-batch-size":       "1",
    "--total-batch-size":        "4096",     # 1 × 512 × 8 ranks (8 GPUs default)
    "--max-seq-len":             "512",
    "--core-metric-every":       "-1",       # skip CORE in smoke
    "--eval-every":              "-1",       # skip eval in smoke
    "--save-every":              "-1",
}


def _argv_for_smoke(base: dict[str, str]) -> dict[str, str]:
    out = dict(base)
    out.update(TRAIN_ARGS_SMOKE)
    # Keep smoke to exactly one microbatch for the selected smoke world size.
    # run_trial.sh defaults smoke to one GPU; real runs still use 8 GPUs.
    nproc = int(os.environ.get("NPROC_PER_NODE", "8"))
    device_bs = int(out["--device-batch-size"])
    seq_len = int(out["--max-seq-len"])
    out["--total-batch-size"] = str(device_bs * seq_len * nproc)
    return out


def _flatten(args_d: dict) -> list[str]:
    """Convert TRAIN_ARGS dict to argv list, with store_true semantics:

    - True            → emit just the flag (e.g. `--fp8`)
    - False / None    → omit entirely (no flag, no value)
    - any other value → emit flag + str(value)

    The store_true path is needed because upstream's `--fp8` is
    `action='store_true'` — passing `--fp8 true` makes argparse choke
    ("unrecognized arguments: true").
    """
    out: list[str] = []
    for k, v in args_d.items():
        if v is True:
            out.append(k)
        elif v is False or v is None:
            continue
        else:
            out.append(k); out.append(str(v))
    return out


# ── Stdout parsing ─────────────────────────────────────────────────────────

# `Step {N:05d} | Validation bpb: {val:.6f}` — multiple per run; we want the LAST.
_BPB_RX = re.compile(r"Validation bpb:\s*([0-9.]+)")
# `Step {N:05d} | CORE metric: {core:.4f}` — last of these is the final value.
_CORE_RX = re.compile(r"CORE metric:\s*([0-9.]+)")
# Final summary line: `Minimum validation bpb: {bpb:.6f}` — useful fallback for val_bpb.
_MIN_BPB_RX = re.compile(r"Minimum validation bpb:\s*([0-9.]+)")


_EXCEPTION_RX = re.compile(
    r"^([A-Z]\w+(?:Error|Exception|Warning))(?::\s*(.+))?$",
    re.MULTILINE,
)


def _extract_kill_reason(log_text: str, rc: int) -> str:
    """Best-effort exception summary for the kill_reason field.

    Looks for the LAST `XxxError: ...` style line in the log (Python traceback
    final line). Falls back to the last non-empty log line if no exception
    line is found, with the unicode banner art filtered out.
    """
    matches = list(_EXCEPTION_RX.finditer(log_text))
    if matches:
        m = matches[-1]
        exc_name = m.group(1)
        exc_msg = (m.group(2) or "").strip()
        # File path with line number from the immediately preceding traceback frame
        path_hint = ""
        head = log_text[:m.start()]
        for line in reversed(head.splitlines()):
            line = line.strip()
            if line.startswith('File "') and "line " in line:
                path_hint = line[:120]
                break
        snippet = f"{exc_name}: {exc_msg}" if exc_msg else exc_name
        if path_hint:
            return f"rc={rc}: {snippet[:160]} [{path_hint}]"[:280]
        return f"rc={rc}: {snippet[:200]}"

    # No exception found — fall back to last non-empty, non-art line
    BAD_CHARS = set("█░▓▒")
    for line in reversed(log_text.splitlines()):
        s = line.strip()
        if not s:
            continue
        if any(c in s for c in BAD_CHARS):
            continue
        return f"rc={rc}: {s[:200]}"
    return f"rc={rc}"


def _parse_metrics(log_text: str) -> tuple[float | None, float | None]:
    """Return (val_bpb, core_metric) from the training log; either may be None."""
    bpb_matches = _BPB_RX.findall(log_text)
    val_bpb = float(bpb_matches[-1]) if bpb_matches else None
    if val_bpb is None:
        m = _MIN_BPB_RX.search(log_text)
        if m:
            val_bpb = float(m.group(1))
    core_matches = _CORE_RX.findall(log_text)
    core_metric = float(core_matches[-1]) if core_matches else None
    return val_bpb, core_metric


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULT_DIR / f"run_seed{SEED}.log"
    out_path = RESULT_DIR / f"run_seed{SEED}.jsonl"

    if not VENDOR.is_dir():
        # Hard-fail loudly — vendor copy must have been seeded by
        # _stage_workdir on first iter. If missing, the staging step
        # failed silently OR the operator deleted vendor/ between iters.
        with out_path.open("w") as f:
            json.dump({
                "status":      "PREFLIGHT_CRASH",
                "core_metric": None, "val_bpb": None,
                "train_s": None, "total_wall_s": None,
                "kill_reason": (
                    f"vendor copy missing at {VENDOR} — _stage_workdir should "
                    f"have seeded multi_agent_nc/vendor/nanochat/ on first iter. "
                    f"Verify the package's vendor/ dir exists "
                    f"(multi_agent_nc/vendor/nanochat/) and re-launch."
                ),
            }, f); f.write("\n")
        return 2

    args_d = _argv_for_smoke(TRAIN_ARGS_BASELINE) if SMOKE_TEST else TRAIN_ARGS_BASELINE
    flat_args = _flatten(args_d)

    if NPROC > 1:
        # Use `python -m torch.distributed.run` instead of the `torchrun`
        # launcher binary: many cluster images ship torch but not the
        # console_scripts entry, so `torchrun` is absent from PATH while
        # `torch.distributed.run` (the underlying module) is always
        # importable. sys.executable picks up the same interpreter
        # run_trial.sh resolved (PYTHON probe), avoiding python/python3
        # mismatch on minimal images.
        train_cmd = [sys.executable, "-m", "torch.distributed.run",
                     "--standalone", f"--nproc_per_node={NPROC}",
                     "-m", "scripts.base_train", *flat_args]
    else:
        # CPU/single-GPU dev mode (WSL etc.) — no distributed launcher.
        train_cmd = [sys.executable, "-m", "scripts.base_train", *flat_args]

    # PYTHONPATH wiring: ensure the 8 torchrun ranks (each a fresh Python
    # process) import nanochat from OUR workdir vendor copy, NOT from any
    # the shared filesystem / /dev/shm copy under NANOCHAT_BASE_DIR. Agent's edits to
    # vendor/nanochat/*.py only matter if the child processes load THIS
    # copy. We prepend VENDOR so it wins over any system / NANOCHAT_BASE_DIR
    # copy on sys.path. Data assets (tokenizer, shards, eval_bundle) still
    # come from NANOCHAT_BASE_DIR — that env var is unchanged here.
    child_env = dict(os.environ)
    existing_pp = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = (
        f"{VENDOR}{os.pathsep}{existing_pp}" if existing_pp else str(VENDOR)
    )
    # the cluster environments are not reliable places to discover/download remote kernels.
    # nanochat/flash_attention.py defaults this to local SDPA fallback; setting
    # it here makes the runtime contract explicit in logs and child ranks.
    child_env.setdefault("NANOCHAT_DISABLE_REMOTE_FA3", "1")
    child_env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    # Environment outbound to huggingface.co is flaky; force offline to short-circuit
    # any accidental Hub call from kernels / datasets / tokenizers / hub APIs.
    # Operators who pre-cached an HF_HOME on the shared filesystem opt back in via env.
    child_env.setdefault("HF_HUB_OFFLINE", "1")
    child_env.setdefault("TRANSFORMERS_OFFLINE", "1")
    if SMOKE_TEST:
        # Preflight is a correctness gate, not a performance measurement.
        # Upstream base_train always torch.compile()s and saves a final
        # checkpoint; both are expensive and have caused 20-minute smoke
        # timeouts on fresh GPU/GPU environments. The vendored base_train.py below
        # honors these env toggles only for the SMOKE path. TORCHDYNAMO_DISABLE
        # catches @torch.compile-decorated optimizer helpers too.
        child_env["NANOCHAT_DISABLE_TORCH_COMPILE"] = "1"
        child_env["NANOCHAT_SKIP_FINAL_SAVE"] = "1"
        child_env["TORCHDYNAMO_DISABLE"] = "1"
        child_env["TORCH_COMPILE_DISABLE"] = "1"

    t0 = time.monotonic()
    with log_path.open("w") as logf:
        logf.write(f"# train_cmd: {shlex.join(train_cmd)}\n")
        logf.write(f"# cwd: {VENDOR}\n")
        logf.write(f"# PYTHONPATH: {child_env['PYTHONPATH']}\n")
        logf.write(f"# NANOCHAT_BASE_DIR: {BASE_DIR}\n")
        logf.write(f"# SMOKE_TEST: {SMOKE_TEST}\n")
        logf.write(f"# NANOCHAT_DISABLE_REMOTE_FA3: {child_env.get('NANOCHAT_DISABLE_REMOTE_FA3')}\n")
        logf.write(f"# HF_HUB_OFFLINE: {child_env.get('HF_HUB_OFFLINE')}\n\n")
        logf.flush()
        proc = subprocess.run(train_cmd, cwd=str(VENDOR), env=child_env,
                              stdout=logf, stderr=subprocess.STDOUT)
    train_s = time.monotonic() - t0

    log_text = log_path.read_text(errors="replace")
    val_bpb, core_metric = _parse_metrics(log_text)

    if proc.returncode != 0:
        status = "CRASH"
        reason = _extract_kill_reason(log_text, proc.returncode)
    elif not SMOKE_TEST and core_metric is None:
        status = "CRASH"
        reason = "rc=0 but no CORE metric was printed; check eval_bundle/core eval logs"
    else:
        status = "OK"
        reason = ""

    with out_path.open("w") as f:
        json.dump({
            "status":       status,
            "core_metric":  core_metric,
            "val_bpb":      val_bpb,
            "train_s":      train_s,
            "total_wall_s": train_s,
            "smoke":        SMOKE_TEST,
            "kill_reason":  reason,
        }, f); f.write("\n")
    # Propagate child failure as our exit code so run_trial.sh can gate
    # phase 2 on phase 1's success. Without this, run_trial sees rc=0,
    # proceeds to real run, and overwrites the crash-row JSONL.
    return 0 if status == "OK" else 2


if __name__ == "__main__":
    sys.exit(main())
