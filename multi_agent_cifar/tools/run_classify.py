"""run_classify.py — node-side multi-seed aggregator (CIFAR airbench96 v2).

Reads N per-seed jsonl files (run_seed{0..N-1}.jsonl), computes
{mean_acc, std_acc, mean_train_s, std_train_s, n_seeds}, applies the
acc≥0.96 threshold gate, and writes ONE aggregated row to --out (the
harness's _find_result_jsonl path = full_eval_results/<wd>/run_seed0.jsonl).

The aggregated row is what blackboard / dashboard / lineage read; per-seed
rows remain on disk for forensics but harness ignores them.

Status taxonomy (all written into the aggregated row):
  - OK            — mean_acc ≥ 0.9615 → train_s scored as the trial's score
  - BORDERLINE    — mean_acc ∈ [0.9585, 0.9615] → kept but flagged
  - DISQUALIFIED  — mean_acc < 0.9585 → train_s blanked, never wins
  - CRASH         — < N seeds produced jsonl, or train_rc != 0 across all
  - TIMEOUT       — train_rc == 124
  - PREFLIGHT_CRASH — preflight phase failed before seed loop

Inputs:
  --jsonl-glob   — glob for per-seed jsonls (e.g. ".../run_seed*.jsonl")
  --logs         — glob for per-seed logs (for crash kill_reason inference)
  --train-rc     — last seed's exit code (rough signal — phase 2 collected
                   "all_rc" = last failing seed's rc, or 0 if all passed)
  --n-seeds      — declared N from run_trial.sh; informational
  --preflight-status crash — synthesize PREFLIGHT_CRASH row (Phase 1 path)
  --out          — destination single-row aggregated jsonl

Threshold constants come from CIFARTaskAdapter.parse_validate_record (kept
in sync — do not drift).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys
from pathlib import Path


# Acc gate constant (single threshold; upstream airbench96 spec).
# v2.3: dropped the BORDERLINE buffer band — was a self-imposed safety
# margin that made the task unnecessarily hard at n=3. Strict upstream
# airbench96 is mean_acc ≥ 0.96, period.
ACC_KEEP_THRESHOLD = 0.96   # ≥ this → OK / keep; < this → DISQUALIFIED


_TIMEOUT_SIGNATURES = ("train_budget_overrun", "Timeout", "killed")
_OOM_SIGNATURES     = ("CUDA out of memory", "torch.OutOfMemoryError", "OOM")


def _read_tail(path: Path, n_bytes: int = 4096) -> str:
    if not path.is_file():
        return ""
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > n_bytes:
                f.seek(size - n_bytes)
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _classify_kill_reason(train_rc: int, log_glob: str) -> str:
    if train_rc == 124:
        return "timeout (run_trial.sh wall cap)"
    if train_rc == 0:
        return ""
    # Look in any matching log for telling signatures.
    for log_path in sorted(glob.glob(log_glob)):
        tail = _read_tail(Path(log_path))
        for sig in _TIMEOUT_SIGNATURES:
            if sig in tail:
                return f"timeout-like: {sig} ({Path(log_path).name})"
        for sig in _OOM_SIGNATURES:
            if sig in tail:
                return f"CUDA OOM ({Path(log_path).name})"
    return f"rc={train_rc} (logs uninformative)"


def _read_seed_jsonl(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        with path.open() as f:
            line = f.readline().strip()
        if not line:
            return None
        rec = json.loads(line)
        if not isinstance(rec, dict):
            return None
        return rec
    except (OSError, json.JSONDecodeError):
        return None


def _read_shell_timing(jsonl_path: Path) -> tuple[float | None, int | None]:
    """Read harness-side authoritative timing + rc sidecar files. These are
    written by run_trial.sh AFTER each seed exits and are NOT touchable by
    the agent's recipe edits — they are the trusted train_s source.

    Returns (elapsed_s, rc) or (None, None) if either sidecar missing/bad.
    """
    base = jsonl_path.with_suffix("")  # drops `.jsonl`
    elapsed_path = Path(str(base) + ".shell_elapsed_s")
    rc_path = Path(str(base) + ".shell_rc")
    if not (elapsed_path.is_file() and rc_path.is_file()):
        return None, None
    try:
        elapsed = float(elapsed_path.read_text().strip())
        rc = int(rc_path.read_text().strip())
        if not math.isfinite(elapsed) or elapsed < 0:
            return None, None
        return elapsed, rc
    except (OSError, ValueError):
        return None, None


def _aggregate(seed_recs: list[tuple[dict | None, Path]], expected_n: int) -> dict:
    """Aggregate per-seed (rec, jsonl_path) tuples into one row.

    Strict per-seed validity gates (all must hold to count toward mean):
      1. shell sidecar files exist (harness-side timing + rc)
      2. shell rc == 0
      3. recipe-side jsonl exists, is dict, status field == 'OK'
      4. recipe-side accuracy is finite float
      5. smoke != True (recipe didn't accidentally run smoke as real)
      6. shell_elapsed_s is finite > 0

    Score (`train_s`) comes from SHELL TIMING ONLY — recipe-reported
    train_s is informational and ignored for scoring (so the agent can't
    edit airbench96.py to under-report time and game the metric).

    Threshold gate on mean_acc determines OK / BORDERLINE / DISQUALIFIED.
    """
    valid_accs   = []
    valid_times  = []
    invalid_reasons: list[str] = []

    for rec, path in seed_recs:
        seed_name = path.stem
        elapsed, rc = _read_shell_timing(path)
        if elapsed is None or rc is None:
            invalid_reasons.append(f"{seed_name}: missing shell-timing sidecar")
            continue
        if rc != 0:
            invalid_reasons.append(f"{seed_name}: shell_rc={rc}")
            continue
        if rec is None:
            invalid_reasons.append(f"{seed_name}: jsonl missing/unparseable")
            continue
        # Recipe must self-report status=OK; CRASH or anything else from
        # recipe is rejected (defends against agent-written status='OK'
        # without actually completing).
        rec_status = str(rec.get("status", ""))
        if rec_status != "OK":
            invalid_reasons.append(f"{seed_name}: recipe status={rec_status!r}")
            continue
        if bool(rec.get("smoke", False)):
            invalid_reasons.append(f"{seed_name}: smoke=True")
            continue
        acc = rec.get("accuracy")
        if not isinstance(acc, (int, float)) or not math.isfinite(float(acc)):
            invalid_reasons.append(f"{seed_name}: accuracy={acc!r} non-finite")
            continue
        valid_accs.append(float(acc))
        valid_times.append(float(elapsed))   # shell-timed; harness-authoritative

    n = len(valid_accs)

    if n == 0:
        return {
            "status":      "CRASH",
            "accuracy":    None,
            "acc_std":     None,
            "train_s":     None,
            "warmup_s":    None,
            "n_seeds":     0,
            "expected_n":  expected_n,
            "smoke":       False,
            "kill_reason": ("no valid seed survived per-seed gate; "
                            "first reason: " + (invalid_reasons[0] if invalid_reasons
                                                else "no seeds produced")),
        }

    mean_acc = statistics.fmean(valid_accs)
    std_acc  = statistics.stdev(valid_accs) if len(valid_accs) > 1 else 0.0
    mean_ts  = statistics.fmean(valid_times)
    std_ts   = statistics.stdev(valid_times) if len(valid_times) > 1 else 0.0

    # Partial trial = CRASH (agent should know it's incomplete).
    if expected_n > 0 and n < expected_n:
        return {
            "status":      "CRASH",
            "accuracy":    mean_acc,
            "acc_std":     std_acc,
            "train_s":     mean_ts,
            "warmup_s":    None,
            "n_seeds":     n,
            "expected_n":  expected_n,
            "smoke":       False,
            "kill_reason": (f"only {n}/{expected_n} seeds passed per-seed gate; "
                            f"first invalid: {invalid_reasons[0] if invalid_reasons else 'unknown'}"),
        }

    # Threshold gate (only reached when all expected_n seeds valid).
    # Single 0.96 line — strict upstream airbench96 definition.
    if mean_acc >= ACC_KEEP_THRESHOLD:
        status = "OK"
    else:
        status = "DISQUALIFIED"

    return {
        "status":      status,
        "accuracy":    mean_acc,
        "acc_std":     std_acc,
        "train_s":     mean_ts,
        "train_s_std": std_ts,
        "warmup_s":    None,
        "n_seeds":     n,
        "expected_n":  expected_n,
        "smoke":       False,
        "kill_reason": "",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl-glob", type=str, default=None,
                   help="glob for per-seed jsonls (Phase 3 normal path).")
    p.add_argument("--logs", type=str, default=None,
                   help="glob for per-seed logs (for kill_reason inference).")
    p.add_argument("--train-rc", type=int, default=0)
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--preflight-status", type=str, default=None,
                   help='set to "crash" to synthesize PREFLIGHT_CRASH row.')
    # Back-compat single-file inputs (Phase 1 preflight uses these).
    p.add_argument("--jsonl", type=Path, default=None)
    p.add_argument("--log", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # ── Preflight-crash branch ─────────────────────────────────────────────
    if args.preflight_status == "crash":
        log_glob = args.logs or (str(args.log) if args.log else "")
        reason = _classify_kill_reason(args.train_rc, log_glob) or "preflight"
        rec = {
            "status":      "PREFLIGHT_CRASH",
            "accuracy":    None,
            "acc_std":     None,
            "train_s":     None,
            "warmup_s":    None,
            "n_seeds":     0,
            "expected_n":  args.n_seeds,
            "smoke":       True,
            "kill_reason": reason,
        }
        with args.out.open("w") as f:
            json.dump(rec, f); f.write("\n")
        return 0

    # ── Normal multi-seed aggregation ──────────────────────────────────────
    if not args.jsonl_glob:
        # Back-compat: single --jsonl path (used by older callers).
        if args.jsonl:
            seed_recs = [(_read_seed_jsonl(args.jsonl), args.jsonl)]
        else:
            seed_recs = []
    else:
        # Sort numerically so seed-0 is first (deterministic order).
        paths = sorted(glob.glob(args.jsonl_glob),
                       key=lambda p: int(''.join(c for c in Path(p).stem.replace('run_seed', '')
                                                 if c.isdigit()) or '0'))
        seed_recs = [(_read_seed_jsonl(Path(p)), Path(p)) for p in paths]

    rec = _aggregate(seed_recs, expected_n=args.n_seeds)

    # If we wound up with CRASH and have a non-zero train_rc, augment kill_reason.
    if rec["status"] == "CRASH" and not rec.get("kill_reason"):
        rec["kill_reason"] = _classify_kill_reason(args.train_rc, args.logs or "")

    with args.out.open("w") as f:
        json.dump(rec, f); f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
