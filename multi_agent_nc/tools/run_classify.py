"""run_classify.py — node-side trial result classifier for NanoChat-d12.

Mirrors CIFAR's classifier shape (status taxonomy, smoke-row rejection).
Inputs / outputs / behavior identical except for the JSONL field shape
(NC has `core_metric` + `val_bpb`; CIFAR has `accuracy`).

  --log <path>          — train.log (or preflight.log)
  --train-rc <int>      — exit code of the experiment.py invocation
  --jsonl <path>        — JSONL written by experiment.py (if it got that far)
  --preflight-status    — "crash" → synthesize PREFLIGHT_CRASH row
  --out <path>          — destination JSONL path (PG-shape)

Always writes exactly one JSON line to --out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_TIMEOUT_SIGNATURES = (
    "train_budget_overrun",
    "Timeout",
    "killed",
)
_OOM_SIGNATURES = (
    "CUDA out of memory",
    "torch.OutOfMemoryError",
    "OOM",
)


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


def _classify_status_and_reason(train_rc: int, tail: str) -> tuple[str, str]:
    if train_rc == 124:                 # `timeout` shell exit code
        return "TIMEOUT", "timeout (run_trial.sh wall cap)"
    for sig in _TIMEOUT_SIGNATURES:
        if sig in tail:
            return "TIMEOUT", f"timeout-like: {sig}"
    for sig in _OOM_SIGNATURES:
        if sig in tail:
            return "CRASH", "CUDA OOM"
    if train_rc == 0:
        return "CRASH", ""
    lines = [l.strip() for l in tail.splitlines() if l.strip()]
    if lines:
        return "CRASH", f"rc={train_rc}: {lines[-1][:200]}"
    return "CRASH", f"rc={train_rc}"


def _synthesize_crash_row(log_path: Path, train_rc: int) -> dict:
    tail = _read_tail(log_path)
    status, reason = _classify_status_and_reason(train_rc, tail)
    return {
        "status":       status,
        "core_metric":  None,
        "val_bpb":      None,
        "train_s":      None,
        "total_wall_s": None,
        "smoke":        False,
        "kill_reason":  reason,
    }


def _parse_existing_jsonl(jsonl_path: Path) -> dict | None:
    if not jsonl_path.is_file():
        return None
    try:
        with jsonl_path.open() as f:
            line = f.readline().strip()
        if not line:
            return None
        rec = json.loads(line)
        if not isinstance(rec, dict):
            return None
        return rec
    except (OSError, json.JSONDecodeError):
        return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, default=None)
    p.add_argument("--train-rc", type=int, default=0)
    p.add_argument("--jsonl", type=Path, default=None)
    p.add_argument("--preflight-status", type=str, default=None)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Preflight-crash branch: experiment.py likely already wrote a CRASH
    # row with a useful kill_reason (e.g. "FileNotFoundError: tokenizer.pkl").
    # Promote that to PREFLIGHT_CRASH while preserving the reason.
    if args.preflight_status == "crash":
        existing = _parse_existing_jsonl(args.jsonl) if args.jsonl else None
        existing_reason = (existing or {}).get("kill_reason", "") or ""
        if existing_reason:
            reason = existing_reason
        else:
            tail = _read_tail(args.log) if args.log else ""
            _, reason = _classify_status_and_reason(args.train_rc, tail)
        rec = {
            "status":       "PREFLIGHT_CRASH",
            "core_metric":  None,
            "val_bpb":      None,
            "train_s":      (existing or {}).get("train_s"),
            "total_wall_s": (existing or {}).get("total_wall_s"),
            "smoke":        True,
            "kill_reason":  reason or "preflight",
        }
        with args.out.open("w") as f:
            json.dump(rec, f); f.write("\n")
        return 0

    # Normal branch: prefer the JSONL experiment.py wrote; otherwise crash row.
    # Defense vs preflight contamination: reject smoke=True rows.
    # Also: if existing row already records a CRASH/TIMEOUT with a non-empty
    # kill_reason, preserve that reason rather than re-synthesizing a worse one.
    existing = _parse_existing_jsonl(args.jsonl) if args.jsonl else None
    has_score = existing is not None and existing.get("core_metric") is not None
    is_smoke  = bool(existing.get("smoke", False)) if existing else False
    existing_status = (existing or {}).get("status", "")
    existing_reason = (existing or {}).get("kill_reason", "") or ""

    if has_score and not is_smoke and args.train_rc == 0:
        if "status" not in existing:
            existing["status"] = "OK"
        rec = existing
    else:
        rec = _synthesize_crash_row(args.log if args.log else Path("/dev/null"),
                                    args.train_rc)
        if existing is not None and is_smoke:
            rec["kill_reason"] = ("smoke JSONL leaked into real-run output — "
                                  "real run did not write a fresh row")
        elif existing_reason and existing_status in ("CRASH", "TIMEOUT", "PREFLIGHT_CRASH"):
            # experiment.py already classified — preserve its richer reason.
            rec["kill_reason"] = existing_reason
            if existing_status == "TIMEOUT":
                rec["status"] = "TIMEOUT"
            for k in ("train_s", "total_wall_s"):
                if existing.get(k) is not None:
                    rec[k] = existing[k]

    with args.out.open("w") as f:
        json.dump(rec, f); f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
