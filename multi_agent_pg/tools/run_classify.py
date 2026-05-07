"""run_classify.py — post-trial log→jsonl classifier.

Invoked by run_trial.sh at the end of a trial. Reads the combined log
(preflight + train + pack), extracts the competition-relevant fields, and
emits a single-line jsonl compatible with the legacy validate_full.py
contract consumed by harness/tracker.parse_validate_result:

    {"status", "val_bpb", "artifact_bytes", "train_s", "eval_s",
     "total_wall_s", "kill_reason"}

Status values (coarse, harness-side):

    VALID        — val_bpb present AND artifact bytes ≤ 16 MB
    DQ_SIZE      — artifact bytes > 16 MB (preflight size_blocked or post-run)
    DQ_TRAIN     — train phase exceeded 600s (train_gpt.py self-caps, so this
                   means the in-file cap was broken by an edit)
    DQ_EVAL      — outer timeout fired OR eval phase exceeded 600s budget
    CRASH        — train non-zero AND no val_bpb, or preflight crashed
    INCOMPLETE   — pack step failed but train ran

Parsing rules (mirror single_agent/research/tracker.parse_metric):

    val_bpb        primary regex: quantized_ttt val_loss:... val_bpb:F
                   fallback:      val_bpb:F (last match)
    artifact_bytes Submission size: N bytes
    train_s        train_time:Nms (train_gpt.py prints on wallclock_cap
                   or normal completion)
    eval_s         tight real eval-phase wall — `--- EVAL_WALL Ns ---`
                   marker emitted by run_trainer.py (time from first
                   post-train keyword hit to child exit). Falls back to
                   `run_elapsed - train_s` only when the marker is absent
                   (pre-fix logs, or runs that exited before post-train).
                   The fallback is known to over-report by 100-200s because
                   it double-counts compile+warmup+teardown.
    total_wall_s   pre_elapsed + run_elapsed
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Regex suite — keep in lockstep with single_agent/research/tracker.py.
_VAL_BPB_PRIMARY = re.compile(
    r"quantized_ttt\s+val_loss:[0-9.]+\s+val_bpb:([0-9.]+)"
)
_VAL_BPB_FALLBACK = re.compile(r"val_bpb:([0-9.]+)")
_SUBMISSION_SIZE = re.compile(r"^Submission size:\s*(\d+)\s*bytes", re.MULTILINE)
_TRAIN_TIME_MS = re.compile(r"train_time:\s*(\d+(?:\.\d+)?)ms\s+step:")
_SMOKE_TOTAL = re.compile(r"^smoke_pack_bytes:.*total=(\d+)", re.MULTILINE)
_OUTER_TIMEOUT = re.compile(r"---\s*OUTER_TIMEOUT\b")
_EVAL_TIMEOUT  = re.compile(r"---\s*EVAL_TIMEOUT\b")
_EVAL_WALL     = re.compile(r"---\s*EVAL_WALL\s+([\d.]+)\s*s")

SIZE_LIMIT_DEFAULT = 16_000_000
TRAIN_BUDGET_S = 600.0
EVAL_BUDGET_S = 600.0
# Train-budget tolerance absorbs the step-atomic overshoot of train_gpt.py's
# own wallclock cap check: `reached_cap` fires at end-of-step, so the last
# step always straddles the 600 s boundary and train_s can legitimately end
# up 0.5–2 s over 600 without any agent wrongdoing. 5 s is the value inherited
# from single_agent; it covers the worst-case straddle without providing a
# meaningful buffer (a typical step is 0.2–1 s).
#
# There is NO equivalent tolerance on eval: the post-train phase is a
# continuous wallclock, has no step granularity, and any EVAL_WALL > 600
# reflects real eval work over budget.
_TRAIN_BUDGET_TOLERANCE_S = 5.0


def _extract_val_bpb(log: str) -> float | None:
    m = _VAL_BPB_PRIMARY.search(log)
    if m:
        return float(m.group(1))
    matches = _VAL_BPB_FALLBACK.findall(log)
    if matches:
        return float(matches[-1])
    return None


def _extract_submission_size(log: str) -> int | None:
    m = _SUBMISSION_SIZE.search(log)
    return int(m.group(1)) if m else None


def _extract_train_s(log: str) -> float | None:
    """train_gpt.py prints `train_time:Nms step:...` at stop. Last wins."""
    matches = _TRAIN_TIME_MS.findall(log)
    if not matches:
        return None
    return float(matches[-1]) / 1000.0


def classify(args) -> dict:
    log_path = Path(args.log)
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""

    pre_elapsed = float(args.pre_elapsed or 0)
    run_elapsed = float(args.run_elapsed or 0)
    total_wall = pre_elapsed + run_elapsed

    # ── Preflight abort paths ────────────────────────────────────────────────
    if args.preflight_status == "size_blocked":
        smoke = int(args.smoke_bytes or 0)
        return {
            "status":         "DQ_SIZE",
            "val_bpb":        None,
            "artifact_bytes": smoke,
            "train_s":        None,
            "eval_s":         None,
            "total_wall_s":   pre_elapsed,
            "kill_reason":    f"preflight_size_blocked: smoke_total={smoke} limit={args.size_limit}",
        }
    if args.preflight_status == "crash":
        return {
            "status":         "CRASH",
            "val_bpb":        None,
            "artifact_bytes": None,
            "train_s":        None,
            "eval_s":         None,
            "total_wall_s":   pre_elapsed,
            "kill_reason":    f"preflight_crash: rc_nonzero after {pre_elapsed:.0f}s",
        }

    # ── Real-run classification ──────────────────────────────────────────────
    val_bpb = _extract_val_bpb(log)
    artifact = _extract_submission_size(log)
    train_s = _extract_train_s(log)
    eval_s: float | None = None
    eval_wall_m = _EVAL_WALL.search(log)
    if eval_wall_m:
        # Tight source of truth: run_trainer.py measured wall from the first
        # post-train keyword to the child exit (no compile/warmup/teardown).
        eval_s = float(eval_wall_m.group(1))
    elif train_s is not None and run_elapsed > 0:
        # Fallback for logs that never emitted EVAL_WALL. Known to inflate
        # by 100-200s via compile+warmup+teardown; kept only for
        # backward-compat with pre-fix archived logs.
        eval_s = max(0.0, run_elapsed - train_s)

    eval_timeout = bool(_EVAL_TIMEOUT.search(log))
    outer_timeout = int(args.outer_timeout or 0) == 1 or bool(_OUTER_TIMEOUT.search(log))
    train_rc = int(args.train_rc or 0)
    pack_rc = int(args.pack_rc or 0)
    size_limit = int(args.size_limit or SIZE_LIMIT_DEFAULT)

    # Priority-ordered rules:
    #  1. eval-budget watchdog fired ⇒ DQ_EVAL (run_trainer.py SIGTERM'd at
    #     eval_elapsed > 600s). Takes precedence over outer_timeout because
    #     it's the more actionable diagnosis (run_trainer.py emits EVAL_TIMEOUT
    #     first chronologically; outer backstop would only fire if that
    #     somehow failed).
    #  2. outer total-wall backstop fired ⇒ DQ_EVAL (softer diagnosis)
    #  3. train self-cap exceeded ⇒ DQ_TRAIN. 5s tolerance absorbs the
    #     step-atomic straddle of the 600s boundary (see TRAIN_BUDGET_TOLERANCE_S
    #     rationale above). Not a "grace window" — just single-step granularity.
    #  4. eval phase ran beyond 600s but watchdog missed it (natural exit
    #     in the window between budget crossing and next 0.5s poll) ⇒
    #     DQ_EVAL. Catches the EVAL_WALL > 600 case where no EVAL_TIMEOUT
    #     marker was emitted. STRICT — no tolerance, eval has no granularity.
    #  5. artifact > 16 MB ⇒ DQ_SIZE
    #  6. no val_bpb / train non-zero ⇒ CRASH
    #  7. pack failed ⇒ INCOMPLETE
    #  8. else ⇒ VALID
    if eval_timeout:
        status = "DQ_EVAL"
        kill = f"eval_budget_overrun: eval phase exceeded {EVAL_BUDGET_S:.0f}s (train_rc={train_rc})"
    elif outer_timeout:
        status = "DQ_EVAL"
        kill = f"outer_timeout after {run_elapsed:.0f}s (train_rc={train_rc})"
    elif train_s is not None and train_s > TRAIN_BUDGET_S + _TRAIN_BUDGET_TOLERANCE_S:
        status = "DQ_TRAIN"
        kill = f"train_s={train_s:.1f} exceeds {TRAIN_BUDGET_S:.0f}s budget"
    elif eval_s is not None and eval_s > EVAL_BUDGET_S:
        status = "DQ_EVAL"
        kill = (
            f"eval_s={eval_s:.1f} exceeds {EVAL_BUDGET_S:.0f}s budget "
            f"(watchdog did not fire — natural exit in poll gap)"
        )
    elif artifact is not None and artifact > size_limit:
        status = "DQ_SIZE"
        kill = f"post_run_size={artifact} limit={size_limit}"
    elif val_bpb is None or train_rc != 0:
        status = "CRASH"
        kill = f"no val_bpb or train_rc={train_rc}"
    elif pack_rc != 0:
        status = "INCOMPLETE"
        kill = f"pack_rc={pack_rc}"
    else:
        status = "VALID"
        kill = ""

    return {
        "status":         status,
        "val_bpb":        val_bpb,
        "artifact_bytes": artifact,
        "train_s":        train_s,
        "eval_s":         eval_s,
        "total_wall_s":   total_wall,
        "kill_reason":    kill,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preflight-status", default="ok",
                    choices=("ok", "crash", "size_blocked"))
    ap.add_argument("--smoke-bytes", default="0")
    ap.add_argument("--pre-elapsed", default="0")
    ap.add_argument("--train-rc", default="0")
    ap.add_argument("--outer-timeout", default="0")
    ap.add_argument("--run-elapsed", default="0")
    ap.add_argument("--pack-rc", default="0")
    ap.add_argument("--size-limit", default=str(SIZE_LIMIT_DEFAULT))
    args = ap.parse_args()

    record = classify(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(record) + "\n", encoding="utf-8")
    print(json.dumps(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
