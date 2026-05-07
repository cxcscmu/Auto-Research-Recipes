#!/usr/bin/env python3
"""Wrap torchrun with an eval-budget watchdog.

Competition rules: train ≤ 600s AND eval ≤ 600s as independent budgets,
with compile time unbounded. `train_gpt.py` self-caps train via
MAX_WALLCLOCK_SECONDS=600 and absorbs compile cost into WARMUP_STEPS
before the clock starts, so the train budget is strict training time.

But train_gpt.py has NO internal eval budget — sliding_window + TTT +
GPTQ can run as long as they want. This wrapper enforces the 600s eval
cap the same way single_agent/research/runner.py did:

  1. launch torchrun in a new session (so we can kill the whole group
     without taking down our own shell)
  2. reader thread classifies each stdout line into a phase via
     _PHASE_TABLE (ported verbatim from single_agent)
  3. 0.5s polling loop: once the first post-train phase
     (wrap-up / gptq / compress / bpb_eval) is entered, start an eval
     timer; at eval_elapsed > EVAL_BUDGET_S send SIGTERM to the process
     group, wait TERM_GRACE_S for PHASE_SUMMARY + logs to flush, then
     SIGKILL; emit `--- EVAL_TIMEOUT after Ns ---` to stdout so
     run_classify.py tags DQ_EVAL
  4. total-wall backstop (--total-timeout, typically 2200s = 600 train +
     600 eval + 1000 compile headroom) as a second safety net, emits
     `--- OUTER_TIMEOUT after Ns ---`
  5. install SIGTERM handler so the outer bash `timeout` wrapper can
     still clean up the grandchild process group on backstop-expiry

Usage (from run_trial.sh):
    python run_trainer.py --eval-budget 600 --total-timeout 2200 -- \\
        torchrun --standalone --nproc_per_node=8 train_gpt.py

Exit code: 124 on either eval-cap or total-timeout fire, otherwise the
real rc from torchrun (0 on success, non-zero on child crash).
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Optional

# Timeline order — later phases have higher index (advance-only).
_PHASE_ORDER = ["init", "compile", "train", "wrap-up", "gptq", "compress", "bpb_eval"]

# (phase_name, [substring keywords]). See single_agent/research/runner.py for
# the extensive keyword-hygiene notes on why each substring was chosen not to
# collide with the Hyperparameters config dump at startup.
_PHASE_TABLE = [
    ("compile",  ["model_params:", "mtp_num_heads:", "world_size:"]),
    ("train",    ["train_loss:"]),
    ("wrap-up",  ["ema:", "lawa:averaging", "swa_avg", "stopping_early", "wallclock_cap"]),
    ("gptq",     ["gptq:collect", "collecting hessians", "quantized weights:"]),
    ("compress", ["quantized+", "submission size", "serialized model quantized"]),
    ("bpb_eval", [
        "quantized val_loss:",
        "quantized_sliding_window val_loss:",
        "quantized_ttt val_loss:",
        "ttt:start",
    ]),
]

_POLL_INTERVAL_S = 0.5
_CHILD_REAP_TIMEOUT_S = 15.0


def _detect_phase(line: str, current: str) -> str:
    ll = line.lower()
    try:
        best_idx = _PHASE_ORDER.index(current)
    except ValueError:
        best_idx, current = 0, "init"
    best_phase = current
    for phase, kws in _PHASE_TABLE:
        if any(kw in ll for kw in kws):
            try:
                idx = _PHASE_ORDER.index(phase)
            except ValueError:
                continue
            if idx > best_idx:
                best_idx, best_phase = idx, phase
    return best_phase


def _first_post_train(entered: dict[str, float]) -> Optional[float]:
    for p in ("wrap-up", "gptq", "compress", "bpb_eval"):
        if p in entered:
            return entered[p]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--eval-budget", type=float, default=600.0,
                    help="Eval-phase wallclock cap in seconds (default: 600).")
    ap.add_argument("--total-timeout", type=float, default=2200.0,
                    help="Total-wall backstop in seconds (default: 2200 = "
                         "600 train + 600 eval + 1000 compile headroom).")
    ap.add_argument("--term-grace", type=float, default=8.0,
                    help="Seconds between SIGTERM and SIGKILL escalation.")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="Command to run after `--` (typically `torchrun ...`).")
    args = ap.parse_args()
    cmd = [a for a in args.cmd if a != "--"]
    if not cmd:
        print("run_trainer: no command given", file=sys.stderr)
        return 2

    t0 = time.perf_counter()

    # start_new_session=True = setsid() → child becomes its own process-group
    # leader. Lets us killpg(child.pid, ...) without touching our own shell.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # merge — simpler + preserves line order
        bufsize=1,
        text=True,
        start_new_session=True,
    )

    # Forward outer SIGTERM/SIGINT to the child's session so bash's outer
    # `timeout 2200` backstop still cleans up grandchildren instead of
    # orphaning torchrun workers.
    def _forward(signum, _frame) -> None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=args.term_grace)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        sys.exit(128 + signum)
    signal.signal(signal.SIGTERM, _forward)
    signal.signal(signal.SIGINT,  _forward)

    phase = ["init"]
    entered: dict[str, float] = {"init": 0.0}
    eval_term_sent_at: Optional[float] = None
    eval_timed_out = False
    total_timed_out = False
    # Captured at the moment we exit the polling loop (child self-exited or
    # we SIGKILLed). Used for the EVAL_WALL marker — we must NOT include the
    # post-loop proc.wait / reader-join reap window, which can add 15s.
    child_exit_elapsed: Optional[float] = None

    def _reader() -> None:
        # proc.stdout is never None because stdout=PIPE above.
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            new_phase = _detect_phase(line, phase[0])
            if new_phase != phase[0]:
                entered.setdefault(new_phase, time.perf_counter() - t0)
                phase[0] = new_phase

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    while True:
        if proc.poll() is not None:
            child_exit_elapsed = time.perf_counter() - t0
            break

        elapsed = time.perf_counter() - t0
        eval_started = _first_post_train(entered)
        eval_elapsed = (elapsed - eval_started) if eval_started is not None else 0.0

        # ── Eval-budget watchdog ─────────────────────────────────────────
        if eval_started is not None and eval_elapsed > args.eval_budget:
            if eval_term_sent_at is None:
                eval_timed_out = True
                eval_term_sent_at = elapsed
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
            elif elapsed - eval_term_sent_at > args.term_grace:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                child_exit_elapsed = elapsed
                break

        # ── Total-wall backstop ──────────────────────────────────────────
        if elapsed >= args.total_timeout:
            total_timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            child_exit_elapsed = elapsed
            break

        time.sleep(_POLL_INTERVAL_S)

    try:
        proc.wait(timeout=_CHILD_REAP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=_CHILD_REAP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            pass
    t.join(timeout=_CHILD_REAP_TIMEOUT_S)

    # Emit the real eval-phase wall: from the first post-train keyword hit
    # to the moment the child exited / was SIGKILLed (reap excluded). This
    # is what the competition's 600s eval budget actually measures. The
    # legacy `eval_s = run_elapsed - train_s` in run_classify.py was a loose
    # proxy that swept in compile+warmup+teardown and routinely over-reported
    # by 100-200s — scary-looking but not an actual overrun.
    post_train_start = _first_post_train(entered)
    if post_train_start is not None and child_exit_elapsed is not None:
        real_eval = max(0.0, child_exit_elapsed - post_train_start)
        sys.stdout.write(f"\n--- EVAL_WALL {real_eval:.1f}s ---\n")
        sys.stdout.flush()

    # Emit marker. Priority: EVAL_TIMEOUT wins over OUTER_TIMEOUT (eval cap
    # fires first chronologically and is the more actionable diagnosis).
    if eval_timed_out:
        sys.stdout.write(f"\n--- EVAL_TIMEOUT after {int(args.eval_budget)}s ---\n")
        sys.stdout.flush()
        return 124
    if total_timed_out:
        sys.stdout.write(f"\n--- OUTER_TIMEOUT after {int(args.total_timeout)}s ---\n")
        sys.stdout.flush()
        return 124

    return proc.returncode if proc.returncode is not None else 1


if __name__ == "__main__":
    sys.exit(main())
