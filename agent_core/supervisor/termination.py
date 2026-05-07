"""Termination predicates for the supervisor.

Run stops when EITHER of:

  * wall-clock elapsed ≥ DEADLINE_HOURS, or
  * no `keep`-status row has been appended for NO_IMPROVEMENT_GRACE_S
    since the most recent keep (or since supervisor start if no keeps
    yet exist).

OR-semantics — first trigger wins. The grace clock resets every time a
new keep lands, so a slow-but-steady swarm never gets starved.

Nothing here mutates the blackboard. `request_stop_if_triggered()`
delegates to blackboard.request_stop so the stop-reason string is
written atomically alongside stop.flag.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from ..harness import blackboard, config, tracker


@dataclass(frozen=True, slots=True)
class TerminationVerdict:
    """Snapshot of termination state at one moment."""
    should_stop: bool
    reason:      str          # "" when should_stop is False
    elapsed_s:   float
    last_keep_s: float        # seconds since the most recent keep (or supervisor start)


def _latest_keep_timestamp_iso() -> Optional[str]:
    """Return the most recent keep row's ISO timestamp, or None."""
    rows = tracker.read_results()
    for r in reversed(rows):
        if r.get("status") == "keep":
            ts = r.get("timestamp", "")
            if ts:
                return ts
    return None


def _iso_to_epoch(iso: str) -> Optional[float]:
    """Parse the exact ISO-8601 shape blackboard writes. None on mismatch."""
    # blackboard writes e.g. "2026-04-21T02:39:07Z" — strip trailing Z and
    # interpret as UTC. datetime.fromisoformat handles this in 3.11+.
    import datetime
    if iso.endswith("Z"):
        iso = iso[:-1]
    try:
        dt = datetime.datetime.fromisoformat(iso).replace(
            tzinfo=datetime.timezone.utc
        )
        return dt.timestamp()
    except ValueError:
        return None


def evaluate(
    started_at_monotonic: float,
    *,
    deadline_hours: float = config.DEADLINE_HOURS,
    no_improvement_grace_s: float = config.NO_IMPROVEMENT_GRACE_S,
) -> TerminationVerdict:
    """Return a TerminationVerdict for the caller's stop-or-continue decision.

    `started_at_monotonic` is the value of time.monotonic() at supervisor
    start. All relative math uses monotonic time so DST / clock-skew never
    changes a verdict; we only touch wall-clock to interpret the blackboard's
    ISO timestamps, and only by converting those to deltas.
    """
    now = time.monotonic()
    elapsed = now - started_at_monotonic
    deadline_s = deadline_hours * 3600.0

    # ── Wall-clock deadline ──────────────────────────────────────────────────
    if elapsed >= deadline_s:
        return TerminationVerdict(
            should_stop=True,
            reason=f"deadline reached: elapsed={elapsed:.0f}s ≥ {deadline_s:.0f}s",
            elapsed_s=elapsed,
            last_keep_s=elapsed,
        )

    # ── No-improvement grace ─────────────────────────────────────────────────
    # We measure "how long since the last keep landed" in wall-clock seconds.
    # If there has never been a keep, use supervisor start as the reference
    # so the grace window starts counting from day zero.
    last_iso = _latest_keep_timestamp_iso()
    if last_iso is None:
        last_keep_s = elapsed
    else:
        last_epoch = _iso_to_epoch(last_iso)
        if last_epoch is None:
            last_keep_s = elapsed
        else:
            last_keep_s = max(0.0, time.time() - last_epoch)

    if last_keep_s >= no_improvement_grace_s:
        return TerminationVerdict(
            should_stop=True,
            reason=(f"no-improvement grace exceeded: "
                    f"{last_keep_s:.0f}s since last keep ≥ {no_improvement_grace_s:.0f}s"),
            elapsed_s=elapsed,
            last_keep_s=last_keep_s,
        )

    return TerminationVerdict(
        should_stop=False,
        reason="",
        elapsed_s=elapsed,
        last_keep_s=last_keep_s,
    )


def request_stop_if_triggered(started_at_monotonic: float, **kw) -> TerminationVerdict:
    """Evaluate + drop stop.flag the first time a condition trips.

    Idempotent: once stop.flag exists, we neither re-check nor re-write it.
    Returns the current verdict either way (callers still use it for the
    end-of-run summary).
    """
    if blackboard.should_stop():
        return TerminationVerdict(
            should_stop=True,
            reason="stop.flag already present",
            elapsed_s=time.monotonic() - started_at_monotonic,
            last_keep_s=0.0,
        )
    v = evaluate(started_at_monotonic, **kw)
    if v.should_stop:
        blackboard.request_stop(v.reason)
    return v
