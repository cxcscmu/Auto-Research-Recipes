"""agent_core.harness — generic plumbing.

Modules:
  config           filesystem roots, swarm_config loader, sandbox probe.
  blackboard       authoritative TSV + workdir + locks.
  tracker          result-jsonl parser, exp_id allocator, crash excerpt.
  events           append-only events.jsonl writer.
  scheduler        Scheduler protocol, JobHandle, JobResult, JobNotFoundError.
  local_scheduler  bundled subprocess-based Scheduler (the default).
  baseline_audit   bootstrap-row synthesis on cold start.
  credentials      ANTHROPIC_API_KEY loader (env + .env file).

A process-wide Scheduler instance is exposed through `get_scheduler()`.
The default is `LocalScheduler`. Operators or tests can override the
instance with `set_scheduler(...)` before any trial is submitted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .scheduler import Scheduler


_scheduler_instance: Optional[Scheduler] = None


def get_scheduler() -> Scheduler:
    """Return the process-wide Scheduler. Lazy-creates a `LocalScheduler` on first call."""
    global _scheduler_instance
    if _scheduler_instance is None:
        from . import config
        from .local_scheduler import LocalScheduler
        log_dir = Path(config.LOCAL_ROOT) / "logs" / "jobs"
        _scheduler_instance = LocalScheduler(log_dir=log_dir)
    return _scheduler_instance


def set_scheduler(scheduler: Scheduler) -> None:
    """Override the process-wide Scheduler. Call this before any trial submits."""
    global _scheduler_instance
    _scheduler_instance = scheduler


__all__ = ["get_scheduler", "set_scheduler"]
