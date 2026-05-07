"""Scheduler protocol for trial execution.

The harness submits each trial through a `Scheduler` object. The bundled
implementation `LocalScheduler` runs `bash run_trial.sh` as a subprocess
on the local node. To target a different backend (cluster scheduler,
batch system, container orchestrator), implement the same protocol
against your backend and pass an instance into the supervisor.

This module defines:

  * Scheduler         — the Protocol every backend must satisfy.
  * JobHandle         — opaque handle returned by submit().
  * JobResult         — result of a finished job (terminal state).
  * JobNotFoundError  — raised when poll/wait sees a vanished job.
  * is_terminal()     — phase-string helper.

Concrete implementations live in sibling modules:

  * local_scheduler.LocalScheduler — subprocess-based, default.

Job naming convention. Names are 1..N character ASCII used by
`stop_all_owned(prefix)` to find supervisor-owned jobs at shutdown
time. Each task adapter contributes a prefix via
`adapter.job_name_prefix` (e.g. "apg" for Parameter Golf).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable


# Phase strings shared across all backends. Submit.py checks `is_terminal`
# rather than each phase explicitly so new backends can extend phases freely
# as long as they still terminate in {succeeded, failed, timeout}.
_TERMINAL_PHASES = frozenset({"succeeded", "failed", "timeout"})


@dataclass(frozen=True)
class JobHandle:
    """Opaque handle returned by Scheduler.submit().

    Backends are free to put extra fields in `extra`; the harness only
    reads `name` and `job_id` directly. The handle round-trips back into
    `wait`, `stop`, and `fetch_logs` calls.
    """
    name: str
    job_id: str
    log_path: Optional[Path] = None
    extra: dict = field(default_factory=dict)


@dataclass
class JobResult:
    """Result of a finished job. `phase` is one of {succeeded, failed, timeout}."""
    name: str
    phase: str
    exit_code: Optional[int]
    elapsed_s: float
    log_path: Optional[Path]


class JobNotFoundError(RuntimeError):
    """Raised when the backend cannot find the job (deleted, expired, never accepted).

    The harness treats this as a terminal failure with exit_code=None.
    """


def is_terminal(phase: str) -> bool:
    """Return True if `phase` is one of {succeeded, failed, timeout}."""
    return phase in _TERMINAL_PHASES


@runtime_checkable
class Scheduler(Protocol):
    """Trial execution backend.

    The harness uses six methods:

      submit            — accept a job, return JobHandle (typically fast).
      wait              — block until the job is terminal, return JobResult.
      stop              — request termination of a running job.
      fetch_logs        — return captured stdout/stderr.
      snapshot_owned    — list (name, job_id) for in-flight jobs whose
                          name begins with `prefix`.
      stop_all_owned    — stop every in-flight job whose name begins
                          with `prefix`. Returns the list of (name, job_id)
                          that were stopped. Used by supervisor shutdown.
    """

    async def submit(
        self,
        *,
        name: str,
        workdir: Path,
        script: str,
        env: dict[str, str],
        timeout_seconds: float,
    ) -> JobHandle:
        ...

    async def wait(self, job: JobHandle) -> JobResult:
        ...

    def stop(self, job: JobHandle) -> None:
        ...

    def fetch_logs(self, job: JobHandle) -> str:
        ...

    def snapshot_owned(self, prefix: str) -> list[tuple[str, str]]:
        ...

    def stop_all_owned(self, prefix: str) -> list[tuple[str, str]]:
        ...


__all__ = [
    "Scheduler",
    "JobHandle",
    "JobResult",
    "JobNotFoundError",
    "is_terminal",
]
