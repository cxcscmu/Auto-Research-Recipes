"""Local subprocess scheduler.

Runs each trial as a subprocess on the same host as the supervisor.
The expectation is that the host has the eight-GPU node attached
locally and the user runs at most one supervisor per task at a time.

Submission shape mirrors `timeout --signal=TERM --kill-after=N` from
the bash run_trial.sh: SIGTERM at deadline, SIGKILL `kill_grace_s`
seconds later if the child ignores SIGTERM. PYTHONUNBUFFERED is forced
on so a SIGKILLed child still flushes its last few seconds of stdout
to the captured log (the classifier's kill_reason extractor depends
on this).

Job registry. The scheduler maintains an in-process registry of
in-flight jobs keyed by name. `snapshot_owned(prefix)` and
`stop_all_owned(prefix)` walk the registry; the supervisor calls
the latter on `SIGINT` or `SIGTERM` to stop everything still alive.
The registry does not persist across processes, so two parallel
supervisors on the same node cannot see each other's jobs.
"""

from __future__ import annotations

import asyncio
import os
import signal
import time
import uuid
from pathlib import Path
from typing import Optional

from .scheduler import JobHandle, JobNotFoundError, JobResult


def _kill_grace_period_s(timeout_s: float) -> float:
    """Default SIGKILL grace = max(30 s, 5% of timeout)."""
    return max(30.0, 0.05 * timeout_s)


def _shell_quote(arg: str) -> str:
    """Single-quote shell-escape; safe for arbitrary string args."""
    return "'" + arg.replace("'", "'\"'\"'") + "'"


class LocalScheduler:
    """Run trials as subprocesses on the local host. See module docstring.

    Args:
      log_dir: directory where per-job log files are written. Created
        on demand. Each job's log is `<log_dir>/<job_name>__<job_id>.log`.
      kill_grace_s: override the default SIGKILL grace window. None
        means use _kill_grace_period_s(timeout_s).
    """

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        *,
        kill_grace_s: Optional[float] = None,
    ) -> None:
        self.log_dir = Path(log_dir) if log_dir is not None else Path.cwd() / "logs" / "jobs"
        self.kill_grace_s = kill_grace_s
        # name -> (job_id, asyncio.subprocess.Process, log_path, t0, timeout_s)
        self._registry: dict[str, tuple[str, asyncio.subprocess.Process, Path, float, float]] = {}

    # ── Submission ──────────────────────────────────────────────────────────

    async def submit(
        self,
        *,
        name: str,
        workdir: Path,
        script: str,
        env: dict[str, str],
        timeout_seconds: float,
    ) -> JobHandle:
        """Spawn `bash <script> <workdir>` in a new process group.

        The full operator environment is inherited; `env` is layered on
        top. PYTHONUNBUFFERED is forced. The handle returned carries
        the log path so the caller can stream or grep it later.
        """
        self.log_dir.mkdir(parents=True, exist_ok=True)
        job_id = uuid.uuid4().hex[:12]
        log_path = self.log_dir / f"{name}__{job_id}.log"

        child_env = dict(os.environ)
        child_env.update(env)
        child_env["PYTHONUNBUFFERED"] = "1"

        cmd = ["bash", script, str(workdir)]
        log_fh = log_path.open("w", buffering=1)

        proc = await asyncio.create_subprocess_shell(
            " ".join(_shell_quote(c) for c in cmd),
            cwd=str(workdir),
            env=child_env,
            stdout=log_fh,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        # Keep the file handle alive on the proc for cleanup; close in _wait.
        proc._log_fh = log_fh  # type: ignore[attr-defined]

        t0 = time.monotonic()
        self._registry[name] = (job_id, proc, log_path, t0, float(timeout_seconds))

        return JobHandle(name=name, job_id=job_id, log_path=log_path)

    # ── Lifecycle ───────────────────────────────────────────────────────────

    async def wait(self, job: JobHandle) -> JobResult:
        """Block until the job is terminal. Returns JobResult.

        Raises JobNotFoundError if the job is not in the registry.
        """
        entry = self._registry.get(job.name)
        if entry is None:
            raise JobNotFoundError(f"job {job.name} not in local registry")
        if entry[0] != job.job_id:
            raise JobNotFoundError(
                f"job {job.name} has different job_id (registry={entry[0]} handle={job.job_id})"
            )

        job_id, proc, log_path, t0, timeout_s = entry
        kill_grace = self.kill_grace_s if self.kill_grace_s is not None else _kill_grace_period_s(timeout_s)

        try:
            exit_code = await asyncio.wait_for(proc.wait(), timeout=timeout_s)
            elapsed = time.monotonic() - t0
            self._close_log(proc)
            self._registry.pop(job.name, None)
            phase = "succeeded" if exit_code == 0 else "failed"
            return JobResult(
                name=job.name, phase=phase, exit_code=exit_code,
                elapsed_s=elapsed, log_path=log_path,
            )
        except asyncio.TimeoutError:
            await self._terminate_process_group(proc, kill_grace)
            elapsed = time.monotonic() - t0
            try:
                exit_code = proc.returncode
            except Exception:
                exit_code = None
            self._close_log(proc)
            self._registry.pop(job.name, None)
            return JobResult(
                name=job.name, phase="timeout", exit_code=exit_code,
                elapsed_s=elapsed, log_path=log_path,
            )

    def stop(self, job: JobHandle) -> None:
        """Synchronously SIGTERM the job. Does not wait for exit."""
        entry = self._registry.get(job.name)
        if entry is None:
            return
        _, proc, _, _, _ = entry
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

    def fetch_logs(self, job: JobHandle) -> str:
        """Return the contents of the captured stdout/stderr log."""
        log_path = job.log_path
        if log_path is None:
            entry = self._registry.get(job.name)
            if entry is None:
                return ""
            log_path = entry[2]
        try:
            return Path(log_path).read_text(errors="replace")
        except FileNotFoundError:
            return ""

    # ── Owned-jobs cleanup ──────────────────────────────────────────────────

    def snapshot_owned(self, prefix: str) -> list[tuple[str, str]]:
        """Return [(name, job_id), ...] for in-flight jobs whose name starts with `prefix`."""
        return [
            (name, entry[0])
            for name, entry in self._registry.items()
            if name.startswith(prefix)
        ]

    def stop_all_owned(self, prefix: str) -> list[tuple[str, str]]:
        """Stop every in-flight job whose name starts with `prefix`.

        Sends SIGTERM to each process group, then SIGKILL after a brief
        grace window. Returns the list of (name, job_id) that were
        targeted.
        """
        targets = self.snapshot_owned(prefix)
        for name, _ in targets:
            entry = self._registry.get(name)
            if entry is None:
                continue
            _, proc, _, _, _ = entry
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        # Best-effort SIGKILL after a short grace (synchronous, blocking).
        time.sleep(2.0)
        for name, _ in targets:
            entry = self._registry.get(name)
            if entry is None:
                continue
            _, proc, _, _, _ = entry
            if proc.returncode is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        return targets

    # ── Internals ───────────────────────────────────────────────────────────

    async def _terminate_process_group(self, proc: asyncio.subprocess.Process, kill_grace_s: float) -> None:
        """SIGTERM the process group, wait grace, SIGKILL if still alive."""
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=kill_grace_s)
        except asyncio.TimeoutError:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                pass

    @staticmethod
    def _close_log(proc: asyncio.subprocess.Process) -> None:
        log_fh = getattr(proc, "_log_fh", None)
        if log_fh is not None:
            try:
                log_fh.flush()
                log_fh.close()
            except Exception:
                pass


__all__ = ["LocalScheduler"]
