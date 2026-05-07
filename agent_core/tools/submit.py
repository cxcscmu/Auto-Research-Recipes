"""submit_trial — task-agnostic trial pipeline against a Scheduler.

Flow when a specialist calls this:

  1. Stage. Copy task helper scripts plus the run script plus, on the
     first iter only, the baseline seed into the workdir. The file
     list, seed, and run script all come from the active task adapter.
  2. Preflight. Local syntax_check plus task-specific size_check. On
     fail, record a `preflight_crash` or `size_blocked` row and return
     early without burning GPU time.
  3. Execute. Submit the workdir to the configured `Scheduler`. The
     bundled `LocalScheduler` runs `bash run_trial.sh <workdir>` as
     a subprocess on the local node.
  4. Parse + record. Read the per-trial result jsonl, classify the
     outcome, and append a row to the blackboard TSV.

Events emitted, in order, for one full trial:
  submit_trial_called → stage_ok → preflight_ok →
  job_submit → job_terminal → classify_done.
On preflight failure: submit_trial_called → stage_ok → preflight_fail
or preflight_block, and the trial returns early.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from pathlib import Path
from typing import Any, Optional

from . import tool
from .code_inspect import _syntax_check_impl
from ..harness import blackboard, config, events, tracker
from ..harness.scheduler import JobNotFoundError, JobResult, is_terminal


# ── Tunables (task-agnostic) ────────────────────────────────────────────────

_DEFAULT_TIMEOUT_S = 4 * 3600    # outer wallclock cap for one trial


# ── Adapter helpers ──────────────────────────────────────────────────────────

def _adapter():
    from agent_core import current_adapter
    return current_adapter()


def _scheduler():
    """Return the process-wide scheduler. Lazy import + singleton."""
    from agent_core.harness import get_scheduler
    return get_scheduler()


# ── Shared MCP wrapper ───────────────────────────────────────────────────────

def _mcp(result: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(result, default=str)}]}


# ── Staging ──────────────────────────────────────────────────────────────────

def _stage_workdir(workdir: Path, pkg_root: Path) -> None:
    """Seed baseline source on first iter, plus refresh helper scripts.

    Adapter properties consulted:
      seed_file       single editable file copied if missing
      editable_tree   optional directory copied recursively if absent in workdir
      stage_files     helper scripts mtime-refreshed every call
    """
    workdir.mkdir(parents=True, exist_ok=True)

    adapter = _adapter()
    seed_file = adapter.seed_file
    target = workdir / seed_file
    if not target.exists():
        seed_src = pkg_root / seed_file
        if seed_src.is_file():
            target.write_bytes(seed_src.read_bytes())

    tree = adapter.editable_tree
    if tree:
        tree_dst = workdir / tree
        if not tree_dst.exists():
            tree_src = pkg_root / tree
            if tree_src.is_dir():
                shutil.copytree(tree_src, tree_dst, symlinks=False)

    # Refresh helper scripts unconditionally every call. Helpers are
    # harness-trusted code (run_trial.sh, run_classify.py, profile_pipeline.py);
    # they form the boundary between agent-edited recipe and the
    # measurement path, so a stale-but-newer-mtime workdir copy must not
    # survive. Cost is one filesystem write per stage_file per iter.
    for rel_src, dst_name in adapter.stage_files:
        src = pkg_root / rel_src
        if src.is_file():
            dst = workdir / dst_name
            dst.write_bytes(src.read_bytes())


def _clear_local_trial_outputs(workdir: Path) -> None:
    """Remove stale trial outputs so parsing never reads old artifacts."""
    for name in _adapter().trial_output_dirs:
        shutil.rmtree(workdir / name, ignore_errors=True)


# ── Result-jsonl lookup ──────────────────────────────────────────────────────

def _find_result_jsonl(workdir: Path) -> Optional[Path]:
    """run_trial.sh + run_classify.py emit full_eval_results/<workdir-name>/run_seed0.jsonl."""
    fe = workdir / "full_eval_results"
    if not fe.is_dir():
        return None
    for sub in fe.iterdir():
        candidate = sub / "run_seed0.jsonl"
        if candidate.is_file():
            return candidate
    return None


def _find_result_log(workdir: Path) -> Optional[Path]:
    fe = workdir / "full_eval_results"
    if not fe.is_dir():
        return None
    for sub in fe.iterdir():
        candidate = sub / "run_seed0.log"
        if candidate.is_file():
            return candidate
    return None


# ── Submit entry point ──────────────────────────────────────────────────────

async def _submit_trial_impl(
    specialist: str,
    hypothesis: str,
    expected_delta: str,
    parent_exp: str,
    notes: str = "",
    repo_root: Optional[str] = None,
) -> dict[str, Any]:
    """Stage → preflight → submit-to-scheduler → parse + record."""
    adapter = _adapter()
    if specialist not in adapter.all_domains:
        return {"ok": False, "error": f"unknown specialist {specialist!r}"}

    events.emit(specialist, "submit_trial_called",
                hypothesis=(hypothesis or "")[:80], parent_exp=parent_exp)

    workdir = config.workdir_for(specialist)
    workdir.mkdir(parents=True, exist_ok=True)
    root = Path(repo_root) if repo_root else adapter.pkg_root

    early = await _stage_and_preflight(
        specialist, hypothesis, expected_delta, parent_exp, workdir, root,
    )
    if early is not None:
        return early

    exec_result = await _execute_via_scheduler(
        specialist, parent_exp, hypothesis, expected_delta, workdir, adapter,
    )
    if exec_result.get("early_return"):
        return exec_result["payload"]

    return await _finalize_trial(
        specialist, hypothesis, expected_delta, parent_exp,
        workdir,
        label=exec_result["label"],
        phase=exec_result["phase"],
        exit_code=exec_result["exit_code"],
        notes=notes,
        adapter=adapter,
    )


# ── Stage + preflight ───────────────────────────────────────────────────────

async def _stage_and_preflight(
    specialist: str,
    hypothesis: str,
    expected_delta: str,
    parent_exp: str,
    workdir: Path,
    root: Path,
) -> Optional[dict[str, Any]]:
    """Run stage + syntax + size_check. Return None to continue, or an
    early-return dict on preflight failure."""
    _stage_workdir(workdir, root)
    await asyncio.to_thread(_clear_local_trial_outputs, workdir)
    events.emit(specialist, "stage_ok", workdir=str(workdir))

    syn = _syntax_check_impl(str(workdir))
    if not syn.get("ok"):
        events.emit(specialist, "preflight_fail", reason="syntax",
                    err=(syn.get("error") or "")[:120])
        row = await asyncio.to_thread(
            blackboard.record_trial,
            specialist=specialist, domain=specialist,
            parent_exp=parent_exp, hypothesis=hypothesis,
            expected_delta=expected_delta,
            validate_row=tracker.empty_validate_row(status="preflight_crash"),
            job_name="",
            workdir=workdir,
            notes=f"syntax: {syn.get('error','')[:200]}",
            keep_decision=False,
        )
        return {"ok": True, "preflight": "syntax_error", **row}

    sz = await asyncio.to_thread(_adapter().size_check, str(workdir))
    if sz.get("ok") and sz.get("verdict") == "block":
        events.emit(specialist, "preflight_block", reason="size",
                    size_bytes=sz.get("total_bytes"),
                    limit_bytes=sz.get("limit_bytes"))
        size_row = tracker.empty_validate_row(status="size_blocked")
        size_row["artifact_bytes"] = str(sz.get("total_bytes", ""))
        row = await asyncio.to_thread(
            blackboard.record_trial,
            specialist=specialist, domain=specialist,
            parent_exp=parent_exp, hypothesis=hypothesis,
            expected_delta=expected_delta,
            validate_row=size_row,
            job_name="",
            workdir=workdir,
            notes=(
                f"preflight size={sz.get('total_bytes')} "
                f"limit={sz.get('limit_bytes')} "
                f"(code={sz.get('code_bytes')} "
                f"model={sz.get('model_bytes') or '?'}); "
                "no GPU time used."
            ),
            keep_decision=False,
        )
        return {"ok": True, "preflight": "size_blocked", **row}

    events.emit(specialist, "preflight_ok",
                size_bytes=sz.get("total_bytes") if sz.get("ok") else None)
    return None


# ── Scheduler execution ─────────────────────────────────────────────────────

async def _execute_via_scheduler(
    specialist: str,
    parent_exp: str,
    hypothesis: str,
    expected_delta: str,
    workdir: Path,
    adapter,
) -> dict[str, Any]:
    """Submit the workdir to the configured Scheduler.

    Returns either {"early_return": True, "payload": ...} on a
    submit-time failure that needs to be forwarded directly, or
    {"label", "phase", "exit_code"} on successful execution. Parse +
    record happens in `_finalize_trial`.
    """
    scheduler = _scheduler()
    sched_cfg = adapter.scheduler_config

    trial_id = int(await asyncio.to_thread(tracker.next_exp_id))
    job_name = config.make_job_name(specialist, trial_id)

    env: dict[str, str] = {"WORKDIR": str(workdir)}
    if "cuda_visible_devices" in sched_cfg:
        env["CUDA_VISIBLE_DEVICES"] = str(sched_cfg["cuda_visible_devices"])
    # Per-task node env hook (kept for adapters that need to inject
    # MAGENT_* variables into run_trial.sh, e.g. data paths).
    pod_env = getattr(adapter, "pod_env_for_trial", None) or {}
    for k, v in pod_env.items():
        env[k] = str(v)

    timeout_s = float(sched_cfg.get("timeout_s", _DEFAULT_TIMEOUT_S))
    priority = config.scheduler_priority_for(specialist)

    events.emit(specialist, "job_submit",
                job=job_name, trial_id=trial_id, priority=priority)
    try:
        job = await scheduler.submit(
            name=job_name,
            workdir=workdir,
            script=adapter.run_script,
            env=env,
            timeout_seconds=timeout_s,
        )
    except Exception as e:
        events.emit(specialist, "job_submit_fail", job=job_name, err=str(e)[:120])
        row = await asyncio.to_thread(
            blackboard.record_trial,
            specialist=specialist, domain=specialist,
            parent_exp=parent_exp, hypothesis=hypothesis,
            expected_delta=expected_delta,
            validate_row=tracker.empty_validate_row(status="preflight_crash"),
            job_name=job_name,
            workdir=workdir,
            notes=f"scheduler submit failed: {e}",
            keep_decision=False,
        )
        return {"early_return": True, "payload":
                {"ok": False, "preflight": "scheduler_submit_failed", **row}}

    try:
        result: JobResult = await scheduler.wait(job)
    except JobNotFoundError as e:
        events.emit(specialist, "job_missing", job=job_name, err=str(e)[:120])
        return {"label": job_name, "phase": "failed", "exit_code": None}

    events.emit(specialist, "job_terminal",
                job=job_name, phase=result.phase,
                exit_code=result.exit_code, elapsed_s=int(result.elapsed_s))
    return {"label": job_name, "phase": result.phase, "exit_code": result.exit_code}


# ── Parse + record ──────────────────────────────────────────────────────────

async def _finalize_trial(
    specialist: str,
    hypothesis: str,
    expected_delta: str,
    parent_exp: str,
    workdir: Path,
    *,
    label: str,
    phase: str,
    exit_code: Optional[int],
    notes: str,
    adapter,
) -> dict[str, Any]:
    """Parse run_seed*.jsonl + record_trial + emit classify_done."""

    def _resolve_excerpt_log() -> Optional[Path]:
        primary = _find_result_log(workdir)
        if primary is not None and primary.is_file() and primary.stat().st_size > 0:
            return primary
        # Fallback paths that the run script may have written before the
        # primary result log existed.
        for fallback in ("run.log", "subprocess.log"):
            p = workdir / fallback
            if p.is_file() and p.stat().st_size > 0:
                return p
        return primary

    jsonl = _find_result_jsonl(workdir)
    if jsonl is None:
        log_path = _resolve_excerpt_log()
        excerpt = ""
        if log_path is not None:
            excerpt = (await asyncio.to_thread(tracker.extract_crash_excerpt, log_path)) or ""
        validate_row = tracker.empty_validate_row(status="crash")
        notes_final = notes or ""
        if excerpt:
            notes_final = f"{notes_final} | {excerpt}" if notes_final else excerpt
    else:
        validate_row = await asyncio.to_thread(tracker.parse_validate_result, jsonl)
        notes_final = notes
        if validate_row["status"] in ("crash", "preflight_crash"):
            log_path = _resolve_excerpt_log()
            if log_path is not None:
                excerpt = await asyncio.to_thread(tracker.extract_crash_excerpt, log_path)
                if excerpt:
                    notes_final = f"{notes_final} | {excerpt}" if notes_final else excerpt

    row = await asyncio.to_thread(
        blackboard.record_trial,
        specialist=specialist, domain=specialist,
        parent_exp=parent_exp, hypothesis=hypothesis,
        expected_delta=expected_delta,
        validate_row=validate_row,
        job_name=label,
        workdir=workdir,
        notes=notes_final,
    )

    score_field = adapter.score_field
    events.emit(specialist, "classify_done",
                exp_id=row.get("exp_id"),
                status=row.get("status"),
                **{score_field: row.get(score_field)},
                delta=row.get("delta_vs_best"))

    return {
        "ok":         True,
        "preflight":  "passed",
        "job_phase":  phase,
        "exit_code":  exit_code,
        **row,
    }


# ── Async @tool wrapper (SDK-facing) ─────────────────────────────────────────

@tool(
    "submit_trial",
    (
        "Submit this specialist's current editable recipe to a real "
        "8xH100 evaluation through the configured scheduler. Runs a "
        "local syntax + size preflight first; failures are recorded "
        "without burning GPU time. On success, blocks until the job "
        "finishes, then writes a row to the blackboard TSV and returns "
        "{exp_id, status, score_field, delta_vs_best, artifact_bytes, "
        "train_s, eval_s, total_s, snapshot_path, job_name, notes}. "
        "Status is one of: keep | discard | crash | size_blocked | "
        "preflight_crash | eval_budget_overrun | train_budget_overrun."
    ),
    {
        "type": "object",
        "properties": {
            "specialist":     {"type": "string",
                               "description": "Your domain key, e.g. 'arch'."},
            "hypothesis":     {"type": "string",
                               "description": "One-sentence description of what you changed."},
            "expected_delta": {"type": "string",
                               "description": "Signed estimate, e.g. '-0.002'."},
            "parent_exp":     {"type": "string",
                               "description": "exp_id you rooted from (usually best.json)."},
            "notes":          {"type": "string",
                               "description": "Optional free-form rationale.",
                               "default": ""},
        },
        "required": ["specialist", "hypothesis", "expected_delta", "parent_exp"],
    },
)
async def submit_trial(args: dict[str, Any]) -> dict[str, Any]:
    result = await _submit_trial_impl(
        specialist=args["specialist"],
        hypothesis=args["hypothesis"],
        expected_delta=args["expected_delta"],
        parent_exp=args["parent_exp"],
        notes=args.get("notes", ""),
    )
    return _mcp(result)
