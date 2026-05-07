"""CLI entrypoint for the supervisor.

Usage on the local node:

    # Default: full swarm. Per-specialist scheduler priority comes from
    # the task package's swarm_config.json.
    python -m multi_agent_pg.supervisor \\
        --deadline-hours 48 \\
        --no-improvement-hours 4

    # Narrower subset (debugging, or to match past runs):
    python -m multi_agent_pg.supervisor \\
        --specialists arch,opt,quant,meta

Reads the blackboard, bootstraps a baseline row if none exists, spawns
one coroutine per specialist with a launch stagger, runs until
termination, prints an end-of-run summary, and on SIGINT or SIGTERM
stops every in-flight job before exit.
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import logging
import math
import os
import signal
import sys
from typing import Optional


def _finite_float(s: str) -> float:
    """argparse type: float, but reject nan/inf so the CLI override path
    can't poison best.json or results.tsv. Mirrors the calibrate_baseline
    helper's check; both layers are kept on purpose."""
    v = float(s)
    if not math.isfinite(v):
        raise argparse.ArgumentTypeError(
            f"baseline score must be finite, got {s!r}"
        )
    return v

# Harness imports are DEFERRED. Reason: agent_core.harness.config reads
# MAGENT_LOCAL_ROOT at module import time, which freezes that constant.
# nc/cifar __init__.py uses os.environ.setdefault to install task-specific
# defaults; if we imported config here, those setdefaults would arrive
# too late and the direct path
# (`MAGENT_TASK=cifar python -m agent_core.supervisor`) would write into
# the wrong blackboard. Imports happen inside main() after
# _ensure_task_package_registered.

_LOG = logging.getLogger("agent_core.supervisor.main")
_SHUTDOWN_COUNT = 0


# ── Task package selection ──────────────────────────────────────────────────
#
# When agent_core.supervisor is invoked directly we need to import the
# task package BEFORE building argparse defaults (those default values
# come from the active adapter). Resolution order:
#
#   1. `--task <name>` CLI flag
#   2. `MAGENT_TASK` env var
#   3. None (caller already imported a task package, e.g. through a
#      task wrapper at multi_agent_pg/supervisor/__main__.py)

_TASK_PKG_MAP = {
    "pg":     "multi_agent_pg",
    "nc":     "multi_agent_nc",
    "cifar":  "multi_agent_cifar",
}


def _peek_named_arg(argv: Optional[list[str]], name: str) -> Optional[str]:
    """Return the value of `--<name> X` or `--<name>=X` from argv, without consuming it."""
    args = argv if argv is not None else sys.argv[1:]
    for i, a in enumerate(args):
        if a == f"--{name}" and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return None


def _peek_task_arg(argv: Optional[list[str]]) -> Optional[str]:
    return _peek_named_arg(argv, "task")


def _apply_state_root_from_argv(argv: Optional[list[str]]) -> None:
    """Pre-parse --state-root from argv and overwrite MAGENT_LOCAL_ROOT.

    Uses os.environ[k] = v (not setdefault) so the CLI value wins over
    any prior shell env. Idempotent. Called from main() before any
    harness import.
    """
    val = _peek_named_arg(argv, "state-root")
    if val:
        os.environ["MAGENT_LOCAL_ROOT"] = os.path.expanduser(val)


def _apply_job_name_prefix_from_argv(argv: Optional[list[str]]) -> None:
    """Pre-parse --job-name-prefix from argv and overwrite MAGENT_JOB_NAME_PREFIX.

    Used to run two supervisors of the same task in parallel without
    colliding on job names. Pass --job-name-prefix apga to one and
    --job-name-prefix apgb to the other so each supervisor's
    `scheduler.stop_all_owned(prefix)` cleanup only affects its own jobs.
    """
    val = _peek_named_arg(argv, "job-name-prefix")
    if val:
        os.environ["MAGENT_JOB_NAME_PREFIX"] = val


def _ensure_task_package_registered(argv: Optional[list[str]]) -> None:
    """Import the active task package so `register_task_adapter` runs.

    No-op if an adapter is already registered. Otherwise resolves
    via --task / MAGENT_TASK / raise.
    """
    from agent_core import _active_adapter
    if _active_adapter is not None:
        return
    name = _peek_task_arg(argv) or os.environ.get("MAGENT_TASK", "").strip().lower()
    if not name:
        raise SystemExit(
            "no task package registered. Set MAGENT_TASK=pg|nc|cifar, "
            "pass --task <name>, or invoke via a task wrapper "
            "(e.g. `python -m multi_agent_pg.supervisor`)"
        )
    pkg = _TASK_PKG_MAP.get(name)
    if pkg is None:
        raise SystemExit(
            f"unknown --task / MAGENT_TASK value {name!r} "
            f"(known: {sorted(_TASK_PKG_MAP)})"
        )
    __import__(pkg)


# ── Shutdown ────────────────────────────────────────────────────────────────


def _cleanup_owned_jobs() -> None:
    """Best-effort: stop every in-flight job whose name starts with the
    active job-name prefix. Called from SIGINT/SIGTERM handlers and from
    atexit so Ctrl+C / kill / normal exit all converge on a clean state.
    """
    try:
        from ..harness import config, get_scheduler
    except ImportError:
        return
    prefix = config.active_job_name_prefix()
    scheduler = get_scheduler()
    pairs = scheduler.snapshot_owned(prefix)
    if not pairs:
        return
    _LOG.warning(
        "shutdown: stopping %d active job(s) with prefix %r: %s",
        len(pairs), prefix,
        ", ".join(f"{n}[{jid}]" for n, jid in pairs),
    )
    stopped = scheduler.stop_all_owned(prefix)
    if stopped:
        _LOG.warning(
            "shutdown: stop issued for: %s",
            ", ".join(f"{n}[{jid}]" for n, jid in stopped),
        )


def _signal_handler(signum: int, _frame) -> None:
    """SIGINT/SIGTERM: stop in-flight jobs, write stop.flag, then re-raise.

    A second signal bypasses cleanup and force-exits in case the cleanup
    itself hangs.
    """
    global _SHUTDOWN_COUNT
    _SHUTDOWN_COUNT += 1
    sig_name = signal.Signals(signum).name
    if _SHUTDOWN_COUNT >= 2:
        sys.stderr.write(
            f"\n[shutdown] second {sig_name} received, force exit without "
            f"further cleanup\n"
        )
        sys.stderr.flush()
        os._exit(130)

    sys.stderr.write(
        f"\n[shutdown] caught {sig_name}; stopping active jobs before "
        f"exit (press {sig_name} again to force-exit)...\n"
    )
    sys.stderr.flush()

    try:
        from ..harness import config
        config.STOP_FLAG.write_text(f"{sig_name} received", encoding="utf-8")
    except OSError:
        pass

    _cleanup_owned_jobs()
    raise KeyboardInterrupt(f"{sig_name} received")


def _install_shutdown_hooks() -> None:
    signal.signal(signal.SIGINT,  _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    atexit.register(_cleanup_owned_jobs)


# ── Argparse ────────────────────────────────────────────────────────────────


def _all_domains() -> tuple[str, ...]:
    from agent_core import current_adapter
    return current_adapter().all_domains


def _build_parser() -> argparse.ArgumentParser:
    from agent_core import current_adapter
    from ..harness import config
    from . import core
    adapter = current_adapter()

    p = argparse.ArgumentParser(description="Run the multi-agent swarm supervisor.")
    p.add_argument(
        "--task",
        choices=sorted(_TASK_PKG_MAP),
        default=None,
        help=(
            "Task package selector: pg / nc / cifar. Equivalent to "
            "MAGENT_TASK env. Required when invoking agent_core.supervisor "
            "directly; ignored if you have already imported a task package."
        ),
    )
    _doer_count    = len(adapter.doer_domains)
    _analyst_count = len(adapter.analyst_domains)
    _all_count     = _doer_count + _analyst_count
    _analyst_str   = (f" + {_analyst_count} analyst{'s' if _analyst_count != 1 else ''}"
                      if _analyst_count else " (no analysts)")
    p.add_argument(
        "--specialists",
        default=",".join(_all_domains()),
        help=(
            f"Comma-separated specialist keys (subset of "
            f"DOER_DOMAINS + ANALYST_DOMAINS). Default is the full "
            f"{_all_count}-agent swarm ({_doer_count} doers{_analyst_str}). "
            f"Per-specialist scheduler priority comes from "
            f"{adapter.pkg_root.name}/swarm_config.json. Pass a narrower "
            f"CSV to debug a subset."
        ),
    )
    p.add_argument("--deadline-hours", type=float,
                   default=float(config.DEADLINE_HOURS))
    p.add_argument("--no-improvement-hours", type=float,
                   default=float(config.NO_IMPROVEMENT_GRACE_S) / 3600.0)
    p.add_argument("--launch-stagger-s", type=float,
                   default=core.LAUNCH_STAGGER_S)
    p.add_argument(
        "--state-root", type=str, default=None, metavar="PATH",
        help=(
            "Override MAGENT_LOCAL_ROOT for this run (blackboard + workdirs "
            "live under PATH/blackboard and PATH/workdirs). The flag is "
            "pre-parsed before harness imports, so it wins over both shell "
            "env and the task package's setdefault. Pass a fresh path per "
            "run when running A/B ablations side-by-side."
        ),
    )
    p.add_argument(
        "--job-name-prefix", type=str, default=None, metavar="STR",
        help=(
            "Override the scheduler job-name prefix (normally task-adapter "
            "default: 'apg' for PG, 'cif' for CIFAR, 'nc' for NC). "
            "Sets MAGENT_JOB_NAME_PREFIX. Job names become "
            "<PREFIX>-<dom[:4]>-NNNN; keep PREFIX <=4 chars. Required when "
            "running two same-task supervisors in parallel so each "
            "supervisor's shutdown cleanup only stops its own jobs."
        ),
    )
    p.add_argument(
        "--max-turns", type=int, default=None, metavar="N",
        help=(
            "Override DoerConfig.max_turns for every specialist this run "
            "(default 200). Lower values discourage in-session multi-submit. "
            "Used in the no-lineage ablation to align session shape across "
            "the A/B pair."
        ),
    )
    p.add_argument(
        "--no-lineage", action="store_true",
        help=(
            "No-lineage ablation: blank LEADERBOARD/KNOWLEDGE/Recent "
            "Activity/Saturation in the per-iteration prompt, drop "
            "read_snapshot/diff_snapshots tools, and deny Bash reads of "
            "blackboard files via the block_bash_blackboard PreToolUse "
            "hook. Sets MAGENT_NO_LINEAGE=1. The current-best exp_id and "
            "score remain visible (needed for rebase_to). Use a fresh "
            "--state-root for the ablation run so its blackboard is "
            "isolated from the lineage-on baseline."
        ),
    )
    _flag_metavar = adapter.baseline_score_flag.lstrip("-").replace("-", "_").upper()
    p.add_argument(adapter.baseline_score_flag, type=_finite_float, dest="baseline_score",
                   metavar=_flag_metavar,
                   help=f"Seed blackboard with this {adapter.score_field} as the "
                        f"baseline if no rows exist yet. Default is "
                        f"{adapter.baseline_score_default:.4f} ("
                        f"{adapter.bootstrap_hypothesis[:80]}...).")
    p.add_argument(
        "--reset-stale-workdirs", action="store_true",
        help=(
            f"Before launch, delete any workdir_<spec>/{adapter.baseline_filename} "
            f"whose sha256 differs from the package-root baseline "
            f"({adapter.pkg_root.name}/{adapter.baseline_filename}). Re-seed "
            f"happens on each specialist's first iter via _stage_workdir. "
            f"Use this after a baseline migration when specialists should "
            f"abandon their prior-era edits. Without this flag, a hash "
            f"mismatch is REPORTED at startup but not acted on."
        ),
    )
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p


def _maybe_bootstrap(baseline_score: Optional[float]) -> None:
    """Seed the blackboard on first run. Idempotent.

    For tasks with `requires_calibrated_baseline=True` (NC, CIFAR), an
    empty blackboard with no explicit `--baseline-*` flag is a hard
    error: the placeholder default would pollute early-iter
    `delta_vs_best`. Operator must first run
    `python -m multi_agent_<task>.calibrate_baseline --score X.XXX`.
    """
    from ..harness import blackboard
    if blackboard.read_best() is not None:
        return
    from agent_core import current_adapter
    adapter = current_adapter()
    score_field = adapter.score_field

    if baseline_score is None and adapter.requires_calibrated_baseline:
        pkg = adapter.pkg_root.name
        raise SystemExit(
            f"refusing to cold-start with placeholder "
            f"{score_field}={adapter.baseline_score_default:.4f}.\n"
            f"  task '{pkg}' requires a calibrated baseline; either:\n"
            f"    (a) run the unedited baseline >= 1 time, then\n"
            f"        python -m {pkg}.calibrate_baseline --score X.XXXX [--score Y.YYYY ...]\n"
            f"    (b) pass {adapter.baseline_score_flag} X.XXXX to supervisor "
            f"if you have a trusted score from a prior run and accept the "
            f"single-source-of-truth risk."
        )

    score_value = baseline_score if baseline_score is not None else adapter.baseline_score_default
    if not math.isfinite(score_value):
        raise SystemExit(
            f"refusing to bootstrap with non-finite {score_field}={score_value!r}; "
            f"check {adapter.baseline_score_flag} or "
            f"{adapter.pkg_root.name}/task_config.py:baseline_score_default"
        )
    blackboard.bootstrap_from_baseline({
        score_field:      f"{score_value:.6f}",
        "hypothesis":     adapter.bootstrap_hypothesis,
        "snapshot_path":  "",
    })


def _print_summary(s) -> None:
    print()
    print("=" * 72)
    print("Supervisor run complete")
    print("=" * 72)
    print(f"  started      : {s.started_iso}")
    print(f"  ended        : {s.ended_iso}")
    print(f"  elapsed      : {s.elapsed_s:.0f} s ({s.elapsed_s/3600:.2f} h)")
    print(f"  stop reason  : {s.stop_reason}")
    print(f"  specialists  : {', '.join(s.specialists)}")
    print(f"  iters/spec   :")
    for k, v in sorted(s.iters_per_spec.items()):
        print(f"    {k:<6} {v}")
    if s.final_best:
        from agent_core import current_adapter
        score_field = current_adapter().score_field
        print(f"  final best   : exp_{s.final_best.get('exp_id','?')} "
              f"{score_field}={s.final_best.get(score_field,'?')} "
              f"({s.final_best.get('specialist','?')})")
    else:
        print(f"  final best   : (none, no VALID runs)")
    print("=" * 72)


def main(argv: Optional[list[str]] = None) -> int:
    # `--state-root` and `--job-name-prefix` are pre-parsed BEFORE the
    # task package is imported, because each task package's __init__.py
    # may use os.environ.setdefault to install MAGENT_LOCAL_ROOT
    # defaults. setdefault is a no-op if the env is already set; by
    # writing the CLI values here, we ensure CLI wins over per-task
    # defaults.
    _apply_state_root_from_argv(argv)
    _apply_job_name_prefix_from_argv(argv)

    # Register a task adapter BEFORE building the parser or importing
    # any harness module. Parser defaults (specialists, baseline flag)
    # and config.LOCAL_ROOT both depend on env vars the task package
    # may setdefault on import.
    _ensure_task_package_registered(argv)

    from ..harness import config, credentials
    from . import core

    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    specialists = [s.strip() for s in args.specialists.split(",") if s.strip()]
    if not specialists:
        print("error: --specialists cannot be empty", file=sys.stderr)
        return 2

    if args.no_lineage:
        os.environ["MAGENT_NO_LINEAGE"] = "1"

    overrides: Optional[dict[str, "core.DoerConfig"]] = None
    if args.max_turns is not None:
        from ..agents.base import DoerConfig
        overrides = {
            s: DoerConfig(specialist=s, max_turns=args.max_turns)
            for s in specialists
        }

    credentials.ensure_api_key()
    config.ensure_dirs()
    _maybe_bootstrap(args.baseline_score)

    _install_shutdown_hooks()

    try:
        summary = asyncio.run(core.run(
            specialists,
            deadline_hours=args.deadline_hours,
            no_improvement_grace_s=args.no_improvement_hours * 3600.0,
            launch_stagger_s=args.launch_stagger_s,
            doer_cfg_overrides=overrides,
            reset_stale_workdirs=args.reset_stale_workdirs,
        ))
    except KeyboardInterrupt:
        print("\n[shutdown] supervisor exited via signal", file=sys.stderr)
        return 130
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
