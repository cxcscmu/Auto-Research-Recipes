"""Path constants and system-wide knobs for the multi-agent harness, task-agnostic.

Filesystem layout. The harness keeps everything on local disk: the
authoritative blackboard, per-specialist editable workdirs, snapshots,
and locks all live under `MAGENT_LOCAL_ROOT` (default `./magent_state`).
Trial subprocesses run on the same host through the bundled
`LocalScheduler`.

Task-specific extension. This module reads task-specific knobs
(`pkg_root`, `all_domains`, baseline filename) via
`agent_core.current_adapter()`. Each task package registers a
`TaskAdapter` on import so this module can resolve them. Per-task
constants such as task data paths or venv overrides live in the task
package's own `harness/config.py`, which re-exports from this module
plus adds those constants.

Nothing here touches the filesystem at import time except for lazy
loading of `<task_pkg>/swarm_config.json` on first read of
scheduler-priority or model-routing knobs. Call `ensure_dirs()` once
from the supervisor before any specialist starts.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    """Read an integer environment override, with a clear error on invalid input."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


# ── Filesystem roots ──────────────────────────────────────────────────────────

# Local root holds the authoritative blackboard, snapshots, and the
# specialists' editable workdirs. Operators can override per machine.
LOCAL_ROOT = Path(os.environ.get(
    "MAGENT_LOCAL_ROOT",
    str(Path.cwd() / "magent_state"),
)).expanduser()

# ── Blackboard layout ─────────────────────────────────────────────────────────

BLACKBOARD_DIR = LOCAL_ROOT / "blackboard"
WORKDIRS_ROOT  = LOCAL_ROOT / "workdirs"
SNAPSHOTS_DIR  = BLACKBOARD_DIR / "snapshots"
LOCKS_DIR      = BLACKBOARD_DIR / "locks"

LEADERBOARD_MD = BLACKBOARD_DIR / "LEADERBOARD.md"
KNOWLEDGE_MD   = BLACKBOARD_DIR / "KNOWLEDGE.md"
TREE_TSV       = BLACKBOARD_DIR / "tree.tsv"
RESULTS_TSV    = BLACKBOARD_DIR / "results.tsv"
BEST_JSON      = BLACKBOARD_DIR / "best.json"
STOP_FLAG      = BLACKBOARD_DIR / "stop.flag"

# ── Job naming ────────────────────────────────────────────────────────────────

DEFAULT_JOB_NAME_PREFIX = "apg"

# Each task picks its own short prefix (1..4 chars) so dashboards and
# the supervisor's shutdown chain can distinguish concurrent swarms.
# PG keeps "apg", CIFAR uses "cif", NC uses "nc". `make_job_name(...)`
# reads the active adapter at call time; the constant above is only
# the fallback when no adapter is registered.

# ── Swarm config (file-driven, editable without exports) ─────────────────────
#
# `<task_pkg>/swarm_config.json` is the canonical place to tune
# per-specialist knobs that would otherwise need a fleet of env vars.
# Used for the per-specialist scheduler priority and Claude model
# selection. The file lives next to the task's editable recipe and is
# committed with the package so a deployment can edit it in place.
#
# Resolution order for each knob:
#   1. swarm_config.json value, if present
#   2. matching MAGENT_* env var (back-compat)
#   3. hard default
#
# Missing file: silent (all defaults). File present but unparseable:
# logged warning and treated as empty; the supervisor never aborts on
# config-file shape.


def _load_swarm_config() -> dict:
    """Read `<task_pkg>/swarm_config.json` from the active adapter's pkg_root."""
    from agent_core import current_adapter
    try:
        path = current_adapter().pkg_root / "swarm_config.json"
    except RuntimeError:
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as e:
        logging.getLogger(__name__).warning(
            "swarm_config.json present at %s but unreadable (%s); "
            "falling back to env vars and hard defaults",
            path, e,
        )
        return {}
    return data if isinstance(data, dict) else {}


# Lazy swarm_config loading. The adapter may not be registered when
# this module first imports. We compute on first access and cache; if
# the adapter is not yet registered, `_load_swarm_config` returns {}
# and we KEEP it uncached so a later access (post-registration)
# re-evaluates against the file.

_SWARM_CFG_CACHE: "dict | None" = None


def _swarm_cfg() -> dict:
    global _SWARM_CFG_CACHE
    if _SWARM_CFG_CACHE is not None:
        return _SWARM_CFG_CACHE
    from agent_core import _active_adapter
    if _active_adapter is None:
        return {}
    cfg = _load_swarm_config()
    _SWARM_CFG_CACHE = cfg
    return cfg


def scheduler_priority_for(specialist: str) -> int:
    """Return the effective scheduler priority for a given specialist.

    Lookup: `swarm_config.json[scheduler_priority][overrides][specialist]`
    → `swarm_config.json[scheduler_priority][default]`
    → `MAGENT_SCHEDULER_PRIORITY` env
    → 10.

    Returned values are forwarded to whichever `Scheduler` backend is
    in use; `LocalScheduler` ignores them, while a cluster backend may
    use them as queue priorities.
    """
    cfg = _swarm_cfg()
    prio_cfg = cfg.get("scheduler_priority", {})
    if not isinstance(prio_cfg, dict):
        prio_cfg = {}
    overrides = {
        str(k): int(v) for k, v in (prio_cfg.get("overrides") or {}).items()
    }
    if specialist in overrides:
        return overrides[specialist]
    return int(prio_cfg.get("default", _env_int("MAGENT_SCHEDULER_PRIORITY", 10)))


def model_for(specialist: str) -> str:
    """Return the effective Claude model id for a given specialist.

    Lookup: swarm_config.json overrides → swarm_config.json default →
    "claude-opus-4-7".
    """
    cfg = _swarm_cfg()
    model_cfg = cfg.get("model", {})
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    overrides = {
        str(k): str(v) for k, v in (model_cfg.get("overrides") or {}).items()
    }
    if specialist in overrides:
        return overrides[specialist]
    return str(model_cfg.get("default", "claude-opus-4-7"))


# ── bwrap sandbox probe ──────────────────────────────────────────────────────
#
# Resolution for "should sandbox be disabled?":
#   1. MAGENT_DISABLE_SANDBOX=1/true/yes/on  → disable
#   2. MAGENT_DISABLE_SANDBOX=0/false/no/off → enable
#   3. unset                                  → probe + container detection;
#      auto-disable if EITHER probe fails OR running inside a container
#      runtime where the SDK's nested-userns proc-mount is known flaky.
#
# Why two-signal: the bare bwrap probe (host-mount T5 pattern) detects
# whether basic unprivileged-userns + mount works. The SDK's bundled
# CLI uses a pivot_root + nested proc-mount pattern that hits a
# different kernel code path. In LXC and similar containers this can
# pass the probe but fail the SDK's actual mount intermittently under
# multi-agent contention. Auto-disabling on container detection is
# the only reliable way to avoid this flake; operator can force-enable
# via MAGENT_DISABLE_SANDBOX=0 if they have verified their config.

_sandbox_decision: "bool | None" = None

_CONTAINER_VIRT_DISABLE = frozenset({
    "lxc", "lxc-libvirt", "docker", "podman",
    "openvz", "wsl", "rkt", "systemd-nspawn",
    "container-other",
})


def _bwrap_pivot_proc_works() -> bool:
    """Probe whether bwrap can do its standard sandbox setup.

    Tests the typical mount profile the SDK bundled CLI needs: host fs
    bound, fresh /dev, fresh /proc, fresh /tmp. Returns True iff the
    5-second self-test exits 0. Anything else (bwrap missing, timeout,
    non-zero exit, OSError) returns False; the safe default is to
    assume the sandbox is broken and disable it.
    """
    import shutil
    import subprocess
    log = logging.getLogger(__name__)
    bwrap_path = shutil.which("bwrap")
    if not bwrap_path:
        log.warning("[bwrap probe] bwrap binary NOT FOUND in PATH")
        return False
    cmd = [
        "bwrap",
        "--bind", "/", "/",
        "--dev", "/dev",
        "--proc", "/proc",
        "--tmpfs", "/tmp",
        "/usr/bin/true",
    ]
    log.info("[bwrap probe] testing: %s  (binary=%s)", " ".join(cmd), bwrap_path)
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=5, text=True)
    except subprocess.TimeoutExpired:
        log.warning("[bwrap probe] result: TIMEOUT after 5s")
        return False
    except OSError as e:
        log.warning("[bwrap probe] result: OSError %s", e)
        return False
    if r.returncode == 0:
        log.info("[bwrap probe] result: OK  (rc=0, sandbox available)")
        return True
    stderr_lines = [line for line in (r.stderr or "").splitlines() if line.strip()]
    tail = "  |  ".join(stderr_lines[-2:])[:240] if stderr_lines else "(no stderr)"
    log.warning("[bwrap probe] result: FAILED  (rc=%d)  stderr: %s",
                r.returncode, tail)
    return False


def _detect_container_virt() -> str:
    """Return the systemd-detect-virt value, or "" if undetectable."""
    import shutil
    import subprocess
    if shutil.which("systemd-detect-virt"):
        try:
            r = subprocess.run(
                ["systemd-detect-virt"],
                capture_output=True, timeout=2, text=True,
            )
            return (r.stdout or "").strip().lower()
        except (subprocess.TimeoutExpired, OSError):
            pass
    try:
        with open("/proc/1/cgroup") as f:
            cg = f.read().lower()
        for token in ("lxc", "docker", "podman", "kubepods"):
            if token in cg:
                return token
    except OSError:
        pass
    return ""


def should_disable_sandbox() -> bool:
    """Return True if SDK sandbox should be off for this run.

    Cached per process. Operator can force either direction via
    `MAGENT_DISABLE_SANDBOX`; absent that, two-signal logic applies.
    """
    global _sandbox_decision
    if _sandbox_decision is not None:
        return _sandbox_decision

    log = logging.getLogger(__name__)
    env = os.environ.get("MAGENT_DISABLE_SANDBOX", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        _sandbox_decision = True
        log.warning(
            "[sandbox] DECISION: DISABLED  (operator forced via MAGENT_DISABLE_SANDBOX=%s)",
            env,
        )
        return _sandbox_decision
    if env in ("0", "false", "no", "off"):
        _sandbox_decision = False
        log.info(
            "[sandbox] DECISION: ENABLED  (operator forced via MAGENT_DISABLE_SANDBOX=%s)",
            env,
        )
        return _sandbox_decision

    works = _bwrap_pivot_proc_works()
    virt = _detect_container_virt()
    in_container = virt in _CONTAINER_VIRT_DISABLE
    log.info(
        "[container detect] systemd-detect-virt -> %r  (problematic-container=%s)",
        virt or "(unknown)", in_container,
    )

    if not works:
        _sandbox_decision = True
        log.warning(
            "[sandbox] DECISION: DISABLED  (auto, bwrap probe failed). "
            "agents/hooks.block_bash_writes hook is the compensating control. "
            "Set MAGENT_DISABLE_SANDBOX=0 to force-enable.",
        )
    elif in_container:
        _sandbox_decision = True
        log.warning(
            "[sandbox] DECISION: DISABLED  (auto, bwrap probe OK but running "
            "inside %r where the SDK pivot_root + nested proc-mount can be "
            "intermittently denied by the kernel under multi-agent load). "
            "agents/hooks.block_bash_writes hook is the compensating control. "
            "Set MAGENT_DISABLE_SANDBOX=0 to force-enable if you have "
            "verified the SDK works in your container config.",
            virt,
        )
    else:
        _sandbox_decision = False
        log.info(
            "[sandbox] DECISION: ENABLED  (auto, probe OK + non-container "
            "environment %r)",
            virt or "(unknown)",
        )
    return _sandbox_decision


# ── Job naming ────────────────────────────────────────────────────────────────

def active_job_name_prefix() -> str:
    """Return the job-name prefix used by both name creation and ownership checks.

    Single source of truth so `make_job_name` (job creation), the
    scheduler's `stop_all_owned(prefix)` (shutdown chain), and any
    dashboard / log-grep all agree.

    Resolution order:
      1. `MAGENT_JOB_NAME_PREFIX` env var (set by the supervisor's
         `--job-name-prefix` CLI). Lets two supervisors of the same
         task run in parallel without colliding.
      2. The active task adapter's `job_name_prefix` (PG="apg",
         CIFAR="cif", NC="nc").
      3. Module-level fallback `DEFAULT_JOB_NAME_PREFIX = "apg"` when
         no adapter is registered.
    """
    override = os.environ.get("MAGENT_JOB_NAME_PREFIX", "").strip()
    if override:
        return override
    try:
        from agent_core import current_adapter
        return current_adapter().job_name_prefix
    except (RuntimeError, AttributeError):
        return DEFAULT_JOB_NAME_PREFIX


def make_job_name(domain: str, trial_id: int) -> str:
    """Return `<prefix>-<domain[:4]>-NNNN`.

    `trial_id` is the monotonic counter from `blackboard.next_exp_id()`,
    zero-padded to four digits so the name sorts lexically.
    """
    if trial_id < 0 or trial_id > 9999:
        raise ValueError(f"trial_id {trial_id} out of [0, 9999]")
    return f"{active_job_name_prefix()}-{domain[:4]}-{trial_id:04d}"


# ── Session model ────────────────────────────────────────────────────────────

DOER_THINKING_BUDGET_TOKENS    = 8_000
ANALYST_THINKING_BUDGET_TOKENS = 4_000

# Termination: stop when wall-clock exceeds DEADLINE OR when no new
# score improvement for NO_IMPROVEMENT_GRACE_S. OR-semantics, not AND.
DEADLINE_HOURS              = 48
NO_IMPROVEMENT_GRACE_S      = 4 * 3600

# Run-trial subprocess soft limit; larger than single-stage tasks
# because run_trial does train + pack + eval, each with its own budget.
TRIAL_WALL_BUDGET_S         = 2_100


def ensure_dirs() -> None:
    """Create blackboard scaffolding. Idempotent; safe to call at every startup."""
    for d in (BLACKBOARD_DIR, WORKDIRS_ROOT, SNAPSHOTS_DIR, LOCKS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _all_domains() -> tuple[str, ...]:
    """Resolve all_domains via the active task adapter."""
    from agent_core import current_adapter
    return current_adapter().all_domains


def workdir_for(domain: str) -> Path:
    """Return (but do not create) the specialist's private workdir."""
    all_doms = _all_domains()
    if domain not in all_doms:
        raise ValueError(f"unknown domain {domain!r}, must be one of {all_doms}")
    return WORKDIRS_ROOT / f"workdir_{domain}"
