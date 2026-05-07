"""TaskAdapter — abstract base for per-task customization.

Each task package (`multi_agent_pg`, `multi_agent_nc`, `multi_agent_cifar`)
implements a `TaskAdapter` subclass and registers it on import via
`agent_core.register_task_adapter(adapter)`. Core code reads
task-specific knobs through `current_adapter()`.

Day 2 status
────────────
This file defines the full abstract contract. Day 5 wires core code to
read from it; until then, the interface is a typed scaffold and most
methods are read but not yet driving behavior.

Method categories:

  paths/files   pkg_root, knowledge_dir, baseline_filename
  schema        tsv_fields, score_field, score_lower_is_better
  specialists   doer_domains, analyst_domains, specialist_classes()
  pipeline      stage_files, seed_file, run_script, trial_output_dirs,
                size_check(workdir), parse_validate_record(record)
  MD render     render_leaderboard(rows), render_knowledge(rows),
                render_tree_tsv(rows)                       (Day 5)
  tools         custom_tool_names, bind_tools()             (Day 5)
  prompts       hard_limits_section(), specialist_preamble(domain),
                keep_discard_semantics()                    (Day 4)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional


class TaskAdapter(ABC):
    """Per-task customization surface; one concrete subclass per research task."""

    # ── Paths / files ────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def pkg_root(self) -> Path:
        """Absolute path to the task package root directory.

        Used by core for knowledge file lookups, swarm_config.json
        location, baseline-seed source path, etc.
        """

    @property
    @abstractmethod
    def knowledge_dir(self) -> Path:
        """Directory holding INIT.md / SOTA_STACK.md / LESSONS.md / pr_library/."""

    @property
    @abstractmethod
    def baseline_filename(self) -> str:
        """Filename of the baseline source — e.g. "train_gpt.py" for PG."""

    # ── TSV schema ───────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def tsv_fields(self) -> list[str]:
        """Ordered list of column names for results.tsv.

        Must include 'exp_id', 'status', and the task's primary score
        field (whatever score_field returns). Order is byte-significant —
        csv.DictWriter preserves it.
        """

    @property
    @abstractmethod
    def score_field(self) -> str:
        """Name of the column holding the primary metric ("val_bpb" for PG)."""

    @property
    @abstractmethod
    def score_lower_is_better(self) -> bool:
        """True if smaller score is better (PG bits-per-byte) vs larger (CIFAR accuracy)."""

    @property
    def score_short_label(self) -> str:
        """Compact label for inline displays (lineage rows, recent activity).

        Default = score_field; tasks may override with a 3-4 char string for
        readable history (PG: "bpb", CIFAR: "acc", NC: "core"). Doesn't
        affect TSV columns — only render-side compactness.
        """
        return self.score_field

    @abstractmethod
    def parse_validate_record(self, record: dict) -> dict:
        """Convert run_classify.py's jsonl line into TSV-row fields.

        Input: dict from json.loads(record_line)
        Output: dict with keys covering at least status / score_field /
                training time / artifact size — task-specific shape, but
                must be writable into a row keyed by tsv_fields.
        """

    @abstractmethod
    def empty_validate_row(self, status: str) -> dict:
        """Empty TSV row dict carrying just `status`; other fields blank.

        Used by submit.py preflight failure paths to avoid hardcoding
        fieldname/value pairs.
        """

    # ── Specialists ──────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def doer_domains(self) -> tuple[str, ...]:
        """Tuple of doer specialist keys, e.g. ("arch","opt","ttt","quant",...)."""

    @property
    @abstractmethod
    def analyst_domains(self) -> tuple[str, ...]:
        """Tuple of analyst keys, typically ("meta",)."""

    @property
    def all_domains(self) -> tuple[str, ...]:
        """Concrete; default = doer_domains + analyst_domains."""
        return tuple(self.doer_domains) + tuple(self.analyst_domains)

    @abstractmethod
    def specialist_classes(self) -> dict[str, type]:
        """Map of domain key → DoerBase subclass.

        Order matters: callers iterate this dict and the iteration order
        affects byte-equal verification (event log order, MCP tool
        registration order). Always return a dict whose insertion order
        matches your config.DOER_DOMAINS + ANALYST_DOMAINS sequence.
        """

    # ── Pipeline / stage / size ──────────────────────────────────────────────

    @property
    @abstractmethod
    def stage_files(self) -> tuple[tuple[str, str], ...]:
        """List of (src_relative_to_pkg_root, dst_basename_in_workdir) pairs.

        Files refreshed into each specialist workdir on every iter
        (helper scripts, not the baseline source).
        """

    @property
    @abstractmethod
    def seed_file(self) -> str:
        """Filename to seed into a fresh workdir on the first iter ("train_gpt.py")."""

    @property
    def editable_tree(self) -> Optional[str]:
        """Optional directory (relative to pkg_root) recursively copied to
        workdir on first iter only — for tasks where the agent edits across
        many files (NC vendor pattern). PG/CIFAR don't override; their
        editable surface is just `seed_file`.

        When set, `_stage_workdir` recursively copies pkg_root/<editable_tree>
        to workdir/<editable_tree> if absent. multi-file tools
        (syntax_check, param_count, diff_snapshots) walk this tree.
        """
        return None

    @property
    @abstractmethod
    def run_script(self) -> str:
        """Bash entrypoint name the scheduler runs ("run_trial.sh")."""

    @property
    @abstractmethod
    def trial_output_dirs(self) -> tuple[str, ...]:
        """Directory names cleared from the workdir before each run.

        Defaults for PG: ("full_eval_results", "ckpt", "logs"). NC and
        CIFAR may differ.
        """

    @property
    def pod_env_for_trial(self) -> dict[str, str]:
        """Env vars to inject into the trial subprocess environment.

        Used by submit.py when invoking the scheduler. Default empty.
        Tasks whose `run_trial.sh` needs explicit data-path or venv-path
        environment variables (NC, CIFAR) override this property to
        return those bindings.

        Values are passed verbatim to the scheduler as part of `env=`;
        the scheduler is responsible for any quoting required by its
        backend (the bundled LocalScheduler does not need quoting).
        """
        return {}

    @abstractmethod
    def size_check(self, workdir: str) -> dict:
        """Run the task-specific preflight size check on a staged workdir.

        Return shape:
          {"ok": bool, "verdict": "ok"|"warn"|"block",
           "code_bytes": int, "model_bytes": Optional[int],
           "total_bytes": int, "limit_bytes": int}
        Tasks without a size cap may return verdict="ok" with limit_bytes=None.
        """

    # ── Tools ────────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def custom_tool_names(self) -> tuple[str, ...]:
        """Names of MCP tools exposed to the agent (used by base.py for allowed_tools)."""

    @abstractmethod
    def bind_tools(self) -> list[Any]:
        """Return the @tool-decorated callables to register in the SDK MCP server."""

    # ── Prompts (Day 4) ──────────────────────────────────────────────────────

    @abstractmethod
    def hard_limits_section(self) -> str:
        """Markdown section for "Hard limits" — task-specific numbers (e.g. 16 MB cap)."""

    @abstractmethod
    def specialist_preamble(self, domain: str) -> str:
        """Per-specialist preamble injected into the system prompt for `domain`."""

    @abstractmethod
    def keep_discard_semantics(self) -> str:
        """Markdown section explaining keep/discard/crash status semantics for this task."""

    @abstractmethod
    def build_system_prompt(self, domain: str) -> str:
        """Return the full system prompt for `domain`.

        Used by `agent_core.agents.base.DoerBase.run_once` so core
        doesn't need to know about prompts.py / GLOBAL_RULES content
        layout. Each task package supplies its own assembly (which mixes
        generic GLOBAL_RULES with task-specific preambles + knowledge MD).
        """

    # ── Bootstrap (Gap 5) ────────────────────────────────────────────────────
    #
    # When the supervisor starts against an empty blackboard, it writes a
    # synthetic "baseline" row whose score field carries the upstream
    # baseline value. Three knobs control the row + the CLI flag that
    # operators use to override it; all three were PG-hardcoded in
    # supervisor/__main__.py before Gap 5 closure.

    @property
    def baseline_score_default(self) -> float:
        """Initial score for empty-blackboard bootstrap.

        PG: 1.0284 (PR #1758 reference). nc/cifar override.
        Default raises so a forgotten override is loud.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement baseline_score_default"
        )

    @property
    def baseline_score_flag(self) -> str:
        """argparse flag name for overriding the bootstrap score.

        Includes leading '--'. PG: '--baseline-bpb'. CIFAR: '--baseline-accuracy'.
        nc: '--baseline-core'.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement baseline_score_flag"
        )

    @property
    def requires_calibrated_baseline(self) -> bool:
        """Whether cold-start MUST go through `calibrate_baseline`.

        When True and blackboard is empty AND operator did NOT pass the
        explicit `baseline_score_flag`, supervisor refuses to launch and
        prints a pointer to `python -m multi_agent_<task>.calibrate_baseline`.

        Default False = legacy behavior (PG): supervisor falls back to
        baseline_score_default placeholder. NC/CIFAR override to True
        because their defaults are placeholders / single-run guesses
        that bias early-iter delta_vs_best signal.
        """
        return False

    @property
    def bootstrap_hypothesis(self) -> str:
        """Free-form hypothesis text recorded in the bootstrap audit row.

        Becomes the `hypothesis` field of the synthetic exp_000 baseline row;
        propagates to KNOWLEDGE.md / audit_log so it stays user-visible.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement bootstrap_hypothesis"
        )

    @property
    def baseline_note(self) -> str:
        """Free-form notes string recorded in the bootstrap row's `notes` col.

        Default = "baseline reference". PG keeps the original literal
        ("seed / PR #1758 reference") for byte-equal vs the frozen
        multi_agent reference; nc/cifar may override or accept the default.
        """
        return "baseline reference"

    # ── Job identity ─────────────────────────────────────────────────────────

    @property
    def job_name_prefix(self) -> str:
        """Short prefix (1..4 chars) for scheduler job names: <prefix>-<domain[:4]>-NNNN.

        Per-task to keep concurrent swarms' job lists distinct and to
        prevent cross-task shutdown-chain false positives (the supervisor
        calls `scheduler.stop_all_owned(prefix)` at exit). PG uses "apg",
        CIFAR uses "cif", NC uses "nc". Default = "apg".
        """
        return "apg"

    # ── Scheduler config ─────────────────────────────────────────────────────

    @property
    def scheduler_config(self) -> dict:
        """Per-task scheduler configuration passed at trial submit time.

        Keys understood by the bundled `LocalScheduler`:

          * cuda_visible_devices (str, default "0,1,2,3,4,5,6,7")
              passed through to the child env as CUDA_VISIBLE_DEVICES.
          * timeout_s (int, default 7200)
              wallclock cap; SIGTERM at deadline, SIGKILL after grace.

        Backends other than LocalScheduler may read additional keys
        such as priority, partition, or image. Per-specialist priority
        is sourced from `swarm_config.json` via
        `agent_core.harness.config.scheduler_priority_for(specialist)`.
        """
        return {"cuda_visible_devices": "0,1,2,3,4,5,6,7", "timeout_s": 7200}


__all__ = ["TaskAdapter"]
