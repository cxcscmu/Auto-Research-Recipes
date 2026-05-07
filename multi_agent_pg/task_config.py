"""PGTaskAdapter — Parameter Golf instantiation of agent_core.TaskAdapter.

Day 2 status
────────────
Most concrete properties read from the existing PG modules (lazy imports
to avoid circular dependency on package import). MD render + bind_tools
are deferred to Day 5 — they currently raise NotImplementedError, which
core code does not yet exercise.

This file is intentionally a thin shim. Day 3-5 will progressively split
the underlying PG modules so that `agent_core` reads from this
adapter rather than from PG modules directly. The public surface here
should remain stable across that work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_core.task_adapter import TaskAdapter


_PKG_ROOT = Path(__file__).resolve().parent


class PGTaskAdapter(TaskAdapter):
    """Parameter Golf adapter — sub-1.07 BPB on FineWeb-edu val, 16 MB / 600 s budget."""

    # ── Paths ────────────────────────────────────────────────────────────────

    @property
    def pkg_root(self) -> Path:
        return _PKG_ROOT

    @property
    def knowledge_dir(self) -> Path:
        return _PKG_ROOT / "knowledge"

    @property
    def baseline_filename(self) -> str:
        return "train_gpt.py"

    # ── Schema ───────────────────────────────────────────────────────────────

    @property
    def tsv_fields(self) -> list[str]:
        # PG TSV schema — 18 columns. Order is byte-significant: csv.DictWriter
        # writes columns in insertion order, and downstream readers / dashboard
        # parsers expect this layout exactly.
        return [
            "exp_id",            # zero-padded, monotone across all specialists
            "timestamp",         # ISO-8601 UTC
            "specialist",        # domain key, e.g. "arch" / "opt" / "meta"
            "parent_exp",        # exp_id this mutation rooted from
            "baseline_exp",      # best exp at submit time
            "domain",            # coarse tag; currently == specialist but reserved
            "hypothesis",        # one-sentence description
            "expected_delta",    # agent's estimate (signed, str)
            "status",            # keep | discard | crash | size_blocked | ...
            "val_bpb",           # authoritative bpb from run_classify.py
            "delta_vs_best",     # signed Δ vs best.json at submit time
            "artifact_bytes",    # total packed bytes from run_classify.py
            "train_s",           # authoritative train wall, ≤600 or DQ_TRAIN
            "eval_s",            # authoritative eval wall,  ≤600 or DQ_EVAL
            "total_s",           # total_wall_s from classify()
            "job_name",       # for log-fetch after the fact
            "snapshot_path",     # relative to BLACKBOARD_DIR; empty if status≠keep
            "notes",             # free-form (trimmed crash excerpt or rationale)
        ]

    @property
    def score_field(self) -> str:
        return "val_bpb"

    @property
    def score_short_label(self) -> str:
        return "bpb"

    @property
    def score_lower_is_better(self) -> bool:
        return True

    def parse_validate_record(self, record: dict) -> dict:
        """Map run_classify JSONL dict → TSV row fields. Delegates to the
        default PG-shape implementation in core's tracker."""
        from agent_core.harness.tracker import _parse_validate_record_default
        return _parse_validate_record_default(record)

    def empty_validate_row(self, status: str) -> dict:
        """Default-empty TSV row carrying just status; matches the 7 hardcoded
        literals in the existing submit.py byte-for-byte (refactor-safe)."""
        return {
            "status":         status,
            "val_bpb":        "",
            "artifact_bytes": "",
            "train_s":        "",
            "eval_s":         "",
            "total_s":        "",
        }

    # ── Specialists ──────────────────────────────────────────────────────────

    @property
    def doer_domains(self) -> tuple[str, ...]:
        from multi_agent_pg.harness.config import DOER_DOMAINS
        return tuple(DOER_DOMAINS)

    @property
    def analyst_domains(self) -> tuple[str, ...]:
        from multi_agent_pg.harness.config import ANALYST_DOMAINS
        return tuple(ANALYST_DOMAINS)

    def specialist_classes(self) -> dict[str, type]:
        # Insertion order MUST match the registry in agents/runner.py +
        # supervisor/core.py for byte-equal events.jsonl ordering.
        from multi_agent_pg.agents import (
            arch, opt, tok, quant, ttt, curr, loss, reg,
            eval as eval_mod, meta,
        )
        return {
            "arch":  arch.ArchDoer,
            "opt":   opt.OptDoer,
            "tok":   tok.TokDoer,
            "quant": quant.QuantDoer,
            "ttt":   ttt.TTTDoer,
            "curr":  curr.CurrDoer,
            "loss":  loss.LossDoer,
            "reg":   reg.RegDoer,
            "eval":  eval_mod.EvalDoer,
            "meta":  meta.MetaDoer,
        }

    # ── Pipeline / stage / size ──────────────────────────────────────────────

    @property
    def stage_files(self) -> tuple[tuple[str, str], ...]:
        # Mirror multi_agent_pg/tools/submit.py:_REFRESH_FILES exactly.
        return (
            ("tools/pack_submission.py", "pack_submission.py"),
            ("tools/run_classify.py",    "run_classify.py"),
            ("tools/run_trainer.py",     "run_trainer.py"),
            ("run_trial.sh",             "run_trial.sh"),
        )

    @property
    def seed_file(self) -> str:
        return "train_gpt.py"

    @property
    def run_script(self) -> str:
        return "run_trial.sh"

    @property
    def trial_output_dirs(self) -> tuple[str, ...]:
        return ("full_eval_results", "ckpt", "logs")

    def size_check(self, workdir: str) -> dict:
        from multi_agent_pg.tools.code_inspect import _size_project_impl
        return _size_project_impl(workdir)

    # ── Tools ────────────────────────────────────────────────────────────────

    @property
    def custom_tool_names(self) -> tuple[str, ...]:
        # Order MUST match multi_agent_pg/agents/base.py:_CUSTOM_TOOL_NAMES.
        # That's the order the SDK registers tools into the MCP server, which
        # in turn determines `allowed_tools` prefix list ordering, which the
        # SDK serialises into events.jsonl. Any reordering breaks byte-equal.
        return (
            "syntax_check", "size_project", "param_count",
            "read_snapshot", "rebase_to", "diff_snapshots",
            "submit_trial", "read_pr_library", "read_pr_source",
        )

    def bind_tools(self) -> list[Any]:
        """Return the @tool-decorated callables to register in the SDK MCP server.

        Order MUST match `custom_tool_names` and base.py's expectation; the SDK
        registers tools in this list order, which determines `mcp__apg__<name>`
        prefix ordering in `allowed_tools` — affects events.jsonl byte-equality.
        """
        from multi_agent_pg.tools import (
            syntax_check, size_project, param_count,
            read_snapshot, rebase_to, diff_snapshots,
            submit_trial, read_pr_library, read_pr_source,
        )
        return [
            syntax_check, size_project, param_count,
            read_snapshot, rebase_to, diff_snapshots,
            submit_trial, read_pr_library, read_pr_source,
        ]

    # ── Prompts (Day 4) ──────────────────────────────────────────────────────

    def specialist_preamble(self, domain: str) -> str:
        """Return the per-domain preamble injected into the system prompt.

        Delegates to the existing prompts.DOMAIN_PREAMBLES dict (PG ships
        10 preambles inline in agents/prompts.py). Day 4 keeps the content
        in PG; nc/cifar forks supply their own DOMAIN_PREAMBLES dict.
        """
        from multi_agent_pg.agents.prompts import DOMAIN_PREAMBLES
        try:
            return DOMAIN_PREAMBLES[domain]
        except KeyError as e:
            raise ValueError(f"unknown domain {domain!r}") from e

    def hard_limits_section(self) -> str:
        """Return the "Hard limits" markdown section.

        Day 4: this is currently embedded inline in prompts.py:GLOBAL_RULES.
        We keep it there; this method exists so Day 5 split has a place to
        delegate to. Day 5 will move the literal here.
        """
        # Verbatim copy of the section text as it appears in prompts.py:141-149
        return (
            "## Hard limits (enforced by the harness)\n"
            "\n"
            "- Submission size ≤ **16,000,000 bytes** "
            "(code packed via lzma+brotli + quantized model file).\n"
            "- Train wall-clock ≤ **600 s** on 8 GPUs SXM "
            "(in-file `MAX_WALLCLOCK_SECONDS=600`).\n"
            "- Eval wall-clock ≤ **600 s** on 8 GPUs SXM.\n"
            "- Single-file: only `train_gpt.py` is scored. Sidecar `.py` files "
            "the agent might write next to it are silently ignored by "
            "`pack_submission.py`."
        )

    def build_system_prompt(self, domain: str) -> str:
        """Assemble the full system prompt for `domain` — delegates to PG's
        prompts.build_system_prompt which combines knowledge/INIT.md +
        SOTA_STACK.md + pr_library + GLOBAL_RULES + per-domain preamble.
        """
        from multi_agent_pg.agents.prompts import build_system_prompt
        return build_system_prompt(domain)

    def keep_discard_semantics(self) -> str:
        """Return the "Keep / discard semantics" markdown section.

        Day 4: as with hard_limits_section, content stays inline in
        prompts.py:GLOBAL_RULES until Day 5 actually moves it. Method exists
        so Day 5 has a delegation target.
        """
        # Verbatim copy of the section text as it appears in prompts.py:375-386
        return (
            "## Keep / discard semantics\n"
            "\n"
            "After `submit_trial` returns the harness-computed `status` is one of:\n"
            "\n"
            "- **`keep`** — VALID + scored, agent decides to keep based on Δ vs best.\n"
            "- **`discard`** — VALID + scored, agent rejects (worse / within noise).\n"
            "- **`crash`** — train or eval child crashed before producing val_bpb.\n"
            "- **`size_blocked`** — preflight measured packed bytes > 16 MB; "
            "  no GPU was used.\n"
            "- **`preflight_crash`** — syntax_check failed or scheduler submit failed.\n"
            "- **`eval_budget_overrun`** / **`train_budget_overrun`** — DQ on time."
        )

    # ── Bootstrap (Gap 5) ────────────────────────────────────────────────────

    @property
    def baseline_score_default(self) -> float:
        # PR #1758 seed bpb claim — preserved verbatim across the Gap-5 move.
        return 1.0284

    @property
    def baseline_score_flag(self) -> str:
        return "--baseline-bpb"

    @property
    def bootstrap_hypothesis(self) -> str:
        return "PR #1758 seed (CaseOps + PreQuant TTT LR=1e-3 Unfrozen, 3-seed claim)"

    @property
    def baseline_note(self) -> str:
        # Original literal preserved verbatim for byte-equal vs frozen multi_agent.
        return "seed / PR #1758 reference"

    # PG inherits the default `scheduler_config` from TaskAdapter
    # (8 GPUs, 7200 s timeout). The actual per-trial budget is
    # enforced by run_trial.sh and the official evaluator, not here.


__all__ = ["PGTaskAdapter"]
