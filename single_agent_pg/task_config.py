"""SinglePGTaskAdapter — Parameter Golf single-generalist task adapter.

Inherits from `multi_agent_pg.task_config.PGTaskAdapter` and overrides
the small set of properties that distinguish a single-agent run from
a swarm:

  - `pkg_root`               — points at single_agent_pg/, NOT multi_agent_pg/
  - `doer_domains`           — ('generalist',) instead of the 10 PG specialists
  - `analyst_domains`        — () (no separate analyst role)
  - `specialist_classes()`   — maps 'generalist' → GeneralistDoer
  - `build_system_prompt()`  — delegates to single_agent_pg.agents.prompts
  - `job_name_prefix`         — 'apg1' (was 'apg' for multi_agent_pg) so
                                concurrent runs do not cross-kill jobs
  - `bootstrap_hypothesis`   — single-agent-aware baseline note

Everything else (TSV schema, score_field='val_bpb', score_lower_is_better,
parse_validate_record, empty_validate_row, custom_tool_names, bind_tools,
stage_files, seed_file, run_script, trial_output_dirs, size_check,
baseline_filename, baseline_score_default, baseline_score_flag,
requires_calibrated_baseline) is inherited unchanged from
PGTaskAdapter, since the underlying task IS Parameter Golf, just run
with a different agent organization.

Stage files (pack_submission.py, run_classify.py, run_trainer.py,
run_trial.sh) live as symlinks under single_agent_pg/ pointing at
multi_agent_pg/ so the inherited `stage_files` paths resolve correctly
under the overridden pkg_root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from multi_agent_pg.task_config import PGTaskAdapter


class SinglePGTaskAdapter(PGTaskAdapter):
    """Single-generalist Parameter Golf adapter.

    Subclasses PGTaskAdapter rather than re-implementing the full
    TaskAdapter ABC, since the task semantics (PG = lower val_bpb,
    16 MB cap, 600 s budgets) are identical and only the agent
    organization differs.
    """

    # ── Package root ─────────────────────────────────────────────────────────

    @property
    def pkg_root(self) -> Path:
        # single_agent_pg/ — where train_gpt.py / run_trial.sh / knowledge/ live
        # (as symlinks to multi_agent_pg/) and where stage_files paths resolve.
        import single_agent_pg
        return Path(single_agent_pg.__file__).resolve().parent

    # ── Specialists ──────────────────────────────────────────────────────────

    @property
    def doer_domains(self) -> tuple[str, ...]:
        return ("generalist",)

    @property
    def analyst_domains(self) -> tuple[str, ...]:
        return ()

    def specialist_classes(self) -> dict[str, type]:
        from single_agent_pg.agents.generalist import GeneralistDoer
        return {"generalist": GeneralistDoer}

    # ── System prompt ────────────────────────────────────────────────────────

    def build_system_prompt(self, domain: str) -> str:
        """Delegate to single_agent_pg.agents.prompts.

        The assembled system prompt is:
            knowledge md files (INIT/SOTA_STACK/LESSONS, shared with PG)
          + GLOBAL_RULES (PG's, with one line patched: swarm framing →
            autonomous-loop framing)
          + _GENERALIST_PREAMBLE (new; ports the legacy single_agent voice
            + scope + counterfactual discipline).
        """
        from single_agent_pg.agents.prompts import build_system_prompt
        return build_system_prompt(domain)

    # ── Job naming ────────────────────────────────────────────────────────

    @property
    def job_name_prefix(self) -> str:
        # 'apg1' (1 = single, distinct from multi_agent_pg's 'apg').
        # Required by the active_job_name_prefix() resolution in
        # agent_core/harness/config.py — keeps concurrent
        # single-agent + multi-agent PG runs from cross-killing each
        # other on supervisor SIGINT. Operator may further override
        # via --job-name-prefix at runtime.
        return "apg1"

    # ── Baseline bookkeeping ─────────────────────────────────────────────────

    @property
    def bootstrap_hypothesis(self) -> str:
        # Distinct from multi_agent_pg's "PR #1758 seed ..." note so a
        # glance at exp_000 in results.tsv tells operator + reviewer
        # which run mode produced the baseline row.
        return "single-agent generalist baseline (PG 1.0810 SOTA stack)"

    @property
    def baseline_note(self) -> str:
        return "single-agent generalist / PG 1.0810 reference"
