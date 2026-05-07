"""GenericMultiPGTaskAdapter — Parameter Golf, 10× generic-specialist variant.

Strict 10× replica of `SinglePGTaskAdapter`, NOT a subclass of
multi_agent_pg's role-decomposed `PGTaskAdapter` semantics. The shape
overrides differ from single_agent_pg only in cardinality:

  - `pkg_root`               — points at multi_agent_generic_pg/
  - `doer_domains`           — ('gena', ..., 'genj') instead of ('generalist',)
  - `analyst_domains`        — () (no separate analyst role)
  - `specialist_classes()`   — maps each gen<x> → Gen<X>Doer
  - `build_system_prompt()`  — delegates to multi_agent_generic_pg.agents.prompts;
                                EVERY domain returns the SAME generic preamble
  - `job_name_prefix`         — 'apgg' (g for generic), distinct from 'apg'
                                (multi_agent_pg) / 'apg1' (single_agent_pg)
                                / 'apga'-'apgb' (Run A / B). Required so
                                concurrent runs do not cross-kill on shutdown.
  - `bootstrap_hypothesis`   — generic-multi-agent-aware baseline note

Everything else (TSV schema, score_field='val_bpb', score_lower_is_better,
parse_validate_record, empty_validate_row, custom_tool_names, bind_tools,
stage_files, seed_file, run_script, trial_output_dirs, size_check,
baseline_filename, baseline_score_default, baseline_score_flag,
requires_calibrated_baseline) is inherited unchanged from
PGTaskAdapter, since the underlying task IS Parameter Golf, just run
with a different agent organization (10 generic agents in parallel
instead of 1 generalist or 10 role-decomposed specialists).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from multi_agent_pg.task_config import PGTaskAdapter


# Specialist coordinates: 10 generic agents, named gena..genj. Each
# shares the same generic preamble (no role decomposition); the
# trailing letter serves only as a workdir and job-name namespace
# coordinate.
#
# Why 4-letter `gen<x>` rather than `gene_<i>` or `gen<i>`:
# `agent_core.harness.config.make_job_name` truncates the domain
# segment via `domain[:4]`, and the segment is constrained to
# lowercase letters only (no digits, no underscores) by the job-name
# convention. With `gene_0..gene_9` all ten truncate to "gene" and
# the rendered job name distinguishes only by `trial_id`. Naming
# each specialist with a unique 4-letter `gen<x>` (x in a..j) gives
# every specialist a distinct fully-rendered job name like
# <prefix>-gena-0001 / <prefix>-genb-0002.
GENERIC_DOMAINS: tuple[str, ...] = tuple(f"gen{c}" for c in "abcdefghij")


class GenericMultiPGTaskAdapter(PGTaskAdapter):
    """Generic 10×-multi-agent Parameter Golf adapter.

    Subclasses PGTaskAdapter rather than re-implementing the full
    TaskAdapter ABC, since the task semantics (PG = lower val_bpb,
    16 MB cap, 600 s budgets) are identical and only the agent
    organization differs.
    """

    # ── Package root ─────────────────────────────────────────────────────────

    @property
    def pkg_root(self) -> Path:
        # multi_agent_generic_pg/ — where train_gpt.py / run_trial.sh /
        # knowledge/ live (as symlinks to multi_agent_pg/) and where
        # stage_files paths resolve.
        import multi_agent_generic_pg
        return Path(multi_agent_generic_pg.__file__).resolve().parent

    # ── Specialists ──────────────────────────────────────────────────────────

    @property
    def doer_domains(self) -> tuple[str, ...]:
        return GENERIC_DOMAINS

    @property
    def analyst_domains(self) -> tuple[str, ...]:
        return ()

    def specialist_classes(self) -> dict[str, type]:
        from multi_agent_generic_pg.agents.generic import GENERIC_DOER_CLASSES
        return dict(GENERIC_DOER_CLASSES)

    # ── System prompt ────────────────────────────────────────────────────────

    def build_system_prompt(self, domain: str) -> str:
        """Delegate to multi_agent_generic_pg.agents.prompts.

        ALL ten domains (gena..genj) return the same assembled prompt:
        knowledge md files (shared with PG) + GLOBAL_RULES (the same
        single-agent-patched version used by single_agent_pg) + the
        same _GENERALIST_PREAMBLE used by single_agent_pg.

        The generic prompt does NOT mention the specialist's own
        coordinate or that other specialists exist — each agent thinks
        of itself as an autonomous researcher. Coordinate-level
        distinctions are carried only by the user message (`Your
        workdir is workdir_genc`) and by the specialist label appearing
        in shared lineage.
        """
        from multi_agent_generic_pg.agents.prompts import build_system_prompt
        return build_system_prompt(domain)

    # ── Job naming ────────────────────────────────────────────────────────

    @property
    def job_name_prefix(self) -> str:
        # 'apgg' (g for generic). Distinct from:
        #   apg   - multi_agent_pg
        #   apg1  - single_agent_pg
        #   apga  - Run A (lineage on)
        #   apgb  - Run B (no-lineage)
        # Operator may further override via --job-name-prefix at runtime
        # to avoid collision with any other concurrent run.
        return "apgg"

    # ── Baseline bookkeeping ─────────────────────────────────────────────────

    @property
    def bootstrap_hypothesis(self) -> str:
        # Distinct from PG / single-agent / Run A / Run B baseline notes
        # so a glance at exp_000 in results.tsv tells operator + reviewer
        # which run mode produced the baseline row.
        return "generic-multi-agent baseline (PG 1.0810 SOTA stack, 10× generic)"

    @property
    def baseline_note(self) -> str:
        return "generic-multi-agent / PG 1.0810 reference"
