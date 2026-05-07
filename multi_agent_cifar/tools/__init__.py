"""SDK @tool wrappers exposed to each CIFAR specialist's Claude session.

Composition:
- 7 generic tools (syntax_check, param_count, read_snapshot, rebase_to,
  diff_snapshots, read_pr_library, read_pr_source) come from
  `agent_core.tools`.
- 1 generic tool (submit_trial) — same core path; description swapped via
  `with_description` so the agent reads CIFAR-flavored text instead of
  PG-flavored ("train_gpt.py / val_bpb / 16 MB").
- 1 task-specific tool (recipe_check) — CIFAR-only; replaces PG's
  `size_project`. No artifact size cap; reports param/recipe diagnostics.

The order of bind_tools() must match `task_config.py:CIFARTaskAdapter
.custom_tool_names` for SDK MCP registration determinism.
"""

from __future__ import annotations

# SDK shim + decorator + override helpers (single source in core).
from agent_core.tools import (                                  # noqa: F401
    tool,
    create_sdk_mcp_server,
    with_description,
    with_overrides,
)

# Generic tools — re-imported from core.
from agent_core.tools.code_inspect import syntax_check as _core_syntax_check
from agent_core.tools.code_inspect import param_count  as _core_param_count
from agent_core.tools.workdir       import read_snapshot   as _core_read_snapshot
from agent_core.tools.workdir       import rebase_to       as _core_rebase_to
from agent_core.tools.workdir       import diff_snapshots  as _core_diff_snapshots
from agent_core.tools.pr_library    import read_pr_library as _core_read_pr_library
from agent_core.tools.pr_source     import read_pr_source  as _core_read_pr_source
from agent_core.tools.submit        import submit_trial    as _core_submit_trial

# CIFAR-specific tool.
from .recipe_check import recipe_check, _recipe_check_impl


# ── Description swaps (Gap 7) ────────────────────────────────────────────────
#
# Each tool description below is what the *model* sees in its tool palette.
# Core's literals are PG-flavored ("train_gpt.py", "val_bpb", "16 MB") and
# would mislead a CIFAR specialist. We swap them here without mutating the
# core objects (PG byte-equal preserved — PG's bind_tools returns the
# unswapped originals).

syntax_check = with_description(
    _core_syntax_check,
    "py_compile-check airbench96.py in your workdir. Returns "
    "{ok, error}. Cheap (~5 ms head-side); run before submit_trial to "
    "catch syntax errors without burning GPU time.",
)

param_count = with_description(
    _core_param_count,
    "Static AST estimate of trainable parameter count in airbench96.py. "
    "Sums nn.Linear / nn.Embedding / nn.Conv2d literal sizes. Fast "
    "(~5 ms) but only catches gross structural changes — for exact counts, "
    "run a trial. Returns {total_params, by_kind, line_hits}.",
)

read_snapshot = with_description(
    _core_read_snapshot,
    "Read airbench96.py from a prior `keep` snapshot. Args: "
    "{exp_id}. Returns the recipe as text — useful for examining what "
    "made an earlier trial work without copying it into your workdir.",
)

rebase_to = with_description(
    _core_rebase_to,
    "Replace your workdir's airbench96.py with the recipe from a "
    "prior `keep` snapshot. Args: {exp_id, workdir}. Use when you want "
    "to start a new mutation from an earlier branch (the current best.json "
    "head has saturated, an older keep had a different angle worth pushing).",
)

diff_snapshots = with_description(
    _core_diff_snapshots,
    "Diff the airbench96.py recipes between two snapshots. Args: "
    "{exp_a, exp_b}. Returns a unified diff (capped 300 lines / 8 KB) so "
    "you can build a focused mutation rather than reading both files in full.",
)

submit_trial = with_overrides(
    _core_submit_trial,
    description=(
        "Submit this specialist's current airbench96.py to a real GPU "
        "evaluation on a single GPU (the other 7 GPUs "
        "idle by design). Runs a local syntax + recipe_check preflight, then "
        "the harness invokes airbench96.py N=10 times (one per seed) inside a "
        "single shell invocation, aggregates {mean_acc, mean_train_s, std} across the "
        "10 seeds, applies the acc≥0.96 threshold gate, and returns one row "
        "to the blackboard TSV: {exp_id, status, train_s, accuracy, "
        "delta_vs_best, snapshot_path, job_name, notes}. "
        "Status is one of: keep | discard | disqualified | crash | "
        "preflight_crash | train_budget_overrun. "
        "v2 score = mean train_s across N=10 seeds (LOWER IS BETTER). The "
        "acc gate is on the n=10 mean, not single seed. Borderline trials "
        "(mean_acc 95.85–96.15%) record as discard with raw_status=BORDERLINE."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "specialist":     {"type": "string",
                               "description": "Your domain key, e.g. 'arch'."},
            "hypothesis":     {"type": "string",
                               "description": "One-sentence description of what you changed."},
            "expected_delta": {"type": "string",
                               "description": "Signed train_s delta vs current best in seconds, "
                                              "e.g. '-0.5' (LOWER train_s wins). The trial "
                                              "must hit mean_acc ≥ 0.96 across 10 seeds OR it's "
                                              "disqualified regardless of train_s."},
            "parent_exp":     {"type": "string",
                               "description": "exp_id you rooted from (usually best.json)."},
            "notes":          {"type": "string",
                               "description": "Optional free-form rationale.",
                               "default": ""},
        },
        "required": ["specialist", "hypothesis", "expected_delta", "parent_exp"],
    },
)

read_pr_library = with_description(
    _core_read_pr_library,
    "Browse external CIFAR speedrun records (airbench95-97 variants and "
    "related vision-CNN speedrun history). Args: {pr_number}. Returns the "
    "record's summary text. The library may be empty until curated; "
    "knowledge/pr_library/L2/pr_NNN_*.md is the on-disk layout.",
)

read_pr_source = with_overrides(
    _core_read_pr_source,
    description=(
        "Read a specific source file from an external CIFAR speedrun record. "
        "Args: {pr_number, path}. Get the file list from read_pr_library's "
        "`available_files` first, then call this with the path you want to "
        "examine in detail."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "pr_number": {"type": "integer",
                          "description": "Synthetic record id, e.g. 1 (airbench95)."},
            "path": {"type": "string",
                     "description": (
                         "Repo-relative path as listed in the record's available_files, "
                         "e.g. './reference_recipes/airbench94_muon.py'."
                     )},
            "offset": {"type": "integer",
                       "description": "Line offset for paginated reads.", "default": 0},
            "limit":  {"type": "integer",
                       "description": "Max lines returned.", "default": 1000},
        },
        "required": ["pr_number", "path"],
    },
)


__all__ = [
    "tool",
    "create_sdk_mcp_server",
    "with_description",
    "syntax_check",
    "recipe_check",
    "param_count",
    "read_snapshot",
    "rebase_to",
    "diff_snapshots",
    "submit_trial",
    "read_pr_library",
    "read_pr_source",
    "_recipe_check_impl",
]
