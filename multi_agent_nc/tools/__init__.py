"""SDK @tool wrappers exposed to each NC specialist's Claude session.

Composition:
- 7 generic tools (syntax_check, param_count, read_snapshot, rebase_to,
  diff_snapshots, read_pr_library, read_pr_source) come from
  `agent_core.tools`. Descriptions swapped via `with_overrides`
  so the agent sees NC-flavored text instead of PG-flavored
  ("train_gpt.py / val_bpb / 16 MB").
- 1 generic tool (submit_trial) — same core path, NC-flavored description
  + schema (expected_delta = +0.005 higher-better; not -0.002).
- 1 task-specific tool (profile_pipeline) — NC-only; replaces PG's
  `size_project`. No artifact size cap; reports recipe knobs.

Order of bind_tools() must match `task_config.py:NCTaskAdapter
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

# NC-specific tool.
from .profile_pipeline import profile_pipeline, _profile_pipeline_impl


# ── Description / schema swaps (Gap 7) ───────────────────────────────────────

syntax_check = with_description(
    _core_syntax_check,
    "py_compile-check experiment.py + every .py under workdir/vendor/. "
    "Returns {ok, error, file?, line?, files_checked} — `file` + "
    "`files_checked` populate when the walk finds the FIRST SyntaxError. "
    "Cheap (~tens of ms for ~37 files); run before submit_trial to catch "
    "any vendor edit that broke imports without burning GPU time.",
)

param_count = with_description(
    _core_param_count,
    "Static AST sum of nn.Linear / nn.Embedding / nn.Conv2d literal sizes "
    "across experiment.py + every .py under workdir/vendor/. Returns "
    "{total_params, by_kind, by_file, files_walked, line_hits}. Most of "
    "the model lives in vendor/nanochat/nanochat/gpt.py — that file "
    "should dominate by_file. The estimate skips dynamic-dim Linears "
    "(common in nanochat), so use it as a relative diff signal vs prior "
    "snapshots, not as an absolute parameter count.",
)

read_snapshot = with_description(
    _core_read_snapshot,
    "Read a file from a prior `keep` snapshot. Args: {exp_id, path?}. "
    "Default `path` = experiment.py (the recipe coordinator). Pass an "
    "explicit path to read any file inside the snapshot's editable "
    "surface, e.g. 'vendor/nanochat/nanochat/gpt.py' or "
    "'vendor/nanochat/scripts/base_train.py'. Path must be relative to "
    "the snapshot root and stay inside vendor/ + experiment.py.",
)

rebase_to = with_description(
    _core_rebase_to,
    "Replace your workdir's editable surface (experiment.py + the entire "
    "vendor/ tree) with the contents of a prior `keep` snapshot. Args: "
    "{exp_id, workdir}. Use when the current best.json head has saturated "
    "and an older keep had a different angle worth resuming. After rebase, "
    "your subsequent edits stack on top of THAT snapshot's vendor.",
)

diff_snapshots = with_description(
    _core_diff_snapshots,
    "Unified diff across experiment.py + the entire vendor/ tree between "
    "two snapshots. Args: {exp_a, exp_b}. Output is the union of changed "
    "files (capped 300 lines / 8 KB total); `files_changed` tells you "
    "how many vendor + experiment.py files actually differ. Read this "
    "before building a focused mutation to avoid reading both whole "
    "trees.",
)

submit_trial = with_overrides(
    _core_submit_trial,
    description=(
        "Submit this specialist's current experiment.py to a real 8 GPUs "
        "evaluation via the scheduler. Runs a local syntax + profile_pipeline preflight "
        "first — failures are recorded WITHOUT burning GPU time. On success, "
        "blocks until the trial finishes (~30-90 min + queue), then writes a "
        "row to the blackboard TSV and returns {exp_id, status, core_metric, "
        "val_bpb, delta_vs_best, train_s, total_s, snapshot_path, job_name, "
        "notes}. Status is one of: keep | discard | crash | preflight_crash | "
        "train_budget_overrun."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "specialist":     {"type": "string",
                               "description": "Your domain key, e.g. 'arch'."},
            "hypothesis":     {"type": "string",
                               "description": "One-sentence description of what you changed."},
            "expected_delta": {"type": "string",
                               "description": "Signed core_metric delta vs current best, "
                                              "e.g. '+0.005' (higher is better for NC)."},
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
    "Browse external nanochat speedrun records / d12 ablation summaries "
    "(when curated). Args: {pr_number}. Returns the record's summary text. "
    "The library may be empty until curated; "
    "knowledge/pr_library/L2/pr_NNN_*.md is the on-disk layout.",
)

read_pr_source = with_overrides(
    _core_read_pr_source,
    description=(
        "Read a specific source file from an external nanochat speedrun "
        "record. Args: {pr_number, path}. Get the file list from "
        "read_pr_library's `available_files` first, then call this with "
        "the path you want to examine in detail."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "pr_number": {"type": "integer",
                          "description": "Synthetic record id, e.g. 1."},
            "path": {"type": "string",
                     "description": (
                         "Repo-relative path as listed in the record's available_files, "
                         "e.g. 'records/track_d12_core/.../experiment.py'."
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
    "with_overrides",
    "syntax_check",
    "profile_pipeline",
    "param_count",
    "read_snapshot",
    "rebase_to",
    "diff_snapshots",
    "submit_trial",
    "read_pr_library",
    "read_pr_source",
    "_profile_pipeline_impl",
]
