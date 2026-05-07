"""PG-side tracker — re-export from core.

The full implementation lives in `agent_core.harness.tracker` and
reads PG-specific schema (TSV_FIELDS, score_field, baseline_filename)
via `current_adapter()` at call time. This file exists only to keep the
import path `multi_agent_cifar.harness.tracker` resolves via this shim.
"""

from __future__ import annotations

from agent_core.harness.tracker import (             # noqa: F401
    read_results,
    append_result,
    next_exp_id,
    find_best,
    parse_validate_result,
    extract_crash_excerpt,
    extract_phase_summary,
    extract_pack_breakdown,
    empty_validate_row,
    _STATUS_INFORMATIVE,
    _STATUS_MAP,
    _fmt_float,
)


# Module-level TSV_FIELDS: resolve lazily via core's adapter-aware lookup.
def __getattr__(name: str):
    if name == "TSV_FIELDS":
        from agent_core.harness.tracker import _tsv_fields
        return _tsv_fields()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
