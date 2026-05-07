"""multi_agent_generic_pg tools — re-export from multi_agent_pg.tools verbatim.

Tool catalogue, schemas, sandbox guards, and submit_trial implementation
are shared with multi_agent_pg unchanged. Generic-multi-agent-specific
behaviour (10 generic specialists sharing one prompt) is imposed at the
adapter layer (`multi_agent_generic_pg.task_config:GenericMultiPGTaskAdapter`),
not here.

The bound-tool order MUST match `multi_agent_pg.tools.__init__` because
the SDK registers tools in that order and the resulting MCP allowed-tools
prefix list determines events.jsonl byte-equal ordering.

Symlinks under multi_agent_generic_pg/tools/ — pack_submission.py,
run_classify.py, run_trainer.py — point at multi_agent_pg/tools/ so
the inherited `stage_files` paths resolve correctly under
GenericMultiPGTaskAdapter.pkg_root.
"""

from __future__ import annotations

# SDK shim + decorator — single source in core.
from agent_core.tools import tool, create_sdk_mcp_server  # noqa: F401

# Re-export PG's tool entry points so `bind_tools()` (inherited from
# PGTaskAdapter, which imports from `multi_agent_pg.tools`) continues to
# resolve every tool exactly as the multi-agent run does.
from multi_agent_pg.tools import (                              # noqa: F401
    size_project,
    param_count,
    syntax_check,
    read_snapshot,
    rebase_to,
    diff_snapshots,
    submit_trial,
    read_pr_library,
    read_pr_source,
)

__all__ = [
    "tool",
    "create_sdk_mcp_server",
    "size_project",
    "param_count",
    "syntax_check",
    "read_snapshot",
    "rebase_to",
    "diff_snapshots",
    "submit_trial",
    "read_pr_library",
    "read_pr_source",
]
