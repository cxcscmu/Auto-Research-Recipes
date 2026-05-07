"""SDK @tool wrappers exposed to each specialist's Claude session — PG composition.

Generic tools (syntax_check / param_count / read_snapshot / rebase_to /
diff_snapshots / read_pr_library / read_pr_source) live in
`agent_core.tools` and are re-exported via this package's
submodules so existing PG imports
(`from multi_agent_pg.tools.workdir import ...`) keep resolving.

PG-specific tools (`size_project`, `submit_trial`) are owned here.

The SDK @tool decorator + create_sdk_mcp_server shim live in
`agent_core.tools.__init__`; we re-export them so PG modules can
keep using `from . import tool` without changes.
"""

from __future__ import annotations

# SDK shim + decorator — single source in core.
from agent_core.tools import tool, create_sdk_mcp_server  # noqa: F401

# Generic tools (resolved via core, surfaced through PG submodules).
from .code_inspect import size_project, param_count, syntax_check
from .workdir import diff_snapshots, read_snapshot, rebase_to
from .pr_library import read_pr_library
from .pr_source import read_pr_source

# PG-specific tool — submit_trial stays in PG until Day 6 (it depends on
# blackboard which is also still PG; Day 6 will move both to core together).
from .submit import submit_trial

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
