"""PG-side pr_library tool — re-export shim from core.

Implementation lives in `agent_core.tools.pr_library`. The
PR-library directory is resolved at call time via
`current_adapter().knowledge_dir / "pr_library"` (PG adapter →
`multi_agent_pg/knowledge/pr_library/`).
"""

from __future__ import annotations

from agent_core.tools.pr_library import (            # noqa: F401
    read_pr_library, _read_pr_library_impl,
)
