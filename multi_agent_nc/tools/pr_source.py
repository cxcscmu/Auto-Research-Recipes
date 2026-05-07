"""PG-side pr_source tool — re-export shim from core.

Implementation lives in `agent_core.tools.pr_source`. The
PR-library directory is resolved at call time via
`current_adapter().knowledge_dir / "pr_library"`.
"""

from __future__ import annotations

from agent_core.tools.pr_source import (             # noqa: F401
    read_pr_source, _read_pr_source_impl,
)
