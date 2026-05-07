"""PG-side workdir tools — re-export shim.

The actual implementations live in `agent_core.tools.workdir`. PG
code that does `from multi_agent_cifar.tools.workdir import X`
resolve via this re-export.
"""

from __future__ import annotations

from agent_core.tools.workdir import (              # noqa: F401
    read_snapshot, _read_snapshot_impl,
    rebase_to,    _rebase_to_impl,
    diff_snapshots, _diff_snapshots_impl,
    _find_snapshot_dir,
)
