"""multi_agent_generic_pg supervisor CLI shim.

Forwards to `agent_core.supervisor.__main__:main` after ensuring
the GenericMultiPGTaskAdapter is registered.

Usage: `python -m multi_agent_generic_pg.supervisor [args...]`. All
agent_core CLI flags work unchanged: --state-root /
--job-name-prefix / --max-turns /
--no-lineage / --baseline-bpb / --deadline-hours / etc.
"""

from __future__ import annotations

import sys

import multi_agent_generic_pg  # noqa: F401  (triggers register_task_adapter)
from agent_core.supervisor.__main__ import main


if __name__ == "__main__":
    sys.exit(main())
