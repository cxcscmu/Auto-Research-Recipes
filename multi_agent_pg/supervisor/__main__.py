"""PG-side supervisor CLI shim.

Forwards to `agent_core.supervisor.__main__:main` after ensuring
the PG TaskAdapter is registered.

Existing command `python -m multi_agent_pg.supervisor [args...]` works
unchanged.
"""

from __future__ import annotations

import sys

import multi_agent_pg  # noqa: F401  (triggers register_task_adapter)
from agent_core.supervisor.__main__ import main


if __name__ == "__main__":
    sys.exit(main())
