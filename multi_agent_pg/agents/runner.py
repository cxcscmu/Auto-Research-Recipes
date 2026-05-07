"""SHIM — re-export from agent_core.agents.runner.

The single-specialist debugger CLI now lives in core. This shim keeps
`python -m multi_agent_pg.agents.runner --specialist <X>` working: the
`multi_agent_pg` import registers PGTaskAdapter, then we forward to
core's main(). Same flags / same summary / same exit codes.
"""

from __future__ import annotations

import sys

import multi_agent_pg                                # noqa: F401  (registers adapter)
from agent_core.agents.runner import main


if __name__ == "__main__":
    sys.exit(main())
