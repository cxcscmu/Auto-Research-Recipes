"""SHIM — re-export from agent_core.agents.runner.

The single-specialist debugger CLI lives in core. This shim keeps
`python -m single_agent_pg.agents.runner --specialist generalist`
working: the `single_agent_pg` import registers SinglePGTaskAdapter,
then we forward to core's main(). Same flags / same summary / same
exit codes.
"""

from __future__ import annotations

import sys

import single_agent_pg                            # noqa: F401  (registers adapter)
from agent_core.agents.runner import main


if __name__ == "__main__":
    sys.exit(main())
