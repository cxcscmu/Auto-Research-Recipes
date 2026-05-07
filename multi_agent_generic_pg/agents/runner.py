"""SHIM — re-export from agent_core.agents.runner.

The single-specialist debugger CLI lives in core. This shim keeps
`python -m multi_agent_generic_pg.agents.runner --specialist genc`
working: the `multi_agent_generic_pg` import registers
GenericMultiPGTaskAdapter, then we forward to core's main(). Same
flags / same summary / same exit codes.
"""

from __future__ import annotations

import sys

import multi_agent_generic_pg                    # noqa: F401  (registers adapter)
from agent_core.agents.runner import main


if __name__ == "__main__":
    sys.exit(main())
