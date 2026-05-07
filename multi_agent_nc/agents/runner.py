"""SHIM — re-export from agent_core.agents.runner.

`python -m multi_agent_nc.agents.runner --specialist <X>` imports
multi_agent_nc (which sets env defaults + registers NCTaskAdapter), then
forwards to core's main(). Same flags as PG/CIFAR.
"""

from __future__ import annotations

import sys

import multi_agent_nc                                # noqa: F401  (registers adapter)
from agent_core.agents.runner import main


if __name__ == "__main__":
    sys.exit(main())
