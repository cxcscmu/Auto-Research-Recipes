"""SHIM — re-export from agent_core.agents.runner.

`python -m multi_agent_cifar.agents.runner --specialist <X>` imports
multi_agent_cifar (which registers CIFARTaskAdapter), then forwards to
core's main(). Same flags / same summary / same exit codes as PG.
"""

from __future__ import annotations

import sys

import multi_agent_cifar                              # noqa: F401  (registers adapter)
from agent_core.agents.runner import main


if __name__ == "__main__":
    sys.exit(main())
