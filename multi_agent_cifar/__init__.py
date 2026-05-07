"""multi_agent_cifar — CIFAR-10 Airbench96 task package.

Importing this package:
  1. Sets per-task state-root default so the CIFAR blackboard does not
     interleave with PG / NC. Uses `os.environ.setdefault` so an
     explicit operator override via shell env still wins.
  2. Registers `CIFARTaskAdapter` with `agent_core`, making
     `current_adapter()` resolve to a CIFAR adapter for downstream
     code.

Order matters: the env setdefault must happen BEFORE any harness
module loads (config, blackboard). The supervisor / runner entry
points defer their harness imports to `main()` which runs AFTER this
body, so this is safe.
"""

from __future__ import annotations

import os
from pathlib import Path

# Per-task state-root default. CIFAR's blackboard / workdirs all live
# under their own root so PG / NC TSVs do not interleave.
os.environ.setdefault(
    "MAGENT_LOCAL_ROOT",
    str(Path.cwd() / "magent_state_cifar"),
)

# ── Adapter registration ─────────────────────────────────────────────────────
from agent_core import register_task_adapter
from multi_agent_cifar.task_config import CIFARTaskAdapter

register_task_adapter(CIFARTaskAdapter())
