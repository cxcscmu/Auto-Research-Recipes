"""multi_agent_nc — NanoChat-d12 miniseries task package.

Importing this package:
  1. Sets per-task state-root default so the NC blackboard does not
     interleave with PG / CIFAR. Uses `os.environ.setdefault` so an
     explicit operator override still wins.
  2. Sets `NANOCHAT_BASE_DIR` default. Upstream nanochat's
     `nanochat/common.py:get_base_dir()` reads this env to find the
     pre-baked tokenizer, training shards, and evaluation bundle.
  3. Registers `NCTaskAdapter` with `agent_core` so
     `current_adapter()` resolves to an NC-aware adapter for downstream
     code.

Order matters: env setdefault must happen BEFORE any harness module
loads (config, blackboard). The supervisor / runner entry points
defer their harness imports to `main()` which runs AFTER this body.
"""

from __future__ import annotations

import os
from pathlib import Path

# Default state-root under <repo>/magent_state_nc.
os.environ.setdefault(
    "MAGENT_LOCAL_ROOT",
    str(Path.cwd() / "magent_state_nc"),
)

# Default NanoChat data root under <repo>/data/nanochat/. Operator
# may either populate this directory (tokenizer, base_data_climbmix,
# eval_bundle) or set `NANOCHAT_BASE_DIR` to point at an alternate
# location. See multi_agent_nc/README.md for data preparation.
_repo_root = os.environ.get("MAGENT_REPO_ROOT", str(Path.cwd()))
os.environ.setdefault(
    "NANOCHAT_BASE_DIR",
    f"{_repo_root}/data/nanochat",
)

# ── Adapter registration ─────────────────────────────────────────────────────
from agent_core import register_task_adapter
from multi_agent_nc.task_config import NCTaskAdapter

register_task_adapter(NCTaskAdapter())
