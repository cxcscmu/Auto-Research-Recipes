"""NanoChat-d12 miniseries harness configuration shim.

Re-exports the task-agnostic core config and adds NC-specific defaults:

  * `PKG_ROOT`  — multi_agent_nc/ directory.
  * `VENV_PATH` — defaults to `./data/nanochat/venv`. Set
                  `MAGENT_NC_VENV=skip` to use the calling environment.
  * `DATA_SRC`  — defaults to `./data/nanochat/` (the NanoChat base dir
                  containing tokenizer/, base_data_climbmix/, eval_bundle/).
  * `DOER_DOMAINS` / `ANALYST_DOMAINS` / `ALL_DOMAINS` — NC specialist tuples.
"""

from __future__ import annotations

import os
from pathlib import Path

from agent_core.harness.config import *           # noqa: F401, F403

from agent_core.harness.config import (           # noqa: F401
    _env_int,
    _load_swarm_config,
    _CONTAINER_VIRT_DISABLE,
    _bwrap_pivot_proc_works,
    _detect_container_virt,
)


PKG_ROOT = Path(__file__).resolve().parent.parent

_REPO_ROOT = Path(os.environ.get("MAGENT_REPO_ROOT", str(Path.cwd())))

VENV_PATH = Path(os.environ.get(
    "MAGENT_NC_VENV",
    str(_REPO_ROOT / "data" / "nanochat" / "venv"),
))

DATA_SRC = Path(os.environ.get(
    "MAGENT_NC_BASE_DIR",
    os.environ.get(
        "NANOCHAT_BASE_DIR",
        str(_REPO_ROOT / "data" / "nanochat"),
    ),
))


# ── Specialist registry (NC taxonomy) ────────────────────────────────────────
DOER_DOMAINS = (
    "arch",          # d12 transformer architecture (vendor/nanochat/gpt.py).
    "opt",           # Muon + AdamW (vendor/nanochat/optim.py).
    "data",          # Data pipeline (vendor/nanochat/dataloader.py + tokenizer.py).
    "sched",         # LR / momentum / wd schedules (vendor/scripts/base_train.py).
    "sys",           # Precision + kernels (vendor/nanochat/fp8.py + flash_attention.py).
)

# `meta` analyst absent in v1; reintroduce when a blackboard-write tool exists.
ANALYST_DOMAINS: tuple[str, ...] = ()

ALL_DOMAINS = DOER_DOMAINS + ANALYST_DOMAINS
