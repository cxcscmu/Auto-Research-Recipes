"""CIFAR-10 Airbench96 harness configuration shim.

Re-exports the task-agnostic core config and adds CIFAR-specific defaults:

  * `PKG_ROOT`  — multi_agent_cifar/ directory.
  * `VENV_PATH` — defaults to `./data/cifar/venv`. Set `MAGENT_CIFAR_VENV=skip`
                  to use the calling environment.
  * `DATA_SRC`  — defaults to `./data/cifar/data` (the cifar-10-batches-py
                  layout that airbench96 expects).
  * `DOER_DOMAINS` / `ANALYST_DOMAINS` / `ALL_DOMAINS` — CIFAR specialist tuples.
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
    "MAGENT_CIFAR_VENV",
    str(_REPO_ROOT / "data" / "cifar" / "venv"),
))

DATA_SRC = Path(os.environ.get(
    "MAGENT_CIFAR_DATA_DIR",
    str(_REPO_ROOT / "data" / "cifar" / "data"),
))


# ── Specialist registry (CIFAR taxonomy) ─────────────────────────────────────
DOER_DOMAINS = (
    "arch",          # CifarNet structure.
    "opt",           # Muon + SGD config, LR schedule.
    "aug",           # Augmentation pipeline.
    "loss",          # Cross-entropy variants.
    "reg",           # Weight decay, dropout, stochastic depth.
)

ANALYST_DOMAINS: tuple[str, ...] = ()
# `meta` analyst intentionally absent until a real blackboard-write
# tool exists. See task_config.py:CIFARTaskAdapter.

ALL_DOMAINS = DOER_DOMAINS + ANALYST_DOMAINS
