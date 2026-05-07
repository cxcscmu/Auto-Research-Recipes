"""Parameter Golf harness configuration shim.

Re-exports the task-agnostic core config and adds PG-specific defaults:

  * `PKG_ROOT`   — multi_agent_pg/ directory (where train_gpt.py lives).
  * `VENV_PATH`  — defaults to `./data/parameter_golf/venv`.
  * `DATA_SRC`   — defaults to `./data/parameter_golf/fineweb10B_sp8192/`.
  * `DOER_DOMAINS` / `ANALYST_DOMAINS` / `ALL_DOMAINS` — PG specialist tuples.

`MAGENT_TOKENIZER_SRC` is read by run_trial.sh, not here.
"""

from __future__ import annotations

import os
from pathlib import Path

# Re-export ALL public names from the core config module so this module
# remains a drop-in superset for legacy imports.
from agent_core.harness.config import *           # noqa: F401, F403

# Re-export selected private names that PG code imports.
from agent_core.harness.config import (           # noqa: F401
    _env_int,
    _load_swarm_config,
    _CONTAINER_VIRT_DISABLE,
    _bwrap_pivot_proc_works,
    _detect_container_virt,
)


# ── Task-package root ────────────────────────────────────────────────────────
# Resolves to `multi_agent_pg/` — used by submit.py (_stage_workdir),
# pr_library/pr_source, baseline_audit (BASELINE source), and base.py.
PKG_ROOT = Path(__file__).resolve().parent.parent


# ── PG-specific path defaults ────────────────────────────────────────────────
#
# Both default to a sibling `data/parameter_golf/` directory under the
# repository root. Operators are expected to either populate this
# directory before running a trial or set `MAGENT_PG_VENV` /
# `MAGENT_PG_DATA_DIR` environment overrides. See multi_agent_pg/README.md
# for data preparation notes.

_REPO_ROOT = Path(os.environ.get("MAGENT_REPO_ROOT", str(Path.cwd())))

VENV_PATH = Path(os.environ.get(
    "MAGENT_PG_VENV",
    str(_REPO_ROOT / "data" / "parameter_golf" / "venv"),
))

DATA_SRC = Path(os.environ.get(
    "MAGENT_PG_DATA_DIR",
    str(_REPO_ROOT / "data" / "parameter_golf" / "fineweb10B_sp8192"),
))


# ── Specialist registry (PG taxonomy) ────────────────────────────────────────
#
# Nine doers + one analyst. Names are 1..4 chars where possible so the
# scheduler job-name prefix stays short.
DOER_DOMAINS = (
    "arch",          # Architecture: blocks, attention, recurrence topology.
    "opt",           # Optimizer: Muon variants, LR schedules, momentum.
    "tok",           # Tokenizer: vocab size, segmentation, BPE/SP.
    "quant",         # Quantization: GPTQ variants, bit-width.
    "ttt",           # Test-Time Training: adaptation, soft prompts.
    "curr",          # Curriculum: data ordering, packing, mixing.
    "loss",          # Loss / auxiliary: z-loss, aux heads, label smoothing.
    "reg",           # Regularization: dropout, weight decay, stochastic depth.
    "eval",          # Evaluation & inference: sliding-window, decoding.
)

ANALYST_DOMAINS = (
    "meta",          # Meta-search: hyperparameter sweeps.
)

ALL_DOMAINS = DOER_DOMAINS + ANALYST_DOMAINS
