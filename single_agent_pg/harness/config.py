"""single_agent_pg harness configuration shim.

Re-exports the task-agnostic core config so `from .harness import config`
works the same as it does in multi_agent_pg (used by dashboard.py and
some tools).

PG-specific path defaults (`VENV_PATH`, `DATA_SRC`) are inherited from
`multi_agent_pg.harness.config` because the underlying task IS Parameter
Golf, same dataset and same data layout.

`DOER_DOMAINS` / `ANALYST_DOMAINS` / `ALL_DOMAINS` are defined locally
(rather than re-exported from multi_agent_pg) so this package's own
dashboard reads the single-agent specialist tuple, not the ten-role
PG tuple.
"""

from __future__ import annotations

from agent_core.harness.config import *           # noqa: F401, F403

from agent_core.harness.config import (           # noqa: F401
    _env_int,
    _load_swarm_config,
    _CONTAINER_VIRT_DISABLE,
    _bwrap_pivot_proc_works,
    _detect_container_virt,
)

# Inherit PG's path defaults (data, venv, tokenizer).
from multi_agent_pg.harness.config import (             # noqa: F401
    VENV_PATH,
    DATA_SRC,
)

# ── Specialist registry (single-agent variant) ───────────────────────────────
DOER_DOMAINS = ("generalist",)
ANALYST_DOMAINS: tuple[str, ...] = ()
ALL_DOMAINS = DOER_DOMAINS + ANALYST_DOMAINS
