"""multi_agent_generic_pg harness configuration shim.

Re-exports the task-agnostic core config so `from .harness import config`
works the same as in single_agent_pg / multi_agent_pg (used by
dashboard.py and some tools).

PG-specific path defaults (`VENV_PATH`, `DATA_SRC`) are inherited from
`multi_agent_pg.harness.config` because the underlying task IS Parameter
Golf, same dataset and same data layout.

`DOER_DOMAINS` / `ANALYST_DOMAINS` / `ALL_DOMAINS` are defined locally
so this package's own dashboard reads the ten generic specialists,
not the ten role-decomposed PG specialists.
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

# ── Specialist registry (generic-multi-agent variant) ────────────────────────
#
# Ten generic specialists named gena..genj. Each carries the same
# generalist preamble; the trailing letter is just a coordinate label
# for workdir and job-name namespacing. 4-letter names fit the
# `[a-z]{1,4}` constraint expected by `agent_core.harness.config:make_job_name`,
# which truncates the domain segment to four characters.
DOER_DOMAINS: tuple[str, ...] = tuple(f"gen{c}" for c in "abcdefghij")
ANALYST_DOMAINS: tuple[str, ...] = ()
ALL_DOMAINS = DOER_DOMAINS + ANALYST_DOMAINS
