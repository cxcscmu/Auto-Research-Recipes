"""multi_agent_pg — Parameter Golf task package.

Importing this package registers `PGTaskAdapter` with `agent_core`,
making `agent_core.current_adapter()` resolve to a PG-aware adapter
for downstream code (supervisor, agents, tools, dashboard).

Per-task state-root convention. Forks for nc / cifar set their own
`MAGENT_LOCAL_ROOT` defaults here via `os.environ.setdefault` so their
blackboards stay separate from PG's. PG itself keeps the historical
default (`./magent_state`); overriding it here would re-route an
operator's existing blackboard, which is destructive. The setdefault
pattern lets operators still override via shell env.
"""

from __future__ import annotations

from agent_core import register_task_adapter
from multi_agent_pg.task_config import PGTaskAdapter

register_task_adapter(PGTaskAdapter())
