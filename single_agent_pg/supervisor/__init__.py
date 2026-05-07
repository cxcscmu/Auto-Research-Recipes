"""single_agent_pg supervisor — re-export from core.

Identical shim to `multi_agent_pg/supervisor/__init__.py`. Implementation
lives in `agent_core.supervisor`; the specialist class registry
comes from `current_adapter().specialist_classes()` (SinglePGTaskAdapter
returns `{'generalist': GeneralistDoer}`).

Entry point `python -m single_agent_pg.supervisor` works via the
adjacent `__main__.py` shim.
"""
from __future__ import annotations
