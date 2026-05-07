"""multi_agent_generic_pg supervisor — re-export from core.

Identical shim to `single_agent_pg/supervisor/__init__.py`. Implementation
lives in `agent_core.supervisor`; the specialist class registry
comes from `current_adapter().specialist_classes()` (GenericMultiPGTaskAdapter
returns 10 entries: gena..genj).

Entry point `python -m multi_agent_generic_pg.supervisor` works via the
adjacent `__main__.py` shim.
"""
from __future__ import annotations
