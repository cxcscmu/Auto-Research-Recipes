"""PG-side supervisor — re-export from core.

Implementation lives in `agent_core.supervisor`; the specialist
class registry comes from `current_adapter().specialist_classes()` (PG
adapter returns the 9 PG doers + meta).

Entry point `python -m multi_agent_pg.supervisor` works via the
adjacent `__main__.py` shim.
"""
from __future__ import annotations
