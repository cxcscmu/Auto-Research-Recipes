"""Specialist agents.

Each specialist is a thin subclass of `DoerBase` that supplies a
domain key + a domain-specific prompt preamble. The heavy lifting
(SDK wiring, blackboard context rendering, tool binding) lives in
`base.py` — specialists should stay tiny and readable.
"""

from __future__ import annotations
