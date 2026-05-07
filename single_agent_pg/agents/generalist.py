"""Generalist specialist — the only doer in the single-agent variant.

The subclass is deliberately tiny: all logic lives in DoerBase, and
`specialist = "generalist"` is the only thing that differs from a
generic doer. Mirrors the multi_agent_pg/agents/<role>.py shape (e.g.
arch.py / opt.py / ttt.py) so the supervisor + blackboard + audit log
treat this agent identically to a multi_agent_pg specialist — only the
domain string and the system prompt differ.
"""

from __future__ import annotations

from .base import DoerBase


class GeneralistDoer(DoerBase):
    """Generalist single-agent — owns the full Parameter Golf recipe surface."""

    specialist = "generalist"
