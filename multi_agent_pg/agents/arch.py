"""Architecture specialist — first concrete doer.

The subclass is deliberately tiny: all logic lives in DoerBase, and
`specialist = "arch"` is the only thing that differs from a generic
doer. Future per-domain hooks (extra preflight, domain-specific
result-parsing) can be added as method overrides here without
touching the base.
"""

from __future__ import annotations

from .base import DoerBase


class ArchDoer(DoerBase):
    """Architecture specialist — owns blocks/attention/recurrence/MLP/norm/embedding."""

    specialist = "arch"
