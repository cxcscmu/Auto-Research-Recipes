"""Curriculum specialist — data ordering, packing, mixing across training."""

from __future__ import annotations

from .base import DoerBase


class CurrDoer(DoerBase):
    """Curriculum specialist — owns data ordering, packing, and mixing strategy."""

    specialist = "curr"
