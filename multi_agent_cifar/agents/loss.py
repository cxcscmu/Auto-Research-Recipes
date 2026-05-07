"""Loss / auxiliary-objective specialist — z-loss, aux heads, label smoothing."""

from __future__ import annotations

from .base import DoerBase


class LossDoer(DoerBase):
    """Loss specialist — owns main loss, auxiliary heads, and loss-scaling."""

    specialist = "loss"
