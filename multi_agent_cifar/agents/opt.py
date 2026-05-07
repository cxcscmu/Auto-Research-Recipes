"""Optimizer specialist — Muon variants, LR schedules, momentum, gradient flow."""

from __future__ import annotations

from .base import DoerBase


class OptDoer(DoerBase):
    """Optimizer specialist — owns optimizer choice, LR/momentum/decay schedules."""

    specialist = "opt"
