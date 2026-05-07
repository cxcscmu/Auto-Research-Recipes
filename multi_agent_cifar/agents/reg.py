"""Regularization specialist — dropout, weight decay, stochastic depth."""

from __future__ import annotations

from .base import DoerBase


class RegDoer(DoerBase):
    """Regularization specialist — owns dropout, weight decay, stochastic depth."""

    specialist = "reg"
