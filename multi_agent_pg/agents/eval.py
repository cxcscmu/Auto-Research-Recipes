"""Evaluation & inference specialist — sliding-window, decoding, eval-time speed."""

from __future__ import annotations

from .base import DoerBase


class EvalDoer(DoerBase):
    """Evaluation specialist — owns sliding-window, decoding, eval-path knobs."""

    specialist = "eval"
