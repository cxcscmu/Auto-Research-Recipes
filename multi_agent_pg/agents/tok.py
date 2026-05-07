"""Tokenizer specialist — vocab size, segmentation, BPE/SP variants."""

from __future__ import annotations

from .base import DoerBase


class TokDoer(DoerBase):
    """Tokenizer specialist — owns vocab, segmentation, and tokenizer pipeline."""

    specialist = "tok"
