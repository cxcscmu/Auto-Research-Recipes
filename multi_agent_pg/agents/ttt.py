"""Test-Time Training specialist — online adaptation at eval time."""

from __future__ import annotations

from .base import DoerBase


class TTTDoer(DoerBase):
    """TTT specialist — owns test-time adaptation, soft prompts, online updates."""

    specialist = "ttt"
