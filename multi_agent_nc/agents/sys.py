"""System / numerics specialist — NC-only domain.

Owns precision and kernel choices: FP8 vs BF16, Flash Attention 3
wiring, torch.compile mode, fused ops. Main vendor files:
nanochat/fp8.py, nanochat/flash_attention.py, scripts/base_train.py
(compile + dtype + FA3 selection logic).
"""

from __future__ import annotations

from .base import DoerBase


class SysDoer(DoerBase):
    """Sys specialist — owns precision/kernel/compile config."""

    specialist = "sys"
