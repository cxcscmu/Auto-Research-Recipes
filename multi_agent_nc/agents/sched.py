"""Schedule specialist — NC-only domain.

Owns time-shape: LR / momentum / weight-decay schedules + training
horizon computation. Main vendor file: scripts/base_train.py
(`get_lr_multiplier / get_muon_momentum / get_weight_decay` + horizon
math from `--target-param-data-ratio` / `--num-iterations` / `--target-flops`).
"""

from __future__ import annotations

from .base import DoerBase


class SchedDoer(DoerBase):
    """Schedule specialist — owns LR/momentum/wd shape + train horizon."""

    specialist = "sched"
