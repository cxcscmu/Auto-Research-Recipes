"""Meta-search analyst — narrow hyperparameter tweaks across recent trials.

Differs from the 9 domain doers in exactly one way: the thinking budget
is the ANALYST budget (4K tokens) rather than the DOER budget (8K). The
meta analyst submits via submit_trial on the same multi-submit path as
the other doers; it just focuses on small-radius hyperparameter moves
the domain specialists have left on the table.

Placing meta on the same DoerBase path means it shares every piece of
harness wiring — blackboard reads, user-message rendering, audit,
supervisor scheduling — with the doers. A separate AnalystBase would
duplicate all of that for one tunable.
"""

from __future__ import annotations

from typing import Optional

from ..harness import config
from .base import DoerBase, DoerConfig


class MetaDoer(DoerBase):
    """Meta-search analyst — hyperparameter sweeps, 4K thinking budget."""

    specialist = "meta"

    def __init__(self, cfg: Optional[DoerConfig] = None) -> None:
        if cfg is None:
            cfg = DoerConfig(
                specialist="meta",
                thinking_budget=config.ANALYST_THINKING_BUDGET_TOKENS,
            )
        super().__init__(cfg=cfg)
