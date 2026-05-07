"""PG cold-start baseline calibration.

PG's existing `--baseline-bpb` flag on supervisor still works for the
common case (single trusted baseline value). This script supersedes it
when you want to feed N measured trials and get an averaged best.json:

    python -m multi_agent_pg.calibrate_baseline --score 1.0742 \\
        [--score 1.0738 --score 1.0750] [--note "h100 node, seeds 0/1/2"]

If you only have one trusted score, pass `--baseline-bpb` to supervisor
instead — that's the simpler path.
"""

from __future__ import annotations

import sys

from agent_core import register_task_adapter
from agent_core.calibrate_baseline import main
from .task_config import PGTaskAdapter


if __name__ == "__main__":
    register_task_adapter(PGTaskAdapter())
    sys.exit(main())
