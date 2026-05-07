"""NC cold-start baseline calibration.

Run the unedited baseline through the live pipeline ≥1 time before
launching the swarm, then feed the measured core_metric value(s) here:

    python -m multi_agent_nc.calibrate_baseline --score 0.301 \\
        [--score 0.296 --score 0.305] [--note "h100 node, seeds 0/1/2"]

This avoids letting `adapter.baseline_score_default` (a placeholder
guess) pollute the first ~10 specialist iters' delta_vs_best signal.
"""

from __future__ import annotations

import sys

from agent_core import register_task_adapter
from agent_core.calibrate_baseline import main
from .task_config import NCTaskAdapter


if __name__ == "__main__":
    register_task_adapter(NCTaskAdapter())
    sys.exit(main())
