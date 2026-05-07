"""single_agent_pg — Parameter Golf, single-generalist variant.

A peer of `multi_agent_pg` that runs the same closed-loop auto-research
protocol but with **one** generalist agent instead of ten role-specialised
agents. Built as a paired single-vs-swarm baseline for the no-lineage
ablation study (see docs/experiment_results.md §11).

Design:
  * Reuses every piece of agent_core infrastructure unchanged
    (DoerBase, supervisor, blackboard, hooks, sandbox, tools).
  * Reuses every PG-specific piece (train_gpt.py, run_trial.sh,
    knowledge/, pack_submission.py, run_classify.py, run_trainer.py)
    via symlink — no code duplication.
  * Differs only in the task adapter: `doer_domains = ('generalist',)`,
    a generalist preamble that ports the voice + counterfactual
    discipline from the legacy single_agent harness.
  * Default job_name_prefix is 'apg1' so concurrent single-agent runs
    do not cross-kill multi_agent_pg's 'apg-*' jobs on shutdown.

Importing this package registers `SinglePGTaskAdapter` with
`agent_core`, making `agent_core.current_adapter()` resolve
to a single-agent-aware adapter for downstream code (supervisor,
agents, tools, dashboard).
"""

from __future__ import annotations

from agent_core import register_task_adapter
from single_agent_pg.task_config import SinglePGTaskAdapter

register_task_adapter(SinglePGTaskAdapter())
