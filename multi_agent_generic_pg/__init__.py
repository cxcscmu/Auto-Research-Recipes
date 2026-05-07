"""multi_agent_generic_pg — Parameter Golf, generic-multi-agent variant.

Strict 10× replica of `single_agent_pg`, NOT a fork of `multi_agent_pg`.
Ten generic-prompt specialists (gena..genj) run in parallel, each
carrying the SAME generalist preamble that `single_agent_pg` uses.
There is no role decomposition: every specialist sees the same scope
statement, same edit-radius guidance, same counterfactual discipline.
Specialist names are coordinate labels (workdir / job-name namespacing),
not roles. The 4-letter `gen<x>` shape (x ∈ a..j) is required by
`agent_core.harness.config:make_job_name` constraint on the job name's
domain segment plus the `domain[:4]` truncation in `job_name` —
see `task_config.py:GENERIC_DOMAINS` for the full rationale.

Position in the experimental design:

    multi_agent_pg          : 10 specialists, role-decomposed prompts
    multi_agent_generic_pg  : 10 specialists, single shared prompt        ← this package
    single_agent_pg         : 1  specialist,  same shared prompt

Together these three runs at matched trial budget separate the two
confounding axes that `single vs swarm` alone cannot:

  - parallelism      = (this package) vs single_agent_pg
  - role decomposition = multi_agent_pg vs (this package)

Built strictly on top of agent_core (unchanged). PG-specific
infrastructure (train_gpt.py, run_trial.sh, knowledge/, tools) is
symlinked or re-exported from multi_agent_pg verbatim — no
duplication, no fork.

Importing this package registers `GenericMultiPGTaskAdapter` with
`agent_core`, making `agent_core.current_adapter()` resolve
to the generic-multi-agent adapter for downstream code.
"""

from __future__ import annotations

from agent_core import register_task_adapter

# Eagerly import single_agent_pg.agents.prompts here so its module load
# (which carries a `register_task_adapter(SinglePGTaskAdapter())` side-
# effect via single_agent_pg/__init__.py) happens NOW, before we register
# our own adapter. Otherwise the import would happen lazily on the first
# call to GenericMultiPGTaskAdapter.build_system_prompt() during doer
# execution, and at that lazy moment single_agent_pg's adapter would
# overwrite ours mid-run — breaking domain validation in DoerBase.__init__
# (the doer would see `current_adapter().all_domains == ('generalist',)`
# and reject 'gene_<i>'). Loading the module here forces the side-effect
# to complete during package import, so Python caches the module and
# downstream `from single_agent_pg...` imports inside our prompts.py are
# no-ops that no longer touch the active-adapter slot.
import single_agent_pg.agents.prompts  # noqa: F401  (eager-load for cache)

from multi_agent_generic_pg.task_config import GenericMultiPGTaskAdapter

register_task_adapter(GenericMultiPGTaskAdapter())
