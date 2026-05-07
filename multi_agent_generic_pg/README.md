# multi_agent_generic_pg

A ten-generic-agent variant of Parameter Golf. Same editable recipe as `multi_agent_pg`, but the ten specialists carry an identical generalist preamble (no role decomposition). The job-name prefix is `apgg` so concurrent runs of `multi_agent_pg`, `single_agent_pg`, and this variant do not cross-kill each other on shutdown.

This package shares its training recipe, run script, and knowledge tree with `multi_agent_pg/` through filesystem symlinks:

```
multi_agent_generic_pg/
├── train_gpt.py        -> ../multi_agent_pg/train_gpt.py
├── run_trial.sh        -> ../multi_agent_pg/run_trial.sh
├── knowledge           -> ../multi_agent_pg/knowledge
├── tools/run_classify.py    -> ../../multi_agent_pg/tools/run_classify.py
├── tools/pack_submission.py -> ../../multi_agent_pg/tools/pack_submission.py
└── tools/run_trainer.py     -> ../../multi_agent_pg/tools/run_trainer.py
```

The variant adapter inherits everything else from `PGTaskAdapter` and overrides only `doer_domains` (gena..genj), `specialist_classes`, the prompt assembler (one shared generalist preamble for all ten), and the `apgg` job-name prefix.

The generic preamble is imported from `single_agent_pg.agents.prompts`, so this package depends on `single_agent_pg/` being present.

## Run

Same data preparation as `multi_agent_pg/`. Use a separate state root so the generic-multi-agent blackboard does not interleave with the role-decomposed multi-agent one:

```bash
python -m multi_agent_generic_pg.supervisor --state-root ./magent_state_pg_genmulti
```

## When to use this

This package is a research artifact for parallelism-vs-role-decomposition ablations. It runs at the same wallclock pace as `multi_agent_pg` (ten parallel sessions on the same hardware) but each session sees only the generalist prompt rather than a role-specific one. Comparing the two side by side isolates the effect of role decomposition while holding parallelism constant.
