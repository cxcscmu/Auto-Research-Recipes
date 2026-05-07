# single_agent_pg

A one-generalist variant of Parameter Golf. Same editable recipe, same scheduler, same blackboard schema as `multi_agent_pg`, but the `doer_domains` reduces to `("generalist",)` and the prompt voice is a single research generalist that owns the entire recipe surface.

This package shares its training recipe, run script, and knowledge tree with `multi_agent_pg/` through filesystem symlinks:

```
single_agent_pg/
├── train_gpt.py        -> ../multi_agent_pg/train_gpt.py
├── run_trial.sh        -> ../multi_agent_pg/run_trial.sh
├── knowledge           -> ../multi_agent_pg/knowledge
├── tools/run_classify.py    -> ../../multi_agent_pg/tools/run_classify.py
├── tools/pack_submission.py -> ../../multi_agent_pg/tools/pack_submission.py
└── tools/run_trainer.py     -> ../../multi_agent_pg/tools/run_trainer.py
```

The variant adapter inherits everything else from `PGTaskAdapter` and overrides only `doer_domains`, `analyst_domains`, `specialist_classes`, and the job-name prefix (`apg1`, distinct from PG's `apg`).

## Run

Same data preparation as `multi_agent_pg/`. Use a separate state root so the single-agent blackboard does not interleave with the multi-agent one:

```bash
python -m single_agent_pg.supervisor --state-root ./magent_state_pg_single
```

## When to use this

This package is a research artifact for single-vs-multi-agent ablations. It runs slower per wallclock than `multi_agent_pg` because it has no parallelism, but it produces the same TSV / events.jsonl shape so the dashboards and trace-analysis scripts work unchanged.
