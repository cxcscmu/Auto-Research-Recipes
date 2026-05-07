# Architecture

The repository is organised around one task-agnostic core and several task packages that plug into it via a single contract.

## agent_core

`agent_core/` provides everything that does not depend on a specific training recipe.

- `harness/` runs the closed loop. Blackboard, event log, result tracker, baseline audit, and trial dispatch all live here.
- `agents/` wraps the Claude Agent SDK into a specialist session. It owns the per-iteration system prompt assembly, the tool-bind protocol, and the lifecycle hooks.
- `tools/` defines task-shared MCP tools such as `syntax_check`, `param_count`, `diff_snapshots`, `read_snapshot`, and `rebase_to`.
- `supervisor/` is the top-level controller. It iterates trial slots, dispatches them to specialist sessions, persists state, and shuts down cleanly on signals.

The core does not know that a task is Parameter Golf or CIFAR. It reads everything task-specific through the `TaskAdapter` interface declared in `agent_core/task_adapter.py`.

## Task packages

Each task package implements one `TaskAdapter` subclass plus the task-side artefacts the harness needs:

- `task_config.py` defines the adapter and registers it on import.
- `train_gpt.py` (PG) or `airbench96.py` (CIFAR) or `experiment.py` (NC) is the editable recipe.
- `run_trial.sh` is the shell entry the harness invokes for each trial.
- `swarm_config.json` carries per-specialist model assignment.
- `knowledge/` carries static markdown documents pinned at the top of every system prompt.
- `agents/` declares per-domain specialist preambles plus the `prompts.py` assembler.
- `tools/` carries task-specific tools such as `pack_submission` for PG.

The variant packages `single_agent_pg/` and `multi_agent_generic_pg/` are peers of `multi_agent_pg/`. They reuse PG's editable recipe, run script, and knowledge tree by symlink, and override only the adapter to change the specialist roster.

## Closed-loop flow

A submitted trial goes through five steps.

1. The supervisor selects a domain that needs work and starts a Claude Agent SDK session with the corresponding specialist preamble.
2. The agent reads the rendered lineage view, picks a hypothesis, edits its workdir copy of the editable recipe, and calls `submit_trial`.
3. `submit_trial` runs local checks (syntax, projected packed size on PG, recipe shape on CIFAR), then dispatches the trial by invoking `bash run_trial.sh` as a subprocess on the local GPU node.
4. The harness collects the per-trial log, parses the score and status with the task's `run_classify`, and appends a row to `results.tsv` plus an event to `events.jsonl`.
5. The next session reads a freshly rendered lineage view that includes this row and refines its proposal.

External evaluators own the score, the legality checks, and the timing source. The recipe cannot rewrite them.

## Where to look next

- `docs/task_adapter.md` is the property-by-property contract every task package must implement.
