# CIFAR-10 Airbench96 task package

Closed-loop search over the editable `airbench96.py` recipe under a strict mean-accuracy gate. Five specialist agents (architecture, optimizer, augmentation, loss, regularization) propose recipe edits; the bundled `LocalScheduler` runs each trial as a single subprocess on one of the host's GPUs; the harness records the shell-measured wallclock and rejects any trial whose mean accuracy across N=10 cold-process seeds falls below 0.96.

## Layout

| Path | Role |
| --- | --- |
| `airbench96.py` | The editable training recipe. |
| `airbench94_muon.py` | Reference baseline (kept for diffing only). |
| `run_trial.sh` | Trial-side shell entrypoint launched by the scheduler. Stages CIFAR-10 to /dev/shm, runs the N-seed loop, calls `run_classify.py`. |
| `task_config.py` | `CIFARTaskAdapter` registered with `agent_core` on import. |
| `agents/` | Per-domain specialist preambles plus `prompts.py` assembler. |
| `tools/` | MCP tools: submit, recipe-check, code-inspect, workdir, run-classify, plus the (optional) PR library readers. |
| `harness/` | CIFAR-specific config + thin re-exports of the agent_core harness. |
| `knowledge/` | Static markdown injected into every system prompt: `INIT.md`, `SOTA_STACK.md`, `LESSONS.md`. |
| `swarm_config.json` | Per-specialist scheduler priority + Claude model assignment. |
| `dashboard.py` | Read-only live view of `results.tsv` + events.jsonl. |

## Data preparation

Before launching a supervisor, populate `data/cifar/` with the standard CIFAR-10 batches and (optionally) a Python virtual environment:

```
data/cifar/
├── venv/              # python -m venv data/cifar/venv (skip if your shell has torch + airbench deps)
└── data/              # cifar-10-batches-py/ as published by Krizhevsky 2009
```

`run_trial.sh` reads `MAGENT_CIFAR_DATA_DIR` and `MAGENT_CIFAR_VENV` if you prefer to keep these assets elsewhere. Set `MAGENT_CIFAR_VENV=skip` to use the calling shell's Python directly.

## Calibrate baseline

CIFAR runtime is hardware-dependent, so the supervisor refuses to cold-start without a measured baseline. Run the unedited recipe once and record the result:

```bash
python -m multi_agent_cifar.calibrate_baseline --score 16.0 --note "h100, n=10"
```

You can pass `--score` multiple times to record several seeds; the supervisor stores their mean.

## Run

```bash
cp ../.env.example ../.env             # fill ANTHROPIC_API_KEY in .env
pip install -e ..

python -m multi_agent_cifar.supervisor \
    --state-root ./magent_state_cifar \
    --deadline-hours 24 \
    --no-improvement-hours 4
```

Each trial runs N=10 cold-process seeds back-to-back inside a single subprocess. Cold compile dominates the first seed (5-10 minutes); subsequent seeds reuse the host's cache and finish in ~30-50 seconds each. Total trial time is therefore 3-5 minutes once the cache is warm. The scheduler `timeout_s` is set to 2400 in `task_config.py:scheduler_config`.

## Output

Same blackboard layout as the other tasks (`results.tsv` / `tree.tsv` / `best.json` / `events.jsonl` / `lineage_snapshots/` / `snapshots/<exp_id>_<role>/`). The CIFAR-specific status semantics are documented in `task_config.py:keep_discard_semantics`; the relevant ones are `keep` (mean acc >= 0.96 and faster than current best), `discard` (mean acc >= 0.96 but slower), and `disqualified` (mean acc < 0.96).
