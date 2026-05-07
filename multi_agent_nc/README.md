# NanoChat-D12 task package

Closed-loop search over a vendored nanochat-d12 pretraining recipe. Five specialist agents (architecture, optimizer, data, schedule, systems) edit a writable copy of the nanochat tree under `vendor/nanochat/`; the bundled `LocalScheduler` runs each trial on the eight-GPU node; the harness records the CORE score parsed from the training log. The metric direction is "higher is better" and the baseline is hardware-dependent, so calibration is mandatory before launch.

## Layout

| Path | Role |
| --- | --- |
| `experiment.py` | Coordinator that wraps `torchrun -m scripts.base_train` and writes the per-trial result jsonl. |
| `vendor/nanochat/` | Vendored upstream nanochat tree (MIT, Andrej Karpathy 2025). The agent edits a per-trial writable copy of this tree (gpt.py, optim.py, dataloader.py, base_train.py, fp8.py, flash_attention.py, ...). |
| `run_trial.sh` | Trial-side shell entrypoint launched by the scheduler. Stages tokenizer + shards + eval_bundle to /dev/shm, runs preflight + real-train phases, calls `run_classify.py`. |
| `task_config.py` | `NCTaskAdapter` registered with `agent_core` on import. |
| `agents/` | Per-domain specialist preambles plus `prompts.py` assembler. |
| `tools/` | MCP tools: submit, profile-pipeline (preflight diagnostics), code-inspect, workdir, run-classify, plus the (optional) PR library readers. |
| `harness/` | NC-specific config + thin re-exports of the agent_core harness. |
| `knowledge/` | Static markdown injected into every system prompt: `INIT.md`, `SOTA_STACK.md`, `LESSONS.md`. |
| `swarm_config.json` | Per-specialist scheduler priority + Claude model assignment. |
| `dashboard.py` | Read-only live view of `results.tsv` + events.jsonl. |

## Data preparation

Before launching a supervisor, populate the NanoChat base directory with the upstream pre-baked assets, plus a Python virtual environment that includes a Flash-Attention 3 capable `torch` build:

```
data/nanochat/
├── venv/                   # python -m venv data/nanochat/venv
├── tokenizer/              # nanochat tokenizer assets
├── base_data_climbmix/     # tokenized training shards
└── eval_bundle/            # CORE evaluation pack
```

`run_trial.sh` reads `MAGENT_NC_BASE_DIR` (or the upstream's own `NANOCHAT_BASE_DIR`) and `MAGENT_NC_VENV` if you keep these elsewhere. Set `MAGENT_NC_VENV=skip` to use the calling shell's Python directly.

## Calibrate baseline

The CORE baseline depends on hardware. Run the unedited recipe at least once and record:

```bash
python -m multi_agent_nc.calibrate_baseline --score 0.1618 --note "h100, d12 pretrain, seed 0"
```

Multiple `--score` arguments record multiple seeds and the supervisor stores their mean.

## Run

```bash
cp ../.env.example ../.env             # fill ANTHROPIC_API_KEY in .env
pip install -e ..

python -m multi_agent_nc.supervisor \
    --state-root ./magent_state_nc \
    --deadline-hours 48 \
    --no-improvement-hours 4
```

Each trial is one fresh subprocess running the full d12 pretraining pass. Default per-trial wallclock cap in `task_config.py:scheduler_config` covers the upstream 90-minute pretraining budget plus margin.

## Output

Same blackboard layout as the other tasks. The NC `score_field` is `core_metric`; `val_bpb` is recorded as a secondary diagnostic but is not searched. Status taxonomy is documented in `task_config.py:keep_discard_semantics`.

## Vendor license

`vendor/nanochat/` is the upstream nanochat tree at the pinned commit, redistributed under MIT. See `vendor/nanochat/LICENSE` for the full text. Agent edits modify a per-trial writable copy of this tree; the source under `vendor/` itself is treated as read-only by the harness's stage step.
