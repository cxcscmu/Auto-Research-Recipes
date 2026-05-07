# Parameter Golf task package

Closed-loop search over the editable Parameter Golf training recipe. Ten specialist agents (architecture, optimizer, tokenizer, quantization, regularization, loss, evaluation, curriculum, test-time training, plus a meta analyst) propose `train_gpt.py` edits, the bundled `LocalScheduler` runs them on an eight-GPU node, and the harness classifies each trial against a 16 MB submission cap and a 600 second train + 600 second eval budget.

## Layout

| Path | Role |
| --- | --- |
| `train_gpt.py` | The editable training recipe. |
| `run_trial.sh` | Trial-side shell entrypoint launched by the scheduler. |
| `run_classify.py`, `run_trainer.py`, `pack_submission.py` (under `tools/`) | Trial-side helpers staged into each workdir. |
| `task_config.py` | `PGTaskAdapter` registered with `agent_core` on import. |
| `agents/` | Per-domain specialist preambles plus `prompts.py` assembler. |
| `tools/` | MCP tools: submit, code-inspect, workdir, pack-submission, run-classify, plus the (optional) PR library readers. |
| `harness/` | PG-specific config + size-check thresholds; thin re-exports of the agent_core harness. |
| `knowledge/` | Static markdown injected into every system prompt: `INIT.md`, `SOTA_STACK.md`, `LESSONS.md`. |
| `swarm_config.json` | Per-specialist scheduler priority + Claude model assignment. |
| `verify_candidate.py` | Optional N-seed Welch t-test of an old-vs-new recipe pair. |
| `dashboard.py` | Read-only live view of `results.tsv` + events.jsonl. |

## Data preparation

Before launching a supervisor, populate `data/parameter_golf/` with the SP8192 FineWeb-edu pretokenised dataset and the matching tokenizer assets, plus a Python virtual environment that includes a Flash-Attention 3 capable `torch` build:

```
data/parameter_golf/
├── venv/                                # python -m venv data/parameter_golf/venv
├── fineweb10B_sp8192/                   # fineweb_train_*.bin + fineweb_val_*.bin
└── tokenizers/                          # SP8192 BPE tokenizer
```

`run_trial.sh` reads `MAGENT_PG_VENV`, `MAGENT_PG_DATA_DIR`, and `MAGENT_PG_TOKENIZER_DIR` if you prefer to keep these assets elsewhere.

## Run

```bash
cp ../.env.example ../.env             # fill ANTHROPIC_API_KEY in .env
pip install -e ..

python -m multi_agent_pg.supervisor \
    --state-root ./magent_state_pg \
    --deadline-hours 48 \
    --no-improvement-hours 4
```

To run a narrower subset of specialists for debugging, pass `--specialists arch,opt,quant,meta`. To replay one specialist's edit through the verifier, see `python -m multi_agent_pg.verify_candidate --help`. To inspect a live run, in a second terminal `python -m multi_agent_pg.dashboard --state-root ./magent_state_pg`.

## Output

Each completed run leaves a self-contained record under `<state-root>/blackboard/`:

- `results.tsv` is the append-only one-row-per-trial log.
- `tree.tsv` is the same rows in preorder-sorted form so a single `awk` slice gets a contiguous subtree.
- `best.json` is the current-best row.
- `events.jsonl` is the within-iter event stream.
- `lineage_snapshots/<session_id>.txt` is the exact rendered user message each specialist saw.
- `snapshots/<exp_id>_<role>/` keeps the frozen workdir at every keep, plus `snapshots/all/<exp_id>.py` for every submitted trial.
