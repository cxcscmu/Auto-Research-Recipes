# Release artifacts

This directory holds the frozen run records that back the experiments reported in the paper. Each subdirectory is a single supervisor run, named after the role it plays in the analysis.

## Subdirectories

| Path | What it is |
| --- | --- |
| `pg_main/` | Parameter Golf headline run. Final best `val_bpb = 1.072210` at `exp_750`. |
| `pg_ablation_lineage_on/` | Parameter Golf A/B ablation, lineage-on arm. Final best `val_bpb = 1.073142` at `exp_176`. |
| `pg_ablation_lineage_off/` | Parameter Golf A/B ablation, lineage-off arm. Plateau `val_bpb = 1.077413` at `exp_075`. |
| `pg_variant_single_agent/` | Parameter Golf single-generalist variant. Final best `val_bpb = 1.075384`. |
| `pg_variant_generic_multi/` | Parameter Golf ten-generic-agent variant. Final best `val_bpb = 1.074495`. |
| `cifar/` | CIFAR-10 Airbench96 run. Final best `train_s = 25.1464` s under the strict 0.96 mean-accuracy gate. |
| `nanochat_d12/` | NanoChat-D12 run. Final best `core_metric = 0.2244`. |

Plus two top-level lineage snapshots: `example_lineage_pg_main_quant.txt` and `example_lineage_pg_lineage_on_arch.txt` are the exact rendered user messages that two specialists saw at one iteration each.

## Layout

```
<run-name>/
└── blackboard/
 ├── results.tsv one row per submitted trial
 ├── tree.tsv same rows preorder-sorted so subtrees are contiguous
 ├── best.json current-best row at end of run
 ├── KNOWLEDGE.md curated tree of prior hypotheses + outcomes
 ├── LEADERBOARD.md Top-N keep rows by score
 └── snapshots/<exp_id>_<role>/ frozen workdir code at every kept trial
 ├── train_gpt.py (PG / variants)
 ├── airbench96.py (CIFAR)
 ├── experiment.py (NanoChat-D12)
 └── vendor/ (NanoChat-D12 only; pruned to source plus
 pyproject.toml plus LICENSE)
```

`results.tsv` columns: `exp_id`, `timestamp`, `specialist`, `parent_exp`, `domain`, `hypothesis`, `expected_delta`, `status`, the task-specific score field, `delta_vs_best`, `artifact_bytes`, `train_s`, `eval_s`, `total_s`, `job_name`, `snapshot_path`, `notes`.

## What is NOT in each subdirectory

- Per-trial run logs and `events.jsonl` / `supervisor_audit.jsonl`. The trajectory is preserved in `results.tsv`; the multi-megabyte training stdout and runtime telemetry are not.
- `lineage_snapshots/<session_id>.txt`. Two illustrative examples live at the top.
- `snapshots/all/<exp_id>.py`. Per-trial code snapshots for non-keep trials are not bundled.
- Vendor noise (`dev/`, `tests/`, `docs/`, `tasks/`, hidden assistant config directories, `*.ipynb`, `README.md`, `uv.lock`, `.commit_pin`) is stripped from the NanoChat-D12 keep snapshots.
- Working scratch (`workdirs/`, `locks/`, `stop.flag`, `*.bak_*`).

## Anonymization

Every text artifact has been passed through one identity-substitution pipeline. No numeric values (scores, byte counts, wallclock seconds, exp_ids, deltas) were parsed or re-formatted. Specifically:

- Operator usernames in path strings are normalised to `user`.
- Internal cluster identifiers (scheduler name, host names, mount points, GPU pool names) are replaced with neutral placeholders.
- Per-task internal job-name prefixes are stripped from trial identifiers; for example a job formerly named `<prefix>-arch-0123` reads as `arch-0123` here.
- Specific agent-model identifiers and per-iteration budget numbers were carried by the omitted files (`events.jsonl` and `supervisor_audit.jsonl`) and are not exposed.
- The vendored upstream NanoChat tree under `nanochat_d12/blackboard/snapshots/<exp>_<role>/vendor/` is the upstream Karpathy MIT-licensed source plus whatever code-level edits an agent made during that trial. Its `LICENSE` is preserved in place.
