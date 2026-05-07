# Task adapter contract

Every task package registers a single `TaskAdapter` subclass on import. The harness reads everything task-specific through this interface, so to add a new task it is enough to implement an adapter and the corresponding shell entry plus knowledge files.

The full base class lives in `agent_core/task_adapter.py`. The reference implementations are `multi_agent_pg/task_config.py`, `multi_agent_cifar/task_config.py`, and `multi_agent_nc/task_config.py`. The variant adapters in `single_agent_pg/task_config.py` and `multi_agent_generic_pg/task_config.py` show how to subclass an existing task adapter and override only what changes.

## Property categories

| Category | Properties |
| --- | --- |
| Paths | `pkg_root`, `knowledge_dir`, `baseline_filename` |
| TSV schema | `tsv_fields`, `score_field`, `score_lower_is_better`, `score_short_label`, `parse_validate_record`, `empty_validate_row` |
| Specialists | `doer_domains`, `analyst_domains`, `all_domains`, `specialist_classes` |
| Pipeline | `stage_files`, `seed_file`, `editable_tree`, `run_script`, `trial_output_dirs`, `size_check` |
| Tools | `custom_tool_names`, `bind_tools` |
| Prompts | `hard_limits_section`, `specialist_preamble`, `keep_discard_semantics`, `build_system_prompt` |
| Bootstrap | `baseline_score_default`, `baseline_score_flag`, `requires_calibrated_baseline`, `bootstrap_hypothesis`, `baseline_note` |
| Identity | `job_name_prefix` |

## Typical adapter

A task adapter at minimum names the editable recipe, declares the specialist roster, and points the harness at its run script.

```python
class PGTaskAdapter(TaskAdapter):
    @property
    def pkg_root(self) -> Path:
        return Path(__file__).resolve().parent

    @property
    def baseline_filename(self) -> str:
        return "train_gpt.py"

    @property
    def doer_domains(self) -> tuple[str, ...]:
        return ("arch", "opt", "quant", "reg", "loss", "eval",
                "curr", "tok", "ttt")

    @property
    def analyst_domains(self) -> tuple[str, ...]:
        return ("meta",)

    @property
    def specialist_classes(self) -> dict[str, type]:
        from .agents import arch, opt, quant, reg, loss, eval, curr, tok, ttt, meta
        return {
            "arch":  arch.ArchSpecialist,
            "opt":   opt.OptSpecialist,
            ...
        }

    @property
    def run_script(self) -> str:
        return "run_trial.sh"

    @property
    def score_field(self) -> str:
        return "val_bpb"

    @property
    def score_lower_is_better(self) -> bool:
        return True

    @property
    def job_name_prefix(self) -> str:
        return "apg"
```

## Variant adapters

A variant package can subclass an existing adapter and override only the differences. The most common overrides are `pkg_root`, `doer_domains`, `analyst_domains`, `specialist_classes`, and `job_name_prefix`. The variant packages keep their own copies of the run script and knowledge tree by symlinking back into the parent task package, so `pkg_root` continues to resolve sensibly.

See `single_agent_pg/task_config.py` for a one-generalist variant and `multi_agent_generic_pg/task_config.py` for a ten-generic-agent variant.

## Registration

Adapters register themselves on import via `register_task_adapter(...)` at the bottom of each `task_config.py`. The harness fetches the active one through `current_adapter()`. If two adapters register inside the same process, the later one replaces the earlier; this is intentional and is what enables a variant package's `__init__.py` to eager-import a base package's adapter and then register its own override on top.
