"""NCTaskAdapter — NanoChat-d12 miniseries instantiation of TaskAdapter.

Task scope:
  * d12 miniseries: 12-layer transformer pretrain, ~110M params
  * Single editable seed: experiment.py (a coordinator wrapping
    `torchrun -m scripts.base_train`); vendor/nanochat lives under
    NANOCHAT_BASE_DIR and is read-only
  * Primary metric: core_metric (higher is better) — emitted by
    base_train internally at the last step (no separate base_eval call)
  * Secondary metric (recorded in TSV but not searched): val_bpb
  * No SFT, no RL, no chat_eval — pretrain-only

Specialists: arch / opt / data / loss / reg (5 doers, no analyst).
`data` replaces PG's `curr` (no curriculum lever in pretrain-only;
data shard mixing + seq-len tuning are the data-side levers).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_core.task_adapter import TaskAdapter


_PKG_ROOT = Path(__file__).resolve().parent


class NCTaskAdapter(TaskAdapter):
    """NanoChat-d12 miniseries adapter — core_metric-at-fixed-compute on 8 GPUs."""

    # ── Paths ────────────────────────────────────────────────────────────────

    @property
    def pkg_root(self) -> Path:
        return _PKG_ROOT

    @property
    def knowledge_dir(self) -> Path:
        return _PKG_ROOT / "knowledge"

    @property
    def baseline_filename(self) -> str:
        return "experiment.py"

    @property
    def seed_file(self) -> str:
        return "experiment.py"

    @property
    def editable_tree(self) -> str:
        # NC v2-B: agent edits across vendor (gpt.py, optim.py, dataloader.py,
        # base_train.py, fp8.py, flash_attention.py, ...). Whole vendor/ tree
        # copied to workdir on first iter; agent freely edits any .py inside.
        # PYTHONPATH in experiment.py points at workdir/vendor/nanochat so
        # all 8 torchrun ranks load the modified code.
        return "vendor"

    @property
    def run_script(self) -> str:
        return "run_trial.sh"

    # ── TSV schema ───────────────────────────────────────────────────────────

    @property
    def tsv_fields(self) -> list[str]:
        return [
            "exp_id", "timestamp", "specialist", "parent_exp", "baseline_exp",
            "domain", "hypothesis", "expected_delta", "status",
            "core_metric",        # primary
            "val_bpb",            # secondary diagnostic
            "delta_vs_best",
            "train_s", "total_s",
            "job_name", "snapshot_path", "notes",
        ]

    @property
    def score_field(self) -> str:
        return "core_metric"

    @property
    def score_short_label(self) -> str:
        return "core"

    @property
    def score_lower_is_better(self) -> bool:
        return False

    def parse_validate_record(self, record: dict) -> dict:
        """Map run_classify JSONL dict → TSV row fields.

        Status taxonomy:
          OK              → keep
          PREFLIGHT_CRASH → preflight_crash
          TIMEOUT         → train_budget_overrun
          CRASH (default) → crash
        """
        status_raw = record.get("status", "CRASH")
        status_map = {
            "OK":              "keep",
            "PREFLIGHT_CRASH": "preflight_crash",
            "TIMEOUT":         "train_budget_overrun",
            "CRASH":           "crash",
        }
        return {
            "status":      status_map.get(status_raw, "crash"),
            "core_metric": _fmt_float(record.get("core_metric")),
            "val_bpb":     _fmt_float(record.get("val_bpb")),
            "train_s":     _fmt_float(record.get("train_s")),
            "total_s":     _fmt_float(record.get("total_wall_s") or record.get("train_s")),
            "raw_status":  status_raw,
            "kill_reason": record.get("kill_reason") or "",
        }

    def empty_validate_row(self, status: str) -> dict:
        return {
            "status":      status,
            "core_metric": "",
            "val_bpb":     "",
            "train_s":     "",
            "total_s":     "",
        }

    # ── Specialists ──────────────────────────────────────────────────────────

    @property
    def doer_domains(self) -> tuple[str, ...]:
        # v2-B specialist redesign (Apr 27):
        # - DROPPED: loss / reg — pretrain CE has no real search surface
        #   exposed by current vendor without writing new training-loop
        #   code; reg at fixed compute is empirically near-zero gain.
        # - KEPT: arch / opt / data — main vendor-file owners.
        # - ADDED: sched (LR/momentum/wd shape + horizon), sys (FP8 + FA3
        #   + compile + numerics). Both are real research dimensions in
        #   nanogpt-speedrun records; both have substantial vendor code.
        return ("arch", "opt", "data", "sched", "sys")

    @property
    def analyst_domains(self) -> tuple[str, ...]:
        # `meta` analyst absent in v1: core's user-message contract pushes
        # every specialist to submit_trial, which contradicts an analyst-only
        # role. Reintroduce when a blackboard-write tool exists.
        return ()

    def specialist_classes(self) -> dict[str, type]:
        from multi_agent_nc.agents import arch, opt, data
        from multi_agent_nc.agents import sched as sched_mod
        from multi_agent_nc.agents import sys as sys_mod
        return {
            "arch":  arch.ArchDoer,
            "opt":   opt.OptDoer,
            "data":  data.DataDoer,
            "sched": sched_mod.SchedDoer,
            "sys":   sys_mod.SysDoer,
        }

    # ── Pipeline / stage / size ──────────────────────────────────────────────

    @property
    def stage_files(self) -> tuple[tuple[str, str], ...]:
        return (
            ("tools/run_classify.py",     "run_classify.py"),
            ("tools/profile_pipeline.py", "profile_pipeline.py"),
            ("run_trial.sh",              "run_trial.sh"),
        )

    @property
    def trial_output_dirs(self) -> tuple[str, ...]:
        # PG-shape so core's cleanup / pull / snapshot work unchanged.
        # base_checkpoints_local mirrors $NANOCHAT_BASE_DIR/base_checkpoints
        # but lives inside the workdir for snapshot cleanup; experiment.py
        # passes --model-tag $WORKDIR_NAME so checkpoints land under that
        # subdir within the NanoChat base directory, not in the workdir.
        return ("full_eval_results", "ckpt", "logs")

    @property
    def pod_env_for_trial(self) -> dict[str, str]:
        """Pass through `NANOCHAT_BASE_DIR` and an optional venv override.

        Default `NANOCHAT_BASE_DIR` resolves to `<repo>/data/nanochat`
        (the package `__init__.py` sets this via os.environ.setdefault).
        Operator overrides via `NANOCHAT_BASE_DIR` directly, or via
        `MAGENT_NC_BASE_DIR` if they prefer task-namespaced env vars.
        """
        import os
        from pathlib import Path
        repo_root = os.environ.get("MAGENT_REPO_ROOT", str(Path.cwd()))
        nb_base = os.environ.get(
            "MAGENT_NC_BASE_DIR",
            os.environ.get(
                "NANOCHAT_BASE_DIR",
                f"{repo_root}/data/nanochat",
            ),
        )
        env = {"NANOCHAT_BASE_DIR": nb_base}
        if "MAGENT_NC_VENV" in os.environ:
            env["MAGENT_NC_VENV"] = os.environ["MAGENT_NC_VENV"]
        return env

    def size_check(self, workdir: str) -> dict:
        """Repurposed as profile_pipeline: returns informational preflight
        diagnostics (recipe knobs + estimated train_s).

        Verdict policy:
          - "block" when ANY warning indicates the recipe will hard-fail
            BEFORE any training step (currently: divisibility asserts in
            base_train.py:407 — substring "base_train will assert"). This
            prevents the swarm from burning ~30-90 s of cold-compile cost
            only to crash at the assert, which would otherwise look like a
            generic CRASH in the lineage and confuse the agent.
          - "ok" for everything else (including non-fatal warnings like
            "param ratio" or "near-cap estimated_train_s").
        """
        from multi_agent_nc.tools.profile_pipeline import _profile_pipeline_impl
        info = _profile_pipeline_impl(workdir)
        warns = info.get("warnings", []) or []
        will_assert = any("base_train will assert" in w for w in warns)
        verdict = "block" if (will_assert or not info.get("syntax_ok", True)) else "ok"
        return {
            "ok":            info.get("syntax_ok", True),
            "verdict":       verdict,
            "code_bytes":    info.get("code_bytes", 0),
            "model_bytes":   None,
            "total_bytes":   0,
            "limit_bytes":   None,
            **info,
        }

    # ── Tools ────────────────────────────────────────────────────────────────

    @property
    def custom_tool_names(self) -> tuple[str, ...]:
        # Same 9 names as PG except size_project → profile_pipeline (NC-specific).
        return (
            "syntax_check", "profile_pipeline", "param_count",
            "read_snapshot", "rebase_to", "diff_snapshots",
            "submit_trial", "read_pr_library", "read_pr_source",
        )

    def bind_tools(self) -> list[Any]:
        from multi_agent_nc.tools import (
            syntax_check, profile_pipeline, param_count,
            read_snapshot, rebase_to, diff_snapshots,
            submit_trial, read_pr_library, read_pr_source,
        )
        return [
            syntax_check, profile_pipeline, param_count,
            read_snapshot, rebase_to, diff_snapshots,
            submit_trial, read_pr_library, read_pr_source,
        ]

    # ── Prompts ──────────────────────────────────────────────────────────────

    def build_system_prompt(self, domain: str) -> str:
        from multi_agent_nc.agents.prompts import build_system_prompt
        return build_system_prompt(domain)

    def specialist_preamble(self, domain: str) -> str:
        from multi_agent_nc.agents.prompts import DOMAIN_PREAMBLES
        try:
            return DOMAIN_PREAMBLES[domain]
        except KeyError as e:
            raise ValueError(f"unknown domain {domain!r}") from e

    def hard_limits_section(self) -> str:
        return (
            "## Hard limits (enforced by the harness)\n"
            "\n"
            "- d12 pretrain on 8 GPUs: pure training ~30-90 min depending on "
            "--num-iterations × batch_size; cold torch.compile adds ~1-3 min.\n"
            "- run_trial wall caps: preflight ≤ 10 min, real run ≤ 90 min "
            "(Phase-1 calibrated). >90 min real wall → train_budget_overrun.\n"
            "- experiment.py is the SINGLE editable seed. vendor/nanochat "
            "(under NANOCHAT_BASE_DIR) is READ-ONLY.\n"
            "- v1 search surface is upstream CLI flags only — do NOT "
            "monkey-patch nanochat.* modules in-process; they don't propagate "
            "to torchrun's child ranks.\n"
            "- No artifact size cap. Submission is the recipe + final core_metric.\n"
        )

    def keep_discard_semantics(self) -> str:
        return (
            "## Keep / discard semantics\n"
            "\n"
            "After `submit_trial` returns the harness-computed `status` is one of:\n"
            "\n"
            "- **`keep`** — VALID + scored, agent decides to keep based on Δ vs best.\n"
            "- **`discard`** — VALID + scored, agent rejects (worse / within noise).\n"
            "- **`crash`** — train child crashed before producing core_metric.\n"
            "- **`preflight_crash`** — syntax_check failed or scheduler submit failed.\n"
            "- **`train_budget_overrun`** — DQ on time (>90 min wall in real run).\n"
        )

    # ── Bootstrap (Gap 5) ────────────────────────────────────────────────────

    @property
    def baseline_score_default(self) -> float:
        # Phase-1 calibrated. Placeholder until operator runs upstream
        # baseline once on our node and records the actual value.
        return 0.30

    @property
    def baseline_score_flag(self) -> str:
        return "--baseline-core"

    @property
    def requires_calibrated_baseline(self) -> bool:
        # NC's default 0.30 is a placeholder; cold-start without
        # calibration would feed specialists a wrong delta_vs_best signal
        # for ~10 iters until the first keep stabilises best.json.
        return True

    @property
    def bootstrap_hypothesis(self) -> str:
        # Vendor commit pin — see the NanoChat base directory/.commit_pin.
        return ("d12 miniseries pretrain baseline "
                "(vendor karpathy/nanochat @ pinned commit)")

    @property
    def baseline_note(self) -> str:
        return "d12 miniseries baseline reference (NANOCHAT_BASE_DIR pre-baked)"

    @property
    def job_name_prefix(self) -> str:
        return "nc"             # job names: nc-arch-NNNN, nc-data-NNNN, etc.

    # ── Scheduler config ──────────────────────────────────────────
    # Default impl on TaskAdapter returns {"priority": scheduler_priority_for(...)}.
    # NC keeps PG default — same 8 GPUs whole-node allocation pattern as PG.


def _fmt_float(v) -> str:
    if v is None or v == "":
        return ""
    try:
        return f"{float(v):.6f}"
    except (TypeError, ValueError):
        return ""


__all__ = ["NCTaskAdapter"]
