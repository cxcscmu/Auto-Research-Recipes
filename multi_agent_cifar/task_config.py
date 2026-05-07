"""CIFARTaskAdapter — CIFAR-10 airbench96 instantiation of TaskAdapter.

v2 (Apr 28, 2026): strict upstream airbench96 task definition. The agent
swarm minimizes train wallclock subject to mean accuracy (n seeds) ≥
0.96 (single threshold, no buffer band — v2.3 dropped the safety
buffer that was making the gate unreachable at low N). Score field is
`train_s` (lower is better). Below threshold gets `disqualified`.
End-of-deadline final paper number from `verify_candidate.py` at N=30.

Keeps PG-shape result layout. Single editable seed (`airbench96.py`,
the upstream airbench96_faster.py with 4 CIFAR-FORK patches).

Specialists: arch / opt / aug / loss / reg (5 doers, no analyst).
`aug` is new vs PG (no PG analogue — augmentation is the highest-leverage
axis for small CNNs at the airbench96 capacity tier).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_core.task_adapter import TaskAdapter


_PKG_ROOT = Path(__file__).resolve().parent


class CIFARTaskAdapter(TaskAdapter):
    """CIFAR-10 airbench96 adapter — strict min-time-to-96% on 1×GPU."""

    # ── Paths ────────────────────────────────────────────────────────────────

    @property
    def pkg_root(self) -> Path:
        return _PKG_ROOT

    @property
    def knowledge_dir(self) -> Path:
        return _PKG_ROOT / "knowledge"

    @property
    def baseline_filename(self) -> str:
        return "airbench96.py"

    @property
    def seed_file(self) -> str:
        return "airbench96.py"

    @property
    def run_script(self) -> str:
        return "run_trial.sh"

    # ── TSV schema ───────────────────────────────────────────────────────────

    @property
    def tsv_fields(self) -> list[str]:
        return [
            "exp_id", "timestamp", "specialist", "parent_exp", "baseline_exp",
            "domain", "hypothesis", "expected_delta", "status",
            "train_s",           # primary score (lower better) — gated on accuracy
            "accuracy",          # diagnostic (n=10 mean), NOT primary
            "acc_std",           # n=10 std, surfaces noise per trial
            "n_seeds",           # how many seeds went into the row (10 default)
            "delta_vs_best",
            "total_s",
            "job_name", "snapshot_path", "notes",
        ]

    @property
    def score_field(self) -> str:
        return "train_s"

    @property
    def score_short_label(self) -> str:
        return "time"

    @property
    def score_lower_is_better(self) -> bool:
        return True

    def parse_validate_record(self, record: dict) -> dict:
        """Map run_classify aggregated JSONL → TSV row fields.

        v2.3 metric semantics: score = mean train_s (lower better),
        gated on mean_acc(n) ≥ 0.96 (strict upstream airbench96 line, no
        buffer band). DISQUALIFIED trials record train_s=blank so
        leaderboard sort never elevates an under-threshold trial above
        a real keep.

        Status taxonomy:
          OK              → keep             (mean_acc ≥ 0.96, train_s scored)
          DISQUALIFIED    → disqualified     (mean_acc < 0.96; train_s blanked)
          PREFLIGHT_CRASH → preflight_crash
          TIMEOUT         → train_budget_overrun
          CRASH (default) → crash

        BORDERLINE was a self-imposed safety band [0.9585, 0.9615] in
        v2.0–v2.2; removed in v2.3 because it kept actual recipes from
        ever entering the leaderboard at low N. Back-compat: if a
        legacy run_classify still emits raw_status=BORDERLINE, treat
        it the same as OK (acc ≥ 0.96 was the gate even then).
        """
        status_raw = record.get("status", "CRASH")
        status_map = {
            "OK":              "keep",
            "BORDERLINE":      "keep",         # v2.3: treat as keep (back-compat only)
            "DISQUALIFIED":    "disqualified",
            "PREFLIGHT_CRASH": "preflight_crash",
            "TIMEOUT":         "train_budget_overrun",
            "CRASH":           "crash",
        }
        # train_s blank for disqualified so leaderboard sort never elevates
        # an under-threshold trial above a real keep.
        train_s_str = (
            "" if status_raw == "DISQUALIFIED"
            else _fmt_float(record.get("train_s"))
        )
        return {
            "status":      status_map.get(status_raw, "crash"),
            "train_s":     train_s_str,
            "accuracy":    _fmt_float(record.get("accuracy")),
            "acc_std":     _fmt_float(record.get("acc_std")),
            "n_seeds":     str(int(record.get("n_seeds") or 0)),
            "total_s":     _fmt_float(record.get("total_wall_s") or record.get("train_s")),
            "raw_status":  status_raw,
            "kill_reason": record.get("kill_reason") or "",
        }

    def empty_validate_row(self, status: str) -> dict:
        return {
            "status":   status,
            "train_s":  "",
            "accuracy": "",
            "acc_std":  "",
            "n_seeds":  "0",
            "total_s":  "",
        }

    # ── Specialists ──────────────────────────────────────────────────────────

    @property
    def doer_domains(self) -> tuple[str, ...]:
        return ("arch", "opt", "aug", "loss", "reg")

    @property
    def analyst_domains(self) -> tuple[str, ...]:
        # `meta` is intentionally absent for v1: the core user-message
        # contract requires every specialist to call submit_trial, which
        # contradicts an analyst-only role. Re-introduce when a real
        # blackboard-write tool exists (and a separate analyst user-message
        # template that doesn't push submit_trial).
        return ()

    def specialist_classes(self) -> dict[str, type]:
        from multi_agent_cifar.agents import arch, opt, aug, loss, reg
        return {
            "arch": arch.ArchDoer,
            "opt":  opt.OptDoer,
            "aug":  aug.AugDoer,
            "loss": loss.LossDoer,
            "reg":  reg.RegDoer,
        }

    # ── Pipeline / stage / size ──────────────────────────────────────────────

    @property
    def stage_files(self) -> tuple[tuple[str, str], ...]:
        return (
            ("tools/run_classify.py", "run_classify.py"),
            ("run_trial.sh",          "run_trial.sh"),
        )

    @property
    def trial_output_dirs(self) -> tuple[str, ...]:
        return ("full_eval_results", "ckpt", "logs")

    @property
    def pod_env_for_trial(self) -> dict[str, str]:
        """Pass through `MAGENT_CIFAR_DATA_DIR` to run_trial.sh.

        Default points at `<repo>/data/cifar/data/`. Operator can
        override with `MAGENT_CIFAR_DATA_DIR` in their environment.
        """
        import os
        from pathlib import Path
        repo_root = os.environ.get("MAGENT_REPO_ROOT", str(Path.cwd()))
        data_dir = os.environ.get(
            "MAGENT_CIFAR_DATA_DIR",
            f"{repo_root}/data/cifar/data",
        )
        return {"MAGENT_CIFAR_DATA_DIR": data_dir}

    def size_check(self, workdir: str) -> dict:
        """Repurposed as recipe_check: returns informational preflight stats
        (param count + estimated train_s) instead of a 16-MB cap. Verdict
        is always "ok" for CIFAR — no artifact size constraint."""
        from multi_agent_cifar.tools.recipe_check import _recipe_check_impl
        info = _recipe_check_impl(workdir)
        return {
            "ok":            info.get("syntax_ok", True),
            "verdict":       "ok",
            "code_bytes":    info.get("code_bytes", 0),
            "model_bytes":   None,
            "total_bytes":   0,
            "limit_bytes":   None,
            **info,
        }

    # ── Tools ────────────────────────────────────────────────────────────────

    @property
    def custom_tool_names(self) -> tuple[str, ...]:
        return (
            "syntax_check", "recipe_check", "param_count",
            "read_snapshot", "rebase_to", "diff_snapshots",
            "submit_trial", "read_pr_library", "read_pr_source",
        )

    def bind_tools(self) -> list[Any]:
        from multi_agent_cifar.tools import (
            syntax_check, recipe_check, param_count,
            read_snapshot, rebase_to, diff_snapshots,
            submit_trial, read_pr_library, read_pr_source,
        )
        return [
            syntax_check, recipe_check, param_count,
            read_snapshot, rebase_to, diff_snapshots,
            submit_trial, read_pr_library, read_pr_source,
        ]

    # ── Prompts ──────────────────────────────────────────────────────────────

    def build_system_prompt(self, domain: str) -> str:
        from multi_agent_cifar.agents.prompts import build_system_prompt
        return build_system_prompt(domain)

    def specialist_preamble(self, domain: str) -> str:
        from multi_agent_cifar.agents.prompts import DOMAIN_PREAMBLES
        try:
            return DOMAIN_PREAMBLES[domain]
        except KeyError as e:
            raise ValueError(f"unknown domain {domain!r}") from e

    def hard_limits_section(self) -> str:
        return (
            "## Hard limits (enforced by the harness)\n"
            "\n"
            "- **OBJECTIVE**: minimize `train_s` (mean across N=10 seeds) "
            "subject to `mean_accuracy ≥ 0.96` (n=10). Lower train_s wins.\n"
            "- **Threshold gate (single line, strict upstream airbench96)**: "
            "mean_acc ≥ 0.96 → `keep` (train_s scored, snapshot saved, "
            "rebase_to-able). mean_acc < 0.96 → `disqualified` (train_s "
            "blanked; row recorded but never wins).\n"
            "- **Multi-seed contract**: every trial runs N=10 seeds. The "
            "harness's run_trial.sh handles the seed loop; airbench96.py "
            "stays at RUNS=1 per invocation. DO NOT modify the seed loop "
            "or the n=10 default — the task definition is multi-seed.\n"
            "- **Per-trial wallclock**: ~30-50 s/seed on 1×GPU (compile + "
            "~14 s train), so N=10 trial ≈ 3-5 min real (cache warm). "
            "First trial pays cold compile (~5-10 min). Budget per seed "
            "is 240 s in run_trial.sh; total trial ≤ 2400 s.\n"
            "- **No artifact size cap, no param cap, no epoch cap**. "
            "Upstream airbench96 has none; respect that.\n"
            "- The trial subprocess uses one of the eight GPUs; the other "
            "seven sit idle. DO NOT try to use them in parallel from one "
            "experiment.\n"
            "- The N=10 swarm value is an in-search proxy. A larger-N "
            "verification is the canonical paper number.\n"
        )

    def keep_discard_semantics(self) -> str:
        return (
            "## Keep / discard semantics\n"
            "\n"
            "After `submit_trial` returns the harness-computed `status` is one of:\n"
            "\n"
            "- **`keep`** — mean_acc ≥ 0.96 AND mean train_s improved over best. "
            "Snapshot saved; descendants can rebase_to it.\n"
            "- **`discard`** — mean_acc ≥ 0.96 but train_s ≥ best (acc passes, "
            "speed doesn't beat current best).\n"
            "- **`disqualified`** — mean_acc < 0.96; train_s blanked. "
            "The recipe failed the upstream airbench96 acc threshold.\n"
            "- **`crash`** — < N seeds passed per-seed gates (status / acc / shell rc).\n"
            "- **`preflight_crash`** — phase-1 smoke failed.\n"
            "- **`train_budget_overrun`** — total trial > 2400 s.\n"
            "\n"
            "Only `keep` rows snapshot + update best.json + are rebase_to-able.\n"
        )

    # ── Bootstrap (Gap 5) ────────────────────────────────────────────────────

    @property
    def baseline_score_default(self) -> float:
        # v2: airbench96 baseline TIME on GPU ~14-18s (n=10 mean). The
        # exact value MUST be calibrated on this hardware
        # (`calibrate_baseline.sh`), NOT trusted from this default.
        # 16.0s placeholder — supervisor will reject without calibration
        # because requires_calibrated_baseline=True.
        return 16.0

    @property
    def baseline_score_flag(self) -> str:
        return "--baseline-train-s"

    @property
    def requires_calibrated_baseline(self) -> bool:
        # v2: airbench96 train_s on this node IS hardware-dependent (GPU
        # vs GPU vs GPU differ substantially). Force operator to
        # measure before launch so Δ_vs_best is comparable.
        return True

    @property
    def bootstrap_hypothesis(self) -> str:
        return ("airbench96 baseline (upstream KellerJordan, 27.3s on GPU; "
                "expected ~14-18s on GPU / faster on GPU)")

    @property
    def baseline_note(self) -> str:
        return "airbench96 baseline reference (strict: min train_s s.t. mean_acc(n=10) ≥ 0.96)"

    @property
    def job_name_prefix(self) -> str:
        return "cif"            # job names: cif-arch-NNNN, cif-aug-NNNN, ...

    @property
    def scheduler_config(self) -> dict:
        """CIFAR runs all 10 cold-process seeds inside one trial subprocess.

        Each seed completes in ~30-50 s on GPU (compile + ~14 s train),
        so the full N=10 trial takes 3-5 min once the host's compile
        cache is warm. The first trial pays the cold compile (5-10
        min). Per-seed budget is 240 s in run_trial.sh; total trial
        timeout is 2400 s here.
        """
        return {"cuda_visible_devices": "0", "timeout_s": 2400}


def _fmt_float(v) -> str:
    if v is None or v == "":
        return ""
    try:
        return f"{float(v):.6f}"
    except (TypeError, ValueError):
        return ""


__all__ = ["CIFARTaskAdapter"]
