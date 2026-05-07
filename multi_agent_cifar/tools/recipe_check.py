"""recipe_check — CIFAR's cheap pre-submit diagnostic tool.

Replaces PG's `size_project` (which packs + measures bytes against 16 MB).
CIFAR has no size cap; instead we emit useful preflight diagnostics:

  * syntax_ok      — py_compile passes
  * param_count    — AST estimate of trainable params
  * estimated_train_s — closed-form from the recipe's epochs × per-epoch
                        wallclock (calibrated by Phase-1 baseline)
  * recipe_summary — one-line summary of arch + opt + aug knobs
  * warnings       — likely-issue strings (param blow-up etc.)

Always returns verdict="ok" — this is a diagnostic, not a gate. A real
crash will surface as `preflight_crash` (syntax) or `crash` (runtime).

The tool is exposed under the same MCP name (`recipe_check`) used by the
adapter's `custom_tool_names` list.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agent_core.tools import tool
from agent_core.tools.code_inspect import _syntax_check_impl, _param_count_impl


# Calibrated against airbench96 baseline (~14 s on GPU for 45 epochs at
# batch_size 1024). Per-epoch is data-pass-bound; per_epoch ≈ 0.31 s on
# GPU at the baseline width 128/384/512. Real numbers come from the
# operator's calibrate_baseline.sh. Compile warmup is ~30-60 s on cold node.
_PER_EPOCH_S_H100 = 0.31
_PER_EPOCH_S_WARMUP = 45.0     # cold torch.compile warmup, one-time per process
_BASELINE_PARAMS = 30_000_000  # airbench96 ~30M params (vs airbench94's ~1.45M)
_BASELINE_BS = 1024            # airbench96 hyp['opt']['batch_size']


def _mcp(result: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(result, default=str)}]}


# ── recipe_check impl ────────────────────────────────────────────────────────

# airbench96 stores epochs in hyp dict: `'train_epochs': 45.0`. The recipe
# may also override via SMOKE_TEST branch. Match the hyp dict literal.
_EPOCHS_RX = re.compile(r"['\"]train_epochs['\"]\s*:\s*([\d\.]+)")
_RUNS_RX   = re.compile(r"\bRUNS\b\s*=\s*int\(os\.environ\.get\(['\"]RUNS['\"]")
# airbench96: `'batch_size': 1024` in hyp; pick LAST occurrence (the
# effective value, which the recipe sometimes shadows for the masked path).
_BS_RX = re.compile(r"['\"]batch_size['\"]\s*:\s*(\d+)")


def _recipe_check_impl(workdir: str) -> dict[str, Any]:
    src_path = Path(workdir) / "airbench96.py"
    if not src_path.is_file():
        return {
            "syntax_ok": False,
            "syntax_err": f"baseline file missing: {src_path}",
            "param_count": None,
            "estimated_train_s": None,
            "recipe_summary": None,
            "warnings": [f"missing: {src_path.name}"],
            "code_bytes": 0,
        }

    syn = _syntax_check_impl(workdir)
    syntax_ok = bool(syn.get("ok"))

    code = src_path.read_text(encoding="utf-8", errors="replace")
    code_bytes = len(code.encode("utf-8"))

    pc = _param_count_impl(workdir)
    param_count = pc.get("total_params") if pc.get("ok") else None

    # Recipe knobs (cheap regex on hyp dict literals).
    epochs_m = _EPOCHS_RX.search(code)
    epochs   = float(epochs_m.group(1)) if epochs_m else None
    bs_matches = _BS_RX.findall(code)
    batch_size = int(bs_matches[-1]) if bs_matches else None
    runs_present = bool(_RUNS_RX.search(code))

    # Estimate per-seed train_s. Per-epoch is roughly inversely proportional
    # to batch_size (linear in steps/epoch). Compile warmup is one-time per
    # cold process. The N=10 seed loop in run_trial.sh adds 10× to total trial
    # but per-seed compile cache helps after first.
    estimated_train_s = None
    if epochs is not None:
        bs_factor = (_BASELINE_BS / batch_size) if batch_size else 1.0
        estimated_train_s = _PER_EPOCH_S_WARMUP + epochs * _PER_EPOCH_S_H100 * bs_factor

    # Warnings.
    warnings: list[str] = []
    if not syntax_ok:
        warnings.append(f"syntax error: {syn.get('error', '')[:160]}")
    if not runs_present:
        warnings.append("RUNS env-var contract missing — recipe may have lost the "
                        "CIFAR-FORK 2 patch; see airbench96.py header")
    if param_count == 0:
        warnings.append(
            "static AST param count is 0 — airbench's CifarNet uses a custom "
            f"`Conv(nn.Conv2d)` subclass + dynamic widths, so the static "
            f"scanner can't sum them. airbench96 baseline is ~{_BASELINE_PARAMS:,}; "
            f"rely on actual trial output if you need verified param count.")
    if estimated_train_s is not None and estimated_train_s > 220:
        warnings.append(f"estimated per-seed train_s {estimated_train_s:.1f}s "
                        f"approaches 240 s per-seed timeout — risk of "
                        f"train_budget_overrun. Total trial across N=10 seeds "
                        f"is ~10× this estimate; trial budget is 2400 s.")

    summary_parts = []
    if epochs is not None:
        summary_parts.append(f"epochs={epochs}")
    if batch_size is not None:
        summary_parts.append(f"bs={batch_size}")
    if param_count is not None:
        summary_parts.append(f"params≈{param_count:,}")
    recipe_summary = ", ".join(summary_parts) or "(unable to extract)"

    return {
        "syntax_ok":         syntax_ok,
        "param_count":       param_count,
        "estimated_train_s": estimated_train_s,
        "recipe_summary":    recipe_summary,
        "warnings":          warnings,
        "code_bytes":        code_bytes,
    }


# ── @tool wrapper ────────────────────────────────────────────────────────────

@tool(
    "recipe_check",
    (
        "Validate the CIFAR airbench96 recipe before submit. Returns: "
        "{syntax_ok, param_count, estimated_train_s, recipe_summary, warnings}. "
        "syntax_ok: airbench96.py compiles. "
        "param_count: total trainable params (informational; AST-only, often "
        "underestimates due to dynamic widths; airbench96 baseline ~30M). "
        "estimated_train_s: per-seed estimate; trial total ≈ 10× this. "
        ">240 s per-seed timeout / >2400 s trial total = train_budget_overrun. "
        "recipe_summary: one-line summary of train_epochs + batch_size + params. "
        "warnings: likely-issue strings. Use BEFORE submit_trial to catch issues "
        "without burning GPU time. (Diagnostic only — never blocks submit.)"
    ),
    {
        "type": "object",
        "properties": {
            "workdir": {"type": "string"},
        },
        "required": ["workdir"],
    },
)
async def recipe_check(args: dict[str, Any]) -> dict[str, Any]:
    return _mcp(_recipe_check_impl(args["workdir"]))


__all__ = ["recipe_check", "_recipe_check_impl"]
