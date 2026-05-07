"""profile_pipeline — NC's cheap pre-submit diagnostic tool.

Replaces PG's `size_project` (which packs + measures bytes against 16 MB).
NC has no size cap; instead we emit useful preflight diagnostics:

  * syntax_ok          — py_compile passes
  * recipe_summary     — one-line summary of TRAIN_ARGS knobs the agent set
  * num_iterations     — what the agent passed
  * estimated_train_s  — closed-form from num_iterations × s/iter (calibrated
                         by Phase-1 baseline; very rough)
  * warnings           — likely-issue strings (e.g. depth!=12, missing tokenizer)

Always returns verdict="ok" — this is a diagnostic, not a gate. A real
crash will surface as `preflight_crash` (syntax) or `crash` (runtime).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from agent_core.tools import tool


# Calibrated ~rough on GPU. Real numbers come from Phase-1.
_S_PER_ITER_H100 = 0.6           # depth-12 step at typical batch_size, ~0.6s
_COMPILE_S = 90                  # cold torch.compile

_DEPTH_RX           = re.compile(r'"--depth"\s*:\s*"?(\d+)"?')
_NUM_ITER_RX        = re.compile(r'"--num-iterations"\s*:\s*"?(-?\d+)"?')
_BATCH_RX           = re.compile(r'"--total-batch-size"\s*:\s*"?(-?\d+)"?')
_DEVICE_BATCH_RX    = re.compile(r'"--device-batch-size"\s*:\s*"?(\d+)"?')
_MAX_SEQ_LEN_RX     = re.compile(r'"--max-seq-len"\s*:\s*"?(\d+)"?')
_MODEL_TAG_PRESENT  = re.compile(r'"--model-tag"')

# Vendor-walking checks (v2-B). The agent edits files under vendor/, so we
# warn on patterns that almost-certainly break the pipeline before they pay
# for a GPU trial.
_DEPTH_FLAG_RX      = re.compile(r'(--depth|args\.depth)')


def _mcp(result: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(result, default=str)}]}


def _walk_vendor_warnings(workdir: Path) -> tuple[list[str], dict[str, Any]]:
    """Walk workdir/vendor/ and warn on broken or suspicious edits.

    v2-B lets the agent edit anything under vendor/ — the cheapest way to
    catch a wholly-removed --depth flag, a deleted gpt.py, or a nuked
    .commit_pin is a static walk. Returns (warnings, info_dict) where
    info_dict carries cheap stats (vendor_files, commit_pin) for the
    return payload.
    """
    warnings: list[str] = []
    info: dict[str, Any] = {"vendor_files": 0, "commit_pin": None}

    vendor_root = workdir / "vendor" / "nanochat"
    if not vendor_root.is_dir():
        warnings.append(
            f"vendor/nanochat/ missing in workdir — first-iter staging "
            f"failed (expected at {vendor_root}). Subsequent submit_trial "
            "will hard-fail because experiment.py points PYTHONPATH there.")
        return warnings, info

    py_files = [p for p in vendor_root.rglob("*.py") if p.is_file()]
    info["vendor_files"] = len(py_files)
    if len(py_files) < 10:
        warnings.append(
            f"vendor/nanochat/ contains only {len(py_files)} .py files "
            "(expected ~36) — partial copy or accidental deletion?")

    # .commit_pin lives at vendor/.commit_pin (sibling of nanochat/) — it's
    # metadata about the vendored repo, not part of nanochat itself.
    pin = vendor_root.parent / ".commit_pin"
    if pin.is_file():
        info["commit_pin"] = pin.read_text(encoding="utf-8", errors="replace").strip()[:80]
    else:
        warnings.append(
            "vendor/.commit_pin missing — provenance lost. If you "
            "intentionally rebased on a new upstream commit, write the "
            "new sha into vendor/.commit_pin so reviewers can audit the diff.")

    gpt_py = vendor_root / "nanochat" / "gpt.py"
    base_train = vendor_root / "scripts" / "base_train.py"
    if not gpt_py.is_file():
        warnings.append(
            f"{gpt_py.relative_to(vendor_root)} missing — this is the "
            "core transformer file. submit_trial will crash on import.")
    if not base_train.is_file():
        warnings.append(
            f"{base_train.relative_to(vendor_root)} missing — this is "
            "the train entrypoint. submit_trial will crash on launch.")
    elif not _DEPTH_FLAG_RX.search(base_train.read_text(encoding="utf-8", errors="replace")):
        warnings.append(
            "scripts/base_train.py no longer references `--depth` or "
            "`args.depth` — depth plumbing severed. experiment.py passes "
            "--depth=12 and base_train will either ignore it (silent "
            "miscompare) or argparse-error.")

    return warnings, info


def _profile_pipeline_impl(workdir: str) -> dict[str, Any]:
    src_path = Path(workdir) / "experiment.py"
    if not src_path.is_file():
        return {
            "syntax_ok": False,
            "syntax_err": f"baseline file missing: {src_path}",
            "recipe_summary": None,
            "num_iterations": None,
            "estimated_train_s": None,
            "warnings": [f"missing: {src_path.name}"],
            "code_bytes": 0,
        }

    code = src_path.read_text(encoding="utf-8", errors="replace")
    code_bytes = len(code.encode("utf-8"))

    # syntax check (py_compile under the hood)
    syntax_ok = True
    syntax_err = None
    try:
        compile(code, str(src_path), "exec")
    except SyntaxError as e:
        syntax_ok = False
        syntax_err = f"{type(e).__name__}: {e}"

    # Extract TRAIN_ARGS knobs by regex.
    depth_m   = _DEPTH_RX.search(code)
    iters_m   = _NUM_ITER_RX.search(code)
    batch_m   = _BATCH_RX.search(code)
    dbatch_m  = _DEVICE_BATCH_RX.search(code)
    seqlen_m  = _MAX_SEQ_LEN_RX.search(code)
    has_tag   = bool(_MODEL_TAG_PRESENT.search(code))

    depth = int(depth_m.group(1))    if depth_m   else None
    num_iterations = int(iters_m.group(1)) if iters_m else None
    total_batch = int(batch_m.group(1)) if batch_m   else None
    device_batch = int(dbatch_m.group(1)) if dbatch_m  else None
    max_seq_len = int(seqlen_m.group(1)) if seqlen_m  else None

    # Estimate. Per-iter is ~0.6s on GPU at the d12 baseline; if total batch
    # changed by N×, cost scales ~N (more grad-accum micro-steps).
    estimated_train_s = None
    if num_iterations is not None and num_iterations > 0:
        bs_factor = (total_batch / 524288.0) if total_batch and total_batch > 0 else 1.0
        estimated_train_s = _COMPILE_S + num_iterations * _S_PER_ITER_H100 * max(bs_factor, 0.5)

    warnings: list[str] = []
    if not syntax_ok:
        warnings.append(f"syntax error: {syntax_err}")
    if depth is not None and depth != 12:
        warnings.append(f"depth={depth} ≠ 12; d12 miniseries requires depth=12 "
                        "(other values invalidate cross-trial comparison)")
    if not has_tag:
        warnings.append("--model-tag missing in TRAIN_ARGS; checkpoints may "
                        "collide with concurrent specialists' runs")
    base_dir = os.environ.get("NANOCHAT_BASE_DIR", "")
    if base_dir and not Path(base_dir, "tokenizer", "tokenizer.pkl").is_file():
        warnings.append(f"tokenizer missing at {base_dir}/tokenizer/tokenizer.pkl — "
                        "experiment.py will hard-fail; pre-bake via "
                        "`python -m scripts.tok_train`")
    if estimated_train_s is not None and estimated_train_s > 4500:
        warnings.append(f"estimated train_s {estimated_train_s/60:.1f}min near 90 min "
                        f"real-run wall cap — risk of train_budget_overrun")

    # Divisibility check: base_train.py asserts
    #   total_batch_size % (device_batch_size * max_seq_len * world_size) == 0
    # before any training step. World size = NPROC_PER_NODE (default 8).
    #
    # Two cases:
    #   (A) total_batch_size > 0  → user-set; check directly
    #   (B) total_batch_size == -1 → upstream auto-derives from
    #       --target-param-data-ratio. For d12 this lands near 524288. We
    #       check the auto value against the same floor so agents who
    #       move max_seq_len up (e.g. 2048→4096) get warned BEFORE eating
    #       30-90 s of cold-compile cost only to assert.
    nproc = int(os.environ.get("NPROC_PER_NODE", "8"))
    AUTO_TOTAL_BATCH_D12 = 524288    # upstream B_REF for d12 (base_train.py:274)
    if device_batch is not None and max_seq_len is not None:
        floor = device_batch * max_seq_len * nproc
        if floor > 0:
            if total_batch is not None and total_batch > 0:
                # Case A: explicit total_batch
                if total_batch % floor != 0:
                    next_ok = ((total_batch // floor) + 1) * floor
                    warnings.append(
                        f"total_batch_size={total_batch} not divisible by "
                        f"(device_batch_size {device_batch} × max_seq_len "
                        f"{max_seq_len} × world_size {nproc}) = {floor}. "
                        f"base_train will assert before training. Use a "
                        f"multiple of {floor}, e.g. {next_ok}.")
            else:
                # Case B: auto. Apply same check against the d12 auto value.
                if AUTO_TOTAL_BATCH_D12 % floor != 0:
                    # Suggest both: smaller device_batch OR explicit total_batch
                    smaller_dbs = max(1, device_batch // 2)
                    next_ok = ((AUTO_TOTAL_BATCH_D12 // floor) + 1) * floor
                    warnings.append(
                        f"--total-batch-size=-1 (auto~{AUTO_TOTAL_BATCH_D12}) is "
                        f"NOT divisible by (device_batch_size {device_batch} × "
                        f"max_seq_len {max_seq_len} × world_size {nproc}) = "
                        f"{floor}. base_train will assert. Either lower "
                        f"--device-batch-size to {smaller_dbs}, or set "
                        f"--total-batch-size explicitly to a multiple of "
                        f"{floor} (e.g. {next_ok}).")

    # Vendor-walking checks (v2-B): agent can edit any vendor/ .py file, so
    # warn when the staged vendor tree is missing, partial, or has key
    # entrypoints removed. Cheap (~few stat() calls + 2 file reads).
    vendor_warnings, vendor_info = _walk_vendor_warnings(Path(workdir))
    warnings.extend(vendor_warnings)

    summary_parts = []
    if depth is not None:
        summary_parts.append(f"depth={depth}")
    if num_iterations is not None:
        summary_parts.append(f"iters={num_iterations:,}")
    if total_batch is not None:
        summary_parts.append(f"batch={total_batch:,}")
    if max_seq_len is not None:
        summary_parts.append(f"seq_len={max_seq_len}")
    recipe_summary = ", ".join(summary_parts) or "(unable to extract)"

    return {
        "syntax_ok":         syntax_ok,
        "recipe_summary":    recipe_summary,
        "num_iterations":    num_iterations,
        "total_batch_size":  total_batch,
        "device_batch_size": device_batch,
        "max_seq_len":       max_seq_len,
        "depth":             depth,
        "estimated_train_s": estimated_train_s,
        "warnings":          warnings,
        "code_bytes":        code_bytes,
        "vendor_files":      vendor_info["vendor_files"],
        "commit_pin":        vendor_info["commit_pin"],
    }


@tool(
    "profile_pipeline",
    (
        "Validate the NC d12 recipe + vendor tree before submit. Returns: "
        "{syntax_ok, recipe_summary, num_iterations, total_batch_size, "
        "device_batch_size, max_seq_len, depth, estimated_train_s, "
        "vendor_files, commit_pin, warnings}. "
        "estimated_train_s is rough (cold-compile + iters × ~0.6s on GPU); "
        ">90 min real-run wall = train_budget_overrun. "
        "Common warnings: depth ≠ 12 (illegal), missing --model-tag, "
        "missing tokenizer pkl, vendor/ partial or .commit_pin nuked, "
        "scripts/base_train.py severed --depth plumbing. "
        "Use BEFORE submit_trial to catch issues without burning GPU time."
    ),
    {
        "type": "object",
        "properties": {
            "workdir": {"type": "string"},
        },
        "required": ["workdir"],
    },
)
async def profile_pipeline(args: dict[str, Any]) -> dict[str, Any]:
    return _mcp(_profile_pipeline_impl(args["workdir"]))


__all__ = ["profile_pipeline", "_profile_pipeline_impl"]
