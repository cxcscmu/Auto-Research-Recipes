"""PG-side code-inspection tools.

Generic helpers (`syntax_check`, `param_count`) live in
`agent_core.tools.code_inspect` and are re-exported here so PG
code that does `from multi_agent_pg.tools.code_inspect import X`
continues to resolve.

PG-only `size_project` (knows the 16 MB cap, calls PG's pack_code) is
defined locally in this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Re-exports of generic tools from core (preserves submodule-level imports
# in verify_candidate.py and elsewhere).
from agent_core.tools.code_inspect import (         # noqa: F401
    syntax_check, _syntax_check_impl,
    param_count, _param_count_impl,
    _mcp,
)

from . import tool
from .pack_submission import pack_code
from ..harness import size_check


# ── size_project (PG-specific: 16 MB cap, calls PG pack_code) ────────────────


def _size_project_impl(workdir: str) -> dict[str, Any]:
    """Pack train_gpt.py locally, report code bytes + warn/block verdict.

    Goes through pack_submission.pack_code (single source of truth) so the
    preflight estimate matches what run_classify.py will measure post-run,
    including the auto comment/docstring strip step.
    """
    wd = Path(workdir)
    code_path = wd / "train_gpt.py"
    if not code_path.is_file():
        return {"ok": False, "error": f"train_gpt.py not found at {code_path}"}

    try:
        code = code_path.read_text(encoding="utf-8")
    except OSError as e:
        return {"ok": False, "error": f"read failed: {e}"}

    code_bytes = len(pack_code(code))

    # Look for a previously-packed model blob from an earlier trial in the
    # same workdir — gives the agent an authoritative total without having
    # to actually train.
    model_blob = wd / "ckpt" / "final_model.int6.ptz"
    model_bytes: int | None = model_blob.stat().st_size if model_blob.is_file() else None

    total = code_bytes + (model_bytes or 0)
    verdict = "ok"
    if total >= size_check.SIZE_BLOCK_BYTES:
        verdict = "block"
    elif total >= size_check.SIZE_WARN_BYTES:
        verdict = "warn"

    return {
        "ok":               True,
        "code_bytes":       code_bytes,
        "model_bytes":      model_bytes,
        "total_bytes":      total,
        "limit_bytes":      size_check.SIZE_BLOCK_BYTES,
        "headroom_bytes":   size_check.SIZE_BLOCK_BYTES - total,
        "verdict":          verdict,       # ok | warn | block
        "model_bytes_source": (
            "prior_run" if model_bytes is not None else
            "unknown (no ckpt/final_model.int6.ptz in this workdir; "
            "total excludes model)"
        ),
    }


@tool(
    "size_project",
    (
        "Estimate the packed submission size of train_gpt.py by running the "
        "real lzma+base85 pack step locally. Returns code_bytes, the "
        "authoritative model_bytes from a prior run if available (None "
        "otherwise), and a warn/block flag vs the 16,000,000-byte limit. "
        "Use this BEFORE submitting a GPU trial so you catch oversize edits "
        "without burning a job."
    ),
    {
        "type": "object",
        "properties": {
            "workdir": {
                "type": "string",
                "description": "Path to the specialist workdir containing train_gpt.py.",
            },
        },
        "required": ["workdir"],
    },
)
async def size_project(args: dict[str, Any]) -> dict[str, Any]:
    return _mcp(_size_project_impl(args["workdir"]))
