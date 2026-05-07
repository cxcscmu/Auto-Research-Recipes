"""Artifact size thresholds for the Parameter Golf challenge.

Authoritative byte measurement happens inside run_trial.sh/run_classify.py —
the trial invokes pack_submission.py after training and emits `Submission
size: N bytes` that tracker.parse_validate_result() reads. No estimator lives here;
see single_agent/research/size_check.py for the history of why the heuristic
was removed.

This module exists only to share the two challenge limits with the
blackboard's size-gate rendering and any agent-side tool that needs to quote
the numbers.
"""

from __future__ import annotations

SIZE_WARN_BYTES:  int = 15_950_000   # warn if total ≥ this (50 KB of headroom)
SIZE_BLOCK_BYTES: int = 16_000_000   # hard block (challenge limit, decimal MB)

# Buffer used only when a specialist wants to run a cheap local preflight-pack
# before submitting the GPU job. On the multi-agent side preflight is optional
# (size_check.py's smoke-pack-as-gate policy from single_agent no longer
# applies — we just let run_classify flag oversize artifacts cleanly),
# so this is surfaced only for documentation and the `size_project` tool.
PREFLIGHT_BUFFER_BYTES: int = 1_000
