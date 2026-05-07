"""Harness layer for Parameter Golf.

PG-side mechanical plumbing: path constants, artifact-size thresholds,
experiment tracking, and the filelock-protected blackboard. Most
modules thinly re-export from `agent_core.harness`; this package adds
PG-specific extensions (size thresholds in `size_check.py`, PG path
defaults in `config.py`).

Everything here is agent-agnostic; the specialists in `../agents/`
sit on top.
"""

from __future__ import annotations
