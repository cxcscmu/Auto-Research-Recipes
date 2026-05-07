"""Harness layer for the multi-agent auto-research framework.

This package contains the *mechanical* plumbing shared by every specialist
agent: path constants, artifact-size thresholds, the local Scheduler,
experiment tracking, and the filelock-protected blackboard.

Everything here is Agent-agnostic — it knows nothing about Claude SDK,
prompts, or personas. The specialists in ../agents/ sit on top.
"""

from __future__ import annotations
