"""Offline query: how are the specialists using their tools?

Reads `blackboard/events.jsonl` (append-only; never locked/rotated) and
prints three views:

  1. Per-tool call totals across all sessions
  2. Per-specialist breakdown (which domain leans on which tool)
  3. PR-library adoption: fraction of sessions that called at least one
     `read_pr_library` or `read_pr_source`

This script is intentionally standalone — it does NOT modify the
blackboard, hot-path, or dashboard. Re-run anytime to see current
usage. No telemetry added; relies on the per-call `tool_called` events
that the supervisor has always emitted, plus the `tool_calls_by_name`
aggregate added to `session_end` for fast per-session lookup.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from ..harness import config


_PR_TOOLS = ("read_pr_library", "read_pr_source")


def _load_events(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _summarize(events: list[dict]) -> dict:
    total_by_tool: Counter = Counter()
    by_spec_tool: dict[str, Counter] = defaultdict(Counter)
    sessions_started: Counter = Counter()
    sessions_with_pr: Counter = Counter()

    # Walk session_start / session_end pairs per (spec). We don't need
    # perfect pairing — counting sessions_started from session_start is
    # enough, and session_end carries tool_calls_by_name for the session.
    for ev in events:
        etype = ev.get("event")
        spec = ev.get("spec", "?")
        if etype == "session_start":
            sessions_started[spec] += 1
        elif etype == "tool_called":
            tool = ev.get("tool", "?")
            total_by_tool[tool] += 1
            by_spec_tool[spec][tool] += 1
        elif etype == "session_end":
            by_name = ev.get("tool_calls_by_name") or {}
            if any(by_name.get(t, 0) > 0 for t in _PR_TOOLS):
                sessions_with_pr[spec] += 1

    return {
        "total_by_tool": total_by_tool,
        "by_spec_tool": dict(by_spec_tool),
        "sessions_started": sessions_started,
        "sessions_with_pr": sessions_with_pr,
    }


def _print_report(s: dict) -> None:
    total_by_tool: Counter = s["total_by_tool"]
    by_spec_tool: dict[str, Counter] = s["by_spec_tool"]
    sessions_started: Counter = s["sessions_started"]
    sessions_with_pr: Counter = s["sessions_with_pr"]

    print("── Total calls per tool (all specs, all sessions) ──")
    if not total_by_tool:
        print("  (no tool calls recorded)")
    else:
        width = max(len(t) for t in total_by_tool)
        for tool, n in total_by_tool.most_common():
            marker = "  ★" if tool in _PR_TOOLS else ""
            print(f"  {tool:<{width}}  {n:>6}{marker}")

    print()
    print("── Per-specialist tool breakdown ──")
    if not by_spec_tool:
        print("  (no specialists seen)")
    else:
        for spec in sorted(by_spec_tool):
            calls = by_spec_tool[spec]
            items = ", ".join(f"{t}={n}" for t, n in calls.most_common())
            print(f"  {spec:<6}  {items}")

    print()
    print("── PR-library adoption (sessions with ≥1 read_pr_library or read_pr_source) ──")
    all_specs = sorted(set(sessions_started) | set(sessions_with_pr))
    if not all_specs:
        print("  (no sessions yet)")
        return
    print(f"  {'spec':<6}  {'sessions':<10} {'with_PR_tool':<14} {'rate'}")
    total_sessions = 0
    total_with_pr = 0
    for spec in all_specs:
        s_total = sessions_started.get(spec, 0)
        s_pr = sessions_with_pr.get(spec, 0)
        total_sessions += s_total
        total_with_pr += s_pr
        rate = f"{(s_pr / s_total * 100):5.1f}%" if s_total else "    —"
        print(f"  {spec:<6}  {s_total:<10} {s_pr:<14} {rate}")
    rate_all = f"{(total_with_pr / total_sessions * 100):5.1f}%" if total_sessions else "    —"
    print(f"  {'TOTAL':<6}  {total_sessions:<10} {total_with_pr:<14} {rate_all}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--events",
        default=None,
        help="Path to events.jsonl (default: config.BLACKBOARD_DIR/events.jsonl)",
    )
    args = ap.parse_args(argv)

    path = Path(args.events) if args.events else config.BLACKBOARD_DIR / "events.jsonl"
    events = _load_events(path)
    print(f"read {len(events)} events from {path}")
    print()
    _print_report(_summarize(events))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
