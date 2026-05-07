"""multi_agent_generic_pg agents/base.py — re-export from core.

Identical shim to `single_agent_pg/agents/base.py` and
`multi_agent_pg/agents/base.py`. Implementation lives in
`agent_core.agents.base`; reads score_field / baseline_filename
/ custom_tool_names / build_system_prompt via the active task adapter
at call time. GenericMultiPGTaskAdapter delegates build_system_prompt
to multi_agent_generic_pg.agents.prompts.
"""

from __future__ import annotations

from agent_core.agents.base import (                 # noqa: F401
    DoerBase, DoerConfig, IterRecord,
    build_system_prompt,
    _PRELOAD_BUILTIN_TOOLS, _MCP_SERVER_NAME,
    _LEADERBOARD_MAX_BYTES, _KNOWLEDGE_MAX_BYTES, _RECENT_N,
    _QUARANTINED_STATUSES,
    _SATURATION_EPSILON, _SATURATION_WINDOW,
    _truncate, _read_md_safely,
    _render_recent_activity, _render_saturation_warning,
    render_user_message, _render_workdir_state,
    _emit_tool_result_event, _parse_tool_result_content, _simplify_usage,
)
