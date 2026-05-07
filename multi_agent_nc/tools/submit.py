"""Task-side submit tool, re-export from core.

Implementation lives in `agent_core.tools.submit`; reads
stage_files / seed_file / run_script / trial_output_dirs / size_check
via current_adapter() at call time.
"""

from __future__ import annotations

from agent_core.tools.submit import (                # noqa: F401
    submit_trial,
    _submit_trial_impl,
    _stage_workdir,
    _find_result_jsonl,
    _find_result_log,
)
