"""multi_agent_generic_pg system-prompt assembly.

Strict 10× replica of `single_agent_pg.agents.prompts`. The patched
GLOBAL_RULES and the _GENERALIST_PREAMBLE are imported VERBATIM from
single_agent_pg so any future tweak to the single-agent prompt
propagates to this generic-multi-agent variant automatically. Every
generic specialist (gena..genj) receives the SAME assembled system
prompt — the per-specialist coordinate label appears only in the
per-iteration user message (rendered by core).

This is the design point of the generic-multi-agent control: the only
axis differing from single_agent_pg is the number of concurrently
active proposal threads (1 → 10). Differing from multi_agent_pg it
removes role decomposition (10 different preambles → 1 shared
preamble). Together with single_agent_pg, the three runs separate the
parallelism and role-decomposition contributions.

Three pieces, in order (matches single_agent_pg / multi_agent_pg
ordering for shared Anthropic prompt-cache prefix):

  1. **Knowledge files** — task-static markdown (INIT.md /
     SOTA_STACK.md / LESSONS.md), shared with PG verbatim via
     `multi_agent_pg.agents.prompts:_KNOWLEDGE_TEXT`.

  2. **Global rules** — the single-agent-patched GLOBAL_RULES from
     `single_agent_pg.agents.prompts:GLOBAL_RULES`. The opening
     "autonomous ML researcher" framing is correct here too: each
     generic agent is autonomous within its own session, and there is
     no role-level swarm framing.

  3. **Generalist preamble** — `single_agent_pg.agents.prompts:
     _GENERALIST_PREAMBLE` verbatim. Same scope, same counterfactual
     mental check, same edit radius, same anti-anchoring guidance,
     same rollback / branch-from semantics via `rebase_to`.
"""

from __future__ import annotations

# Re-export the single-agent-patched GLOBAL_RULES + the _GENERALIST_PREAMBLE
# so this module's prompt is byte-equal to single_agent_pg's. If the
# single-agent prompt is updated upstream, this generic variant
# auto-inherits the change — there is no second place to patch.
from single_agent_pg.agents.prompts import (
    GLOBAL_RULES as _SINGLE_GLOBAL_RULES,
    _GENERALIST_PREAMBLE as _SINGLE_GENERALIST_PREAMBLE,
)
from multi_agent_pg.agents.prompts import _KNOWLEDGE_TEXT


# Re-bind under local names so external introspection (and possible
# future per-variant tweaks) have a clear hook.
GLOBAL_RULES = _SINGLE_GLOBAL_RULES
_GENERALIST_PREAMBLE = _SINGLE_GENERALIST_PREAMBLE


# ── Registry ────────────────────────────────────────────────────────────────

# Ten generic specialists, all sharing the SAME preamble. The dict
# values point at the same Python string object (no copy) so any
# byte-equality check against single_agent_pg is preserved.
#
# Naming rationale: `agent_core.harness.config:make_job_name` and
# `job_name`'s `domain[:4]` truncation require the first 4
# characters of every domain to (a) match `[a-z]{1,4}` (letters only)
# and (b) be unique across specialists. `gene_0..gene_9` would all
# collapse to "gene" in the job name. `gena..genj` (4 chars, unique
# under `[:4]`) keeps every job name distinct.
_GENERIC_DOMAINS: tuple[str, ...] = tuple(f"gen{c}" for c in "abcdefghij")

DOMAIN_PREAMBLES = {name: _GENERALIST_PREAMBLE for name in _GENERIC_DOMAINS}


def build_system_prompt(domain: str) -> str:
    """Assemble: knowledge md files + (patched) GLOBAL_RULES + generalist preamble.

    `domain` MUST be one of `gena` ... `genj`. All ten domains receive
    the SAME assembled prompt — the per-specialist coordinate label
    appears only in the per-iteration user message.
    """
    if domain not in DOMAIN_PREAMBLES:
        raise ValueError(
            f"multi_agent_generic_pg accepts only gena..genj; got {domain!r}"
        )
    parts: list[str] = []
    if _KNOWLEDGE_TEXT:
        parts.append(_KNOWLEDGE_TEXT)
    parts.append(GLOBAL_RULES)
    parts.append(_GENERALIST_PREAMBLE)
    return "\n\n".join(parts)
