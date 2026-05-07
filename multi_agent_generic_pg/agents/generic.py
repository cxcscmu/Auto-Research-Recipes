"""Ten generic specialists — gena ... genj — sharing one prompt.

Each subclass is deliberately tiny: all logic lives in DoerBase, and
`specialist = "gen<x>"` (x ∈ a..j) is the only thing that differs
from a generic doer (and from each other). The 10 names exist for
workdir + job-name namespacing only; the SYSTEM PROMPT each agent
receives is identical (see `multi_agent_generic_pg.agents.prompts`).

Naming rationale (see `task_config.py:GENERIC_DOMAINS` for the long
form): `agent_core.harness.config:make_job_name` requires the
middle part of a job name to be 1-4 lowercase letters, and
`job_name` truncates `domain[:4]`. Naming specialists `gene_0..
gene_9` would collapse all 10 to `gene` in the job name and lose
specialist identity in the runner stdout / events.jsonl / dashboard
in-flight panel. `gena..genj` (4 chars, unique under `[:4]`) keeps
every job name distinct — e.g. <prefix>-gena-0001 vs <prefix>-genb-0002.

Mirrors the multi_agent_pg/agents/<role>.py (arch.py / opt.py / ...)
shape so the supervisor + blackboard + audit log treat each generic
agent identically to a multi_agent_pg specialist — only the domain
string differs (and the prompt content does not).
"""

from __future__ import annotations

from .base import DoerBase


class _GenericDoerBase(DoerBase):
    """Common base for the ten generic agents.

    Concrete subclasses override `specialist` only. Defining a per-letter
    class (rather than a single shared class with a per-instance
    specialist attribute) keeps parity with the multi_agent_pg per-role
    pattern (arch.py / opt.py / ...) — DoerBase reads the class attr
    `specialist` at __init__ time, so each per-letter class is the right
    shape for the supervisor's specialist_classes() registry.
    """


def _make_generic_doer(letter: str) -> type[_GenericDoerBase]:
    name = f"gen{letter}"
    cls = type(
        f"Gen{letter.upper()}Doer",
        (_GenericDoerBase,),
        {
            "specialist": name,
            "__doc__": (
                f"Generic specialist {name} — owns the full Parameter Golf "
                f"recipe surface, identical scope to every other gen<x>."
            ),
            "__module__": __name__,
        },
    )
    return cls


# Build the 10 concrete classes once at import time, in alphabetical
# order so the supervisor's specialist_classes() registry produces a
# stable job-name + events.jsonl ordering across runs.
GenADoer = _make_generic_doer("a")
GenBDoer = _make_generic_doer("b")
GenCDoer = _make_generic_doer("c")
GenDDoer = _make_generic_doer("d")
GenEDoer = _make_generic_doer("e")
GenFDoer = _make_generic_doer("f")
GenGDoer = _make_generic_doer("g")
GenHDoer = _make_generic_doer("h")
GenIDoer = _make_generic_doer("i")
GenJDoer = _make_generic_doer("j")


GENERIC_DOER_CLASSES: dict[str, type[_GenericDoerBase]] = {
    "gena": GenADoer,
    "genb": GenBDoer,
    "genc": GenCDoer,
    "gend": GenDDoer,
    "gene": GenEDoer,
    "genf": GenFDoer,
    "geng": GenGDoer,
    "genh": GenHDoer,
    "geni": GenIDoer,
    "genj": GenJDoer,
}
