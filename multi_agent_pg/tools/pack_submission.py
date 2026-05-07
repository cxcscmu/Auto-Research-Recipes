"""
pack_submission.py — external packaging step for the Parameter Golf submission.

Reads the trained artifact pair (train_gpt.py + ckpt/final_model.int6.ptz),
produces the self-extracting packed code (lzma-RAW + base85) that matches the
reference 1.0810 SOTA submission shape, and prints the final submission size.

In the multi_agent deployment this file lives inside `multi_agent/tools/` and is
copied into each specialist workdir by `tools/submit._stage_workdir`. The
Python module `pack_code` helper is also imported directly by `code_inspect.
size_project` and by `train_gpt.py`'s SMOKE_TEST `smoke_pack_bytes` line so
both the pre-flight size estimator and the real post-train pack go through
the exact same bytes.

Usage (invoked by run_trial.sh after a successful train, or manually):

    python pack_submission.py [--code train_gpt.py] [--model ckpt/final_model.int6.ptz]

Outputs:
  ckpt/train_gpt_packed.py       — self-extracting, ready-to-ship code file
  stdout: "Submission size: N bytes"  (N = len(packed_code) + len(model_blob))
"""

from __future__ import annotations

import argparse
import base64
import lzma
import sys
from pathlib import Path

from python_minifier import minify as _python_minify


# Conservative minify flags tuned empirically (Apr 2026).
#
# `rename_locals=True` was tested and produces a *smaller* minified source but
# a *larger* lzma-packed output: random 1-char names break lzma's high-frequency
# token reuse and hurt dictionary efficiency. Likewise `hoist_literals` shifts
# repeated literals into a single binding, which on our codebase shaved a few
# minified bytes but added ~150 packed bytes after lzma. Keep both False.
#
# `remove_literal_statements=True` strips every bare-string statement —
# including all module/func/class docstrings — without touching __doc__
# semantics that any pack-path code relies on. This is the only meaningful
# byte saver we keep on.
_MINIFY_KW = dict(
    rename_locals=False,
    hoist_literals=False,
    remove_literal_statements=True,
)


def _minify(src: str) -> str:
    """Run python_minifier with our tuned flags. On failure (syntax error,
    unsupported construct), fall back to the raw source so pack_code still
    produces a valid self-extracting wrapper — the caller's syntax_check or
    a real run will surface the underlying problem."""
    try:
        return _python_minify(src, **_MINIFY_KW)
    except Exception:
        return src


def pack_code(code: str) -> bytes:
    """lzma-RAW + base85, wrapped in a 2-line self-extracting exec().

    Single source for every byte-count call site (preflight size_project,
    train_gpt.py SMOKE_TEST `smoke_pack_bytes`, agent self-size, run_trial.sh
    CLI). All callers go through this so the byte count is consistent.
    """
    minified = _minify(code)
    raw = lzma.compress(
        minified.encode("utf-8"),
        format=lzma.FORMAT_RAW,
        filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}],
    )
    encoded = base64.b85encode(raw).decode("ascii")
    return (
        f'import lzma as L,base64 as B\n'
        f'exec(L.decompress(B.b85decode("{encoded}"),'
        f'format=L.FORMAT_RAW,filters=[{{"id":L.FILTER_LZMA2}}]))'
    ).encode("utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="train_gpt.py")
    ap.add_argument("--model", default="ckpt/final_model.int6.ptz")
    ap.add_argument("--out", default="ckpt/train_gpt_packed.py")
    args = ap.parse_args()

    code_path = Path(args.code)
    model_path = Path(args.model)
    out_path = Path(args.out)

    if not code_path.is_file():
        print(f"pack_submission: code file not found: {code_path}", file=sys.stderr)
        return 2
    if not model_path.is_file():
        print(f"pack_submission: model file not found: {model_path}", file=sys.stderr)
        return 2

    packed = pack_code(code_path.read_text(encoding="utf-8"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(packed)

    code_bytes = len(packed)
    raw_bytes = len(code_path.read_bytes())
    model_bytes = model_path.stat().st_size
    total = code_bytes + model_bytes
    print(f"Packed code: {code_bytes} bytes (raw={raw_bytes} bytes)")
    print(f"Model blob : {model_bytes} bytes")
    print(f"Submission size: {total} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
