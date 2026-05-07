"""Data specialist — NC-only domain (replaces PG's `curr`).

Owns pretrain data shape: shard mixing, sequence length, batch sizing.
For pretrain-only d12, there's no curriculum knob, but data shard ordering
+ max_seq_len + total_batch_size are real levers.
"""

from __future__ import annotations

from .base import DoerBase


class DataDoer(DoerBase):
    """Data specialist — owns shard mixing / max-seq-len / batch sizing."""

    specialist = "data"
