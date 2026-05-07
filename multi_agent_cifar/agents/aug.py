"""Augmentation specialist — CIFAR-only domain (no PG analogue).

Owns the augmentation pipeline: hflip, random crop, mixup/cutout, label
smoothing, augmentation scheduling. Often the highest-leverage axis for
small CNNs at fixed compute.
"""

from __future__ import annotations

from .base import DoerBase


class AugDoer(DoerBase):
    """Augmentation specialist — owns flip / crop / mixup / cutout / TTA settings."""

    specialist = "aug"
