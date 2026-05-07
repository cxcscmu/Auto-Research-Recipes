"""Quantization specialist — GPTQ variants, bit-width, calibration."""

from __future__ import annotations

from .base import DoerBase


class QuantDoer(DoerBase):
    """Quantization specialist — owns GPTQ pipeline, bit-width, calibration strategy."""

    specialist = "quant"
