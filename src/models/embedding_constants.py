"""Shared embedding constants to avoid duplication across modules."""

from __future__ import annotations

from typing import Literal

ImageBackboneName = Literal[
    "auto",
    "openclip_vitl14",
    "openclip_vith14",
    "siglip_base",
    "bge_visualized",
]

__all__ = ["ImageBackboneName"]
