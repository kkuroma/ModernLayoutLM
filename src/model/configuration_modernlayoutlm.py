"""
Configuration for ModernLayoutLM

Extends `ModernBertConfig` with the one thing ModernBERT lacks, a coordinate vocabulary size for
the 2D layout embedding tables. Everything else about the backbone is inherited untouched.

This module is copied into exported model directories, so it must import only from transformers.
"""

from __future__ import annotations

from typing import Any

from transformers.models.modernbert.configuration_modernbert import ModernBertConfig

# Coordinates are normalized to [0, COORDINATE_SCALE] before lookup, so the table needs at least
# COORDINATE_SCALE + 1 rows. 1024 is the LayoutLM value and leaves headroom.
DEFAULT_MAX_2D_POSITION_EMBEDDINGS = 1024
DEFAULT_COORDINATE_SCALE = 1000


class ModernLayoutLmConfig(ModernBertConfig):
    """
    Configuration for a ModernLayoutLM model

    Args:
        max_2d_position_embeddings: Rows in each 2D position embedding table. Coordinates are
            clamped into `[0, max_2d_position_embeddings - 1]` at lookup time.
        coordinate_scale: The scale bounding boxes are normalized to before lookup. Recorded on
            the config so the processor and the model cannot disagree about it.
        zero_init_2d: Whether to initialize the four layout tables to zero. True makes a freshly
            grafted model numerically identical to its ModernBERT backbone.
        **kwargs: Forwarded to `ModernBertConfig`.
    """

    model_type = "modernlayoutlm"

    def __init__(
        self,
        max_2d_position_embeddings: int = DEFAULT_MAX_2D_POSITION_EMBEDDINGS,
        coordinate_scale: int = DEFAULT_COORDINATE_SCALE,
        zero_init_2d: bool = True,
        **kwargs: Any,
    ) -> None:
        # Grafting reads a ModernBERT config.json, whose "model_type" would otherwise be set as an
        # instance attribute and shadow the class one, breaking Auto dispatch and re-export.
        kwargs.pop("model_type", None)
        super().__init__(**kwargs)
        if coordinate_scale >= max_2d_position_embeddings:
            raise ValueError(
                f"coordinate_scale {coordinate_scale} needs "
                f"max_2d_position_embeddings > {coordinate_scale}, "
                f"got {max_2d_position_embeddings}"
            )
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.coordinate_scale = coordinate_scale
        self.zero_init_2d = zero_init_2d


__all__ = ["ModernLayoutLmConfig"]
