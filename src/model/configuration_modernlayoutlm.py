"""
Configuration for ModernLayoutLM

Extends ModernBertConfig with a coordinate vocabulary size and inherits the backbone untouched
This module is copied into exported model directories and may import only from transformers
"""

from __future__ import annotations

from typing import Any

from transformers.models.modernbert.configuration_modernbert import ModernBertConfig

# Size to COORDINATE_SCALE + 1 rows minimum, 1024 matches LayoutLM
DEFAULT_MAX_2D_POSITION_EMBEDDINGS = 1024
DEFAULT_COORDINATE_SCALE = 1000


class ModernLayoutLmConfig(ModernBertConfig):
    """
    Configuration for a ModernLayoutLM model

    Args:
        max_2d_position_embeddings: Rows per 2D table, which lookups clamp coordinates into.
        coordinate_scale: The scale boxes are normalized to, kept here so the processor agrees.
        zero_init_2d: Zero the layout tables, so a fresh graft matches its backbone exactly.
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
        # Set on the class as an instance "model_type" breaks Auto dispatch
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
