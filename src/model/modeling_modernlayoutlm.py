"""
ModernLayoutLM: LayoutLM's 2D layout embeddings summed into a ModernBERT backbone

The graft is confined to the embedding module so attention keeps its fused kernels and RoPE
This module is copied into exported model directories and may import only torch and transformers
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn
from transformers.masking_utils import (
    create_bidirectional_mask,
    create_bidirectional_sliding_window_mask,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.models.modernbert.modeling_modernbert import (
    ModernBertEmbeddings,
    ModernBertEncoderLayer,
    ModernBertForMaskedLM,
    ModernBertForSequenceClassification,
    ModernBertForTokenClassification,
    ModernBertModel,
    ModernBertPreTrainedModel,
)
from transformers.utils.auto_docstring import auto_docstring
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from .configuration_modernlayoutlm import ModernLayoutLmConfig

# Bounding boxes arrive as (x0, y0, x1, y1) with the origin at the top left corner
BBOX_COORDS = 4


class LayoutEmbedding(nn.Embedding):
    """
    A coordinate embedding table, marked by its type so weight init can find it

    `_init_weights` is called on leaf modules, not on their container, so an ordinary
    `nn.Embedding` here would be indistinguishable from the token embedding table and would pick
    up ModernBERT's default normal init instead of being zeroed.
    """


class ModernLayoutLmEmbeddings(ModernBertEmbeddings):
    """
    ModernBERT's embeddings plus LayoutLM's four 2D position embedding tables

    Args:
        config: The model config, supplying `max_2d_position_embeddings` and `zero_init_2d`.
    """

    def __init__(self, config: ModernLayoutLmConfig) -> None:
        super().__init__(config)
        max_2d = config.max_2d_position_embeddings
        self.x_position_embeddings = LayoutEmbedding(max_2d, config.hidden_size)
        self.y_position_embeddings = LayoutEmbedding(max_2d, config.hidden_size)
        self.w_position_embeddings = LayoutEmbedding(max_2d, config.hidden_size)
        self.h_position_embeddings = LayoutEmbedding(max_2d, config.hidden_size)
        if getattr(config, "zero_init_2d", True):
            self.reset_layout_parameters()

    def reset_layout_parameters(self) -> None:
        """
        Zeroes the four layout tables so the model reduces exactly to its ModernBERT backbone
        """
        for table in self.layout_tables():
            nn.init.zeros_(table.weight)

    def layout_tables(self) -> list[LayoutEmbedding]:
        """
        Returns the four 2D position embedding tables

        Used by the optimizer to give the freshly grafted parameters their own learning rate.

        Returns:
            The x, y, w and h embedding tables, in that order.
        """
        return [
            self.x_position_embeddings,
            self.y_position_embeddings,
            self.w_position_embeddings,
            self.h_position_embeddings,
        ]

    def layout_embeddings(self, bbox: torch.Tensor) -> torch.Tensor:
        """
        Looks up and sums the six layout embeddings for each box

        Coordinates are clamped into the table's range, and width and height are derived as
        `x1 - x0` and `y1 - y0` then clamped at zero, so a malformed box degrades rather than
        raising. `x0` and `x1` share the X table and `y0` and `y1` share the Y table, matching
        LayoutLM v1.

        Args:
            bbox: Long tensor of shape (batch, seq_len, 4) holding (x0, y0, x1, y1).

        Returns:
            A tensor of shape (batch, seq_len, hidden_size).

        Raises:
            ValueError: If the last dimension of `bbox` is not 4.
        """
        if bbox.shape[-1] != BBOX_COORDS:
            raise ValueError(f"bbox must have last dimension {BBOX_COORDS}, got {bbox.shape[-1]}")

        limit = self.x_position_embeddings.num_embeddings - 1
        boxes = bbox.long().clamp(0, limit)
        x0, y0, x1, y1 = boxes.unbind(dim=-1)
        width = (x1 - x0).clamp(0, limit)
        height = (y1 - y0).clamp(0, limit)

        return (
            self.x_position_embeddings(x0)
            + self.y_position_embeddings(y0)
            + self.x_position_embeddings(x1)
            + self.y_position_embeddings(y1)
            + self.w_position_embeddings(width)
            + self.h_position_embeddings(height)
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        bbox: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Sums token and layout embeddings, then applies the inherited LayerNorm and dropout

        Args:
            input_ids: Token ids of shape (batch, seq_len).
            bbox: Boxes of shape (batch, seq_len, 4). None contributes nothing, which makes the
                model behave as plain ModernBERT.
            inputs_embeds: Precomputed token embeddings, used instead of `input_ids`.

        Returns:
            Normalized embeddings of shape (batch, seq_len, hidden_size).
        """
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.tok_embeddings(input_ids)
        if bbox is not None:
            hidden_states = hidden_states + self.layout_embeddings(bbox)
        return self.drop(self.norm(hidden_states))


class ModernLayoutLmPreTrainedModel(ModernBertPreTrainedModel):
    """
    Shared base class, extending ModernBERT's weight initialization to the layout tables
    """

    config_class = ModernLayoutLmConfig
    base_model_prefix = "model"
    _no_split_modules = ["ModernLayoutLmEmbeddings", "ModernBertEncoderLayer"]

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        """
        Initializes a module, adding layout table handling on top of ModernBERT's rules

        Without this the four tables would pick up ModernBERT's default embedding init, which is
        far too large to add onto a pretrained token embedding. This is also the hook
        `from_pretrained` calls for keys missing from a ModernBERT checkpoint, which is exactly
        the four tables, so it is what makes grafting produce a zeroed start.

        Args:
            module: The module to initialize.
        """
        super()._init_weights(module)
        if isinstance(module, LayoutEmbedding):
            if getattr(self.config, "zero_init_2d", True):
                nn.init.zeros_(module.weight)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)


@auto_docstring
class ModernLayoutLmModel(ModernLayoutLmPreTrainedModel, ModernBertModel):
    """
    The bare ModernLayoutLM encoder

    Identical to `ModernBertModel` except that `bbox` is threaded into the embedding module.
    """

    def __init__(self, config: ModernLayoutLmConfig) -> None:
        super().__init__(config)
        self.embeddings = ModernLayoutLmEmbeddings(config)
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        bbox: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> BaseModelOutput:
        r"""
        bbox (`torch.LongTensor` of shape `(batch_size, sequence_length, 4)`, *optional*):
            Bounding box of each token as (x0, y0, x1, y1), normalized to
            `[0, config.coordinate_scale]` with the origin at the top left. Omitting it makes the
            model behave as plain ModernBERT.
        """
        # Branched rather than written as an xor so that the non-None case is provable
        if inputs_embeds is not None:
            if input_ids is not None:
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
            seq_len, device = inputs_embeds.shape[1], inputs_embeds.device
        elif input_ids is not None:
            seq_len, device = input_ids.shape[1], input_ids.device
        else:
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        hidden_states = self.embeddings(
            input_ids=input_ids,
            bbox=bbox,
            inputs_embeds=inputs_embeds,
        )

        if not isinstance(attention_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": hidden_states,
                "attention_mask": attention_mask,
            }
            attention_mask_mapping = {
                "full_attention": create_bidirectional_mask(**mask_kwargs),
                "sliding_attention": create_bidirectional_sliding_window_mask(**mask_kwargs),
            }

        layer_types: list[str] = self.config.layer_types or []
        position_embeddings = {}
        for layer_type in set(layer_types):
            position_embeddings[layer_type] = self.rotary_emb(
                hidden_states, position_ids, layer_type
            )

        # Cast because nn.Module.__getattr__ erases the element type of self.layers
        for module in self.layers:
            encoder_layer = cast(ModernBertEncoderLayer, module)
            attention_type = encoder_layer.attention_type
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=attention_mask_mapping[attention_type],
                position_embeddings=position_embeddings[attention_type],
                **kwargs,
            )

        hidden_states = self.final_norm(hidden_states)

        return BaseModelOutput(last_hidden_state=hidden_states)


@auto_docstring
class ModernLayoutLmForMaskedLM(ModernLayoutLmPreTrainedModel, ModernBertForMaskedLM):
    """
    ModernLayoutLM with a masked language modeling head, for the layout-aware MLM stage
    """

    _tied_weights_keys = {"decoder.weight": "model.embeddings.tok_embeddings.weight"}

    def __init__(self, config: ModernLayoutLmConfig) -> None:
        super().__init__(config)
        self.model = ModernLayoutLmModel(config)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        bbox: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor] | MaskedLMOutput:
        """
        Runs the encoder and the MLM head

        `bbox` is named explicitly rather than left to `**kwargs` because `Trainer` inspects this
        signature to decide which dataset columns to keep. Were it implicit, the boxes would be
        silently dropped before they ever reached the model.

        Args:
            input_ids: Token ids of shape (batch, seq_len).
            attention_mask: Mask of shape (batch, seq_len).
            bbox: Boxes of shape (batch, seq_len, 4), normalized to the coordinate scale.
            position_ids: Optional explicit 1D positions.
            inputs_embeds: Precomputed token embeddings, used instead of `input_ids`.
            labels: MLM targets of shape (batch, seq_len), `-100` where unmasked.
            **kwargs: Forwarded to the backbone.

        Returns:
            A MaskedLMOutput, or a tuple when the config asks for one.
        """
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            # Ignore as the ModernBert head forwards bbox opaquely and pyright loses it
            bbox=bbox,  # pyright: ignore[reportCallIssue]
            **kwargs,
        )


@auto_docstring
class ModernLayoutLmForTokenClassification(
    ModernLayoutLmPreTrainedModel, ModernBertForTokenClassification
):
    """
    ModernLayoutLM with a token classification head, for form and receipt entity extraction
    """

    def __init__(self, config: ModernLayoutLmConfig) -> None:
        super().__init__(config)
        self.model = ModernLayoutLmModel(config)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        bbox: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor] | TokenClassifierOutput:
        """
        Runs the encoder and the per-token classifier

        Args:
            input_ids: Token ids of shape (batch, seq_len).
            attention_mask: Mask of shape (batch, seq_len).
            bbox: Boxes of shape (batch, seq_len, 4), normalized to the coordinate scale.
            position_ids: Optional explicit 1D positions.
            inputs_embeds: Precomputed token embeddings, used instead of `input_ids`.
            labels: Class index per token, `-100` to ignore.
            **kwargs: Forwarded to the backbone.

        Returns:
            A TokenClassifierOutput, or a tuple when the config asks for one.
        """
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            # Ignore as the ModernBert head forwards bbox opaquely and pyright loses it
            bbox=bbox,  # pyright: ignore[reportCallIssue]
            **kwargs,
        )


@auto_docstring
class ModernLayoutLmForSequenceClassification(
    ModernLayoutLmPreTrainedModel, ModernBertForSequenceClassification
):
    """
    ModernLayoutLM with a pooled sequence classification head, for document classification
    """

    def __init__(self, config: ModernLayoutLmConfig) -> None:
        super().__init__(config)
        self.model = ModernLayoutLmModel(config)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        bbox: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor] | SequenceClassifierOutput:
        """
        Runs the encoder, pools per the config, then classifies the document

        Args:
            input_ids: Token ids of shape (batch, seq_len).
            attention_mask: Mask of shape (batch, seq_len).
            bbox: Boxes of shape (batch, seq_len, 4), normalized to the coordinate scale.
            position_ids: Optional explicit 1D positions.
            inputs_embeds: Precomputed token embeddings, used instead of `input_ids`.
            labels: One class index per document.
            **kwargs: Forwarded to the backbone.

        Returns:
            A SequenceClassifierOutput, or a tuple when the config asks for one.
        """
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            # Ignore as the ModernBert head forwards bbox opaquely and pyright loses it
            bbox=bbox,  # pyright: ignore[reportCallIssue]
            **kwargs,
        )


__all__ = [
    "ModernLayoutLmEmbeddings",
    "ModernLayoutLmForMaskedLM",
    "ModernLayoutLmForSequenceClassification",
    "ModernLayoutLmForTokenClassification",
    "ModernLayoutLmModel",
    "ModernLayoutLmPreTrainedModel",
]
