"""
A masked language modelling collator that keeps the boxes

The stock collator drops any column it does not recognize, so a layout model would train on none
Text is masked and boxes are left untouched, so recovering a token rewards reading its position
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

# Keep BERT's split as ModernBERT raises only the overall rate
REPLACE_FRACTION = 0.8
RANDOM_FRACTION = 0.5

IGNORE_INDEX = -100


@dataclass
class LayoutMlmCollator:
    """
    Masks tokens for MVLM while passing boxes through unchanged

    Args:
        tokenizer: Supplies the mask token id, the vocabulary size, and the special token mask.
        mlm_probability: Fraction of tokens selected for prediction. ModernBERT pretrains at 0.3,
            not BERT's 0.15.
        pad_to_multiple_of: Pad the sequence length up to a multiple of this, or None.
    """

    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 0.3
    pad_to_multiple_of: int | None = None

    def __post_init__(self) -> None:
        mask_token_id = self.tokenizer.mask_token_id
        if not isinstance(mask_token_id, int):
            raise ValueError("The tokenizer has no single mask token, so MVLM is impossible")
        self.mask_token_id = mask_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Collates a list of encoded pages into a masked batch

        Args:
            features: Encoded rows, each holding `input_ids`, `attention_mask` and `bbox`.

        Returns:
            A batch of `input_ids`, `attention_mask`, `bbox` and `labels`, where `labels` is
            `IGNORE_INDEX` everywhere except the masked positions.

        Raises:
            KeyError: If a feature is missing `bbox`, which means the tokenization step dropped it.
        """
        for feature in features:
            if "bbox" not in feature:
                raise KeyError(
                    "A feature has no bbox. The tokenizer step dropped the boxes, so this run "
                    "would train a layout model on no layout."
                )

        batch = {
            "input_ids": _stack(features, "input_ids", torch.long),
            "attention_mask": _stack(features, "attention_mask", torch.long),
            "bbox": _stack(features, "bbox", torch.long),
        }
        batch["input_ids"], batch["labels"] = self.mask_tokens(
            batch["input_ids"],
            batch["attention_mask"],
        )
        return batch

    def mask_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Chooses tokens to predict and corrupts them, leaving boxes alone

        Special tokens and padding are never selected. Of the selected tokens, 80 percent become
        the mask token, half of the remainder become a random token, and the rest are left as they
        are so the model cannot assume a masked input means a masked position.

        Args:
            input_ids: Token ids of shape (batch, seq_len).
            attention_mask: Mask of shape (batch, seq_len), zero on padding.

        Returns:
            The corrupted input ids and the label tensor, which is `IGNORE_INDEX` off the
            selected positions.
        """
        labels = input_ids.clone()
        probability = torch.full(labels.shape, self.mlm_probability)

        special = torch.tensor(
            [
                self.tokenizer.get_special_tokens_mask(row, already_has_special_tokens=True)
                for row in labels.tolist()
            ],
            dtype=torch.bool,
        )
        probability.masked_fill_(special, value=0.0)
        probability.masked_fill_(attention_mask == 0, value=0.0)

        selected = torch.bernoulli(probability).bool()
        labels[~selected] = IGNORE_INDEX

        replaced = torch.bernoulli(torch.full(labels.shape, REPLACE_FRACTION)).bool() & selected
        input_ids = input_ids.clone()
        input_ids[replaced] = self.mask_token_id

        randomized = (
            torch.bernoulli(torch.full(labels.shape, RANDOM_FRACTION)).bool() & selected & ~replaced
        )
        random_ids = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[randomized] = random_ids[randomized]

        return input_ids, labels


def _stack(features: list[dict[str, Any]], key: str, dtype: torch.dtype) -> torch.Tensor:
    """
    Stacks one column of a feature list into a tensor

    Args:
        features: The encoded rows.
        key: Which column to stack.
        dtype: Target tensor dtype.

    Returns:
        A tensor whose first dimension is the batch.

    Raises:
        ValueError: If the rows disagree in length, which padding should have prevented.
    """
    values = [feature[key] for feature in features]
    lengths = {len(value) for value in values}
    if len(lengths) != 1:
        raise ValueError(f"Column {key!r} has ragged lengths {sorted(lengths)}; pad first")
    return torch.tensor(values, dtype=dtype)
