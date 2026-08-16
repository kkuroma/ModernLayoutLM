"""
Tests for the MVLM collator

This class exists because `DataCollatorForLanguageModeling` drops `bbox`
The tests that matter check that boxes arrive intact and that only text is corrupted
"""

from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from src.train.collator import IGNORE_INDEX, LayoutMlmCollator


@pytest.fixture(scope="module")
def tokenizer():
    """The pulled ModernBERT tokenizer."""
    backbone = Path("models/backbones/ModernBERT-base")
    if not (backbone / "tokenizer.json").is_file():
        pytest.skip(f"backbone not pulled to {backbone}")
    return AutoTokenizer.from_pretrained(backbone)


def make_features(count: int = 4, length: int = 16) -> list[dict[str, list]]:
    """Builds encoded rows with distinctive boxes, so a dropped column is obvious."""
    return [
        {
            "input_ids": [100 + index * length + position for position in range(length)],
            "attention_mask": [1] * length,
            "bbox": [[position, index, position + 5, index + 5] for position in range(length)],
        }
        for index in range(count)
    ]


def test_boxes_survive_collation(tokenizer) -> None:
    """
    The boxes must come through unchanged, position for position

    This is the failure the class exists to prevent. The stock collator silently drops bbox, and
    the resulting run trains a layout model on no layout while reporting a healthy falling loss.
    """
    features = make_features()
    batch = LayoutMlmCollator(tokenizer=tokenizer)(features)

    assert "bbox" in batch
    assert batch["bbox"].shape == (4, 16, 4)
    expected = torch.tensor([feature["bbox"] for feature in features], dtype=torch.long)
    assert torch.equal(batch["bbox"], expected)


def test_missing_boxes_raise(tokenizer) -> None:
    """A feature without bbox means the tokenizer step lost it, which must not be tolerated."""
    features = make_features()
    del features[1]["bbox"]
    with pytest.raises(KeyError, match="no bbox"):
        LayoutMlmCollator(tokenizer=tokenizer)(features)


def test_labels_only_where_masked(tokenizer) -> None:
    """Everything not selected for prediction must be IGNORE_INDEX."""
    torch.manual_seed(0)
    batch = LayoutMlmCollator(tokenizer=tokenizer, mlm_probability=0.3)(make_features())
    selected = batch["labels"] != IGNORE_INDEX
    assert selected.any(), "nothing was masked at all"
    assert not selected.all(), "everything was masked"


def test_labels_recover_the_original_tokens(tokenizer) -> None:
    """
    Where a label is set, it must hold the token that was originally there

    Otherwise the model is trained to predict whatever the corruption happened to write, which is
    an objective that teaches nothing.
    """
    torch.manual_seed(0)
    features = make_features()
    original = torch.tensor([feature["input_ids"] for feature in features], dtype=torch.long)
    batch = LayoutMlmCollator(tokenizer=tokenizer, mlm_probability=0.5)(features)

    selected = batch["labels"] != IGNORE_INDEX
    assert torch.equal(batch["labels"][selected], original[selected])


def test_masking_rate_is_roughly_the_requested_one(tokenizer) -> None:
    """
    Over a large batch the selected fraction must approach the configured probability

    ModernBERT pretrains at 0.3, not BERT's 0.15, and silently masking half as much would slow
    the whole stage down without any error.
    """
    torch.manual_seed(0)
    batch = LayoutMlmCollator(tokenizer=tokenizer, mlm_probability=0.3)(
        make_features(count=64, length=64)
    )
    rate = (batch["labels"] != IGNORE_INDEX).float().mean().item()
    assert 0.24 < rate < 0.36, rate


def test_padding_is_never_masked(tokenizer) -> None:
    """A masked pad token is a wasted prediction and a spurious gradient."""
    torch.manual_seed(0)
    features = make_features(count=2, length=16)
    for feature in features:
        feature["attention_mask"] = [1] * 8 + [0] * 8
    batch = LayoutMlmCollator(tokenizer=tokenizer, mlm_probability=0.9)(features)
    assert torch.all(batch["labels"][:, 8:] == IGNORE_INDEX)


def test_ragged_rows_raise(tokenizer) -> None:
    """Unpadded rows cannot be stacked, and a clear error beats a torch shape message."""
    features = make_features(count=2, length=16)
    features[1]["input_ids"] = features[1]["input_ids"][:8]
    with pytest.raises(ValueError, match="ragged"):
        LayoutMlmCollator(tokenizer=tokenizer)(features)


def test_tokenizer_without_a_mask_token_raises(tokenizer) -> None:
    """MVLM is impossible without a mask token, so fail at construction not mid-run."""

    class NoMask:
        mask_token_id = None

    with pytest.raises(ValueError, match="mask token"):
        LayoutMlmCollator(tokenizer=NoMask())  # type: ignore[arg-type]
