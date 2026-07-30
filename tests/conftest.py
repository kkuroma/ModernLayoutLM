"""Shared fixtures. Most tests use a tiny random model so the suite stays in milliseconds."""

from pathlib import Path

import pytest
import torch

from src.model import ModernLayoutLmConfig

BACKBONE_PATH = Path("models/backbones/ModernBERT-base")


@pytest.fixture
def tiny_config() -> ModernLayoutLmConfig:
    """A ModernLayoutLM config small enough to instantiate in a test."""
    return ModernLayoutLmConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=512,
        local_attention=16,
        global_attn_every_n_layers=2,
        pad_token_id=0,
        max_2d_position_embeddings=1024,
        coordinate_scale=1000,
    )


@pytest.fixture
def batch() -> dict[str, torch.Tensor]:
    """A fixed two-sequence batch with random in-range boxes."""
    generator = torch.Generator().manual_seed(0)
    return {
        "input_ids": torch.randint(1, 100, (2, 12), generator=generator),
        "attention_mask": torch.ones(2, 12, dtype=torch.long),
        "bbox": torch.randint(0, 1001, (2, 12, 4), generator=generator),
    }


@pytest.fixture
def backbone_path() -> Path:
    """Path to the pulled ModernBERT backbone, skipping the test when it is absent."""
    if not (BACKBONE_PATH / "config.json").is_file():
        pytest.skip(f"backbone not pulled to {BACKBONE_PATH}")
    return BACKBONE_PATH
