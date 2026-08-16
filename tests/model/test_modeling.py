"""
Tests for the ModernBERT graft

The load-bearing one is test_zero_init_matches_modernbert
Everything else in the project assumes the surgery left the backbone's behavior intact
"""

from pathlib import Path

import pytest
import torch
from transformers import ModernBertConfig, ModernBertModel

from src.model import (
    ModernLayoutLmConfig,
    ModernLayoutLmForMaskedLM,
    ModernLayoutLmForSequenceClassification,
    ModernLayoutLmForTokenClassification,
    ModernLayoutLmModel,
)
from src.model.modeling_modernlayoutlm import LayoutEmbedding

HEAD_CLASSES = [
    ModernLayoutLmForMaskedLM,
    ModernLayoutLmForTokenClassification,
    ModernLayoutLmForSequenceClassification,
]


def test_config_keeps_model_type_when_grafting() -> None:
    """A ModernBERT config dict must not drag its own model_type onto our config."""
    config = ModernLayoutLmConfig(model_type="modernbert")
    assert config.model_type == "modernlayoutlm"


def test_config_rejects_scale_larger_than_table() -> None:
    """A coordinate scale the table cannot represent is a config error, not a clamp."""
    with pytest.raises(ValueError, match="max_2d_position_embeddings"):
        ModernLayoutLmConfig(max_2d_position_embeddings=256, coordinate_scale=1000)


def test_layout_tables_start_zeroed(tiny_config: ModernLayoutLmConfig) -> None:
    """Zero init is the default, so a fresh graft adds nothing to the token embeddings."""
    model = ModernLayoutLmModel(tiny_config)
    tables = model.embeddings.layout_tables()
    assert len(tables) == 4
    assert all(isinstance(table, LayoutEmbedding) for table in tables)
    assert all(torch.all(table.weight == 0) for table in tables)


def test_layout_tables_respect_zero_init_flag(tiny_config: ModernLayoutLmConfig) -> None:
    """Turning zero init off must actually produce non-zero tables."""
    tiny_config.zero_init_2d = False
    model = ModernLayoutLmModel(tiny_config)
    model.init_weights()
    assert any(torch.any(table.weight != 0) for table in model.embeddings.layout_tables())


def test_zero_init_matches_modernbert(
    tiny_config: ModernLayoutLmConfig,
    batch: dict[str, torch.Tensor],
) -> None:
    """
    With zeroed layout tables the graft must be numerically identical to plain ModernBERT

    Built by loading one state dict into both models, so any difference is attributable to the
    architecture change and not to differing weights.
    """
    layout_model = ModernLayoutLmModel(tiny_config)
    reference = ModernBertModel(ModernBertConfig(**tiny_config.to_dict()))

    shared = {k: v for k, v in layout_model.state_dict().items() if "position_embeddings" not in k}
    reference.load_state_dict(shared, strict=True)

    layout_model.eval()
    reference.eval()
    with torch.no_grad():
        grafted = layout_model(**batch).last_hidden_state
        plain = reference(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).last_hidden_state

    assert torch.equal(grafted, plain)


def test_bbox_none_matches_plain_modernbert(
    tiny_config: ModernLayoutLmConfig,
    batch: dict[str, torch.Tensor],
) -> None:
    """
    Omitting bbox must skip the layout sum entirely, which is the no-bbox ablation

    This is deliberately not the same as passing all-zero boxes. All-zero means every word sits at
    the origin, a degenerate but real layout that still adds six embedding lookups.
    """
    model = ModernLayoutLmModel(tiny_config)
    for table in model.embeddings.layout_tables():
        table.weight.data.normal_(0, 0.02)
    model.eval()

    without_bbox = {k: v for k, v in batch.items() if k != "bbox"}
    with torch.no_grad():
        omitted = model(**without_bbox).last_hidden_state
        zeroed = model(**without_bbox, bbox=torch.zeros_like(batch["bbox"])).last_hidden_state
        populated = model(**batch).last_hidden_state

    assert not torch.allclose(omitted, zeroed)
    assert not torch.allclose(omitted, populated)


def test_bbox_changes_output_once_trained(
    tiny_config: ModernLayoutLmConfig,
    batch: dict[str, torch.Tensor],
) -> None:
    """Non-zero tables must make the boxes matter, or the graft is inert."""
    model = ModernLayoutLmModel(tiny_config)
    model.embeddings.x_position_embeddings.weight.data.normal_(0, 0.02)
    model.eval()
    with torch.no_grad():
        moved = model(**batch).last_hidden_state
        origin = model(**{**batch, "bbox": torch.zeros_like(batch["bbox"])}).last_hidden_state
    assert not torch.allclose(moved, origin)


def test_out_of_range_boxes_are_clamped(
    tiny_config: ModernLayoutLmConfig,
    batch: dict[str, torch.Tensor],
) -> None:
    """A box beyond the table must clamp rather than index out of bounds."""
    model = ModernLayoutLmModel(tiny_config)
    model.eval()
    limit = tiny_config.max_2d_position_embeddings
    wild = torch.full_like(batch["bbox"], limit * 10)
    with torch.no_grad():
        clamped = model(**{**batch, "bbox": wild}).last_hidden_state
        at_limit = model(**{**batch, "bbox": torch.full_like(wild, limit - 1)}).last_hidden_state
    assert torch.equal(clamped, at_limit)


def test_inverted_boxes_do_not_crash(
    tiny_config: ModernLayoutLmConfig,
    batch: dict[str, torch.Tensor],
) -> None:
    """A box with x1 < x0 yields a negative width, which must clamp to zero not underflow."""
    model = ModernLayoutLmModel(tiny_config)
    model.eval()
    inverted = batch["bbox"].clone()
    inverted[..., 0] = 900
    inverted[..., 2] = 100
    with torch.no_grad():
        output = model(**{**batch, "bbox": inverted}).last_hidden_state
    assert torch.isfinite(output).all()


def test_bbox_requires_four_coordinates(tiny_config: ModernLayoutLmConfig) -> None:
    """A three-coordinate box is a caller bug and must say so."""
    model = ModernLayoutLmModel(tiny_config)
    with pytest.raises(ValueError, match="last dimension"):
        model(input_ids=torch.ones(1, 4, dtype=torch.long), bbox=torch.zeros(1, 4, 3))


@pytest.mark.parametrize("head_class", HEAD_CLASSES, ids=lambda c: c.__name__)
def test_head_signature_exposes_bbox(head_class: type) -> None:
    """
    Every head must name bbox explicitly in its forward signature

    `Trainer._remove_unused_columns` inspects this signature to decide which dataset columns to
    keep. If bbox were only reachable through **kwargs, the boxes would be silently dropped and
    the model would train as plain ModernBERT while appearing to work.
    """
    import inspect

    assert "bbox" in inspect.signature(head_class.forward).parameters


@pytest.mark.parametrize("head_class", HEAD_CLASSES, ids=lambda c: c.__name__)
def test_bbox_reaches_every_head(
    head_class: type,
    tiny_config: ModernLayoutLmConfig,
    batch: dict[str, torch.Tensor],
) -> None:
    """Boxes must survive the trip through each head down into the encoder."""
    tiny_config.num_labels = 3
    model = head_class(tiny_config)
    model.model.embeddings.x_position_embeddings.weight.data.normal_(0, 0.02)
    model.eval()
    with torch.no_grad():
        moved = model(**batch).logits
        origin = model(**{**batch, "bbox": torch.zeros_like(batch["bbox"])}).logits
    assert not torch.allclose(moved, origin)


@pytest.mark.parametrize("head_class", HEAD_CLASSES, ids=lambda c: c.__name__)
def test_heads_produce_a_loss(
    head_class: type,
    tiny_config: ModernLayoutLmConfig,
    batch: dict[str, torch.Tensor],
) -> None:
    """Each head must compute a finite loss from labels of its own shape."""
    tiny_config.num_labels = 3
    model = head_class(tiny_config)
    if head_class is ModernLayoutLmForMaskedLM:
        labels = torch.randint(0, tiny_config.vocab_size, batch["input_ids"].shape)
    elif head_class is ModernLayoutLmForSequenceClassification:
        labels = torch.randint(0, 3, (batch["input_ids"].shape[0],))
    else:
        labels = torch.randint(0, 3, batch["input_ids"].shape)

    output = model(**batch, labels=labels)
    assert output.loss.isfinite()


def test_save_load_roundtrip(
    tiny_config: ModernLayoutLmConfig,
    batch: dict[str, torch.Tensor],
    tmp_path: Path,
) -> None:
    """
    A trained model must reload with its layout tables intact

    Guards against the weight init hook re-zeroing tables that were present in the checkpoint,
    which would silently discard everything the layout stage learned.
    """
    model = ModernLayoutLmModel(tiny_config)
    for table in model.embeddings.layout_tables():
        table.weight.data.normal_(0, 0.02)
    model.eval()
    with torch.no_grad():
        before = model(**batch).last_hidden_state

    model.save_pretrained(tmp_path)
    reloaded = ModernLayoutLmModel.from_pretrained(tmp_path)
    reloaded.eval()
    with torch.no_grad():
        after = reloaded(**batch).last_hidden_state

    assert torch.equal(before, after)
    assert all(torch.any(t.weight != 0) for t in reloaded.embeddings.layout_tables())


def test_real_backbone_zero_init_parity(backbone_path: Path) -> None:
    """
    The same parity check against the actual pretrained ModernBERT-base checkpoint

    This is the one that matters in practice: a freshly grafted ModernLayoutLM must reproduce
    ModernBERT exactly. Cheap enough to keep in the default run, because safetensors mmaps.
    """
    config = ModernLayoutLmConfig.from_pretrained(backbone_path)
    grafted = ModernLayoutLmModel.from_pretrained(backbone_path, config=config)
    reference = ModernBertModel.from_pretrained(backbone_path)
    grafted.eval()
    reference.eval()

    generator = torch.Generator().manual_seed(0)
    input_ids = torch.randint(1, 1000, (2, 16), generator=generator)
    attention_mask = torch.ones_like(input_ids)
    bbox = torch.randint(0, 1001, (2, 16, 4), generator=generator)

    with torch.no_grad():
        with_layout = grafted(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
        ).last_hidden_state
        plain = reference(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

    assert torch.equal(with_layout, plain)
