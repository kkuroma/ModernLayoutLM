"""
Tests the handoff from the pretraining stage to fine-tuning

_init_weights zeroes the layout tables only for keys missing from a checkpoint
Firing on present keys would silently reset everything pretraining had learned
"""

import torch

from src.model import (
    ModernLayoutLmConfig,
    ModernLayoutLmForMaskedLM,
    ModernLayoutLmForSequenceClassification,
    ModernLayoutLmForTokenClassification,
)


def test_layout_tables_survive_into_a_downstream_head(
    tiny_config: ModernLayoutLmConfig,
    tmp_path,
) -> None:
    """
    Tables trained during MLM must arrive intact in a token classification head

    Saved from `ForMaskedLM` and loaded into `ForTokenClassification`, which is exactly the
    stage-1 to stage-2 handoff.
    """
    pretrained = ModernLayoutLmForMaskedLM(tiny_config)
    for table in pretrained.model.embeddings.layout_tables():
        table.weight.data.normal_(0, 0.02)
    expected = [table.weight.clone() for table in pretrained.model.embeddings.layout_tables()]
    pretrained.save_pretrained(tmp_path)

    downstream = ModernLayoutLmForTokenClassification.from_pretrained(tmp_path, num_labels=7)
    actual = downstream.model.embeddings.layout_tables()

    assert all(torch.any(table.weight != 0) for table in actual), "tables were re-zeroed"
    for before, after in zip(expected, actual, strict=True):
        assert torch.equal(before, after.weight)


def test_encoder_weights_survive_into_a_downstream_head(
    tiny_config: ModernLayoutLmConfig,
    batch: dict[str, torch.Tensor],
    tmp_path,
) -> None:
    """The backbone must carry over too, not only the layout tables."""
    pretrained = ModernLayoutLmForMaskedLM(tiny_config)
    pretrained.eval()
    with torch.no_grad():
        before = pretrained.model(**batch).last_hidden_state
    pretrained.save_pretrained(tmp_path)

    downstream = ModernLayoutLmForSequenceClassification.from_pretrained(tmp_path, num_labels=3)
    downstream.eval()
    with torch.no_grad():
        after = downstream.model(**batch).last_hidden_state

    assert torch.equal(before, after)


def test_a_fresh_graft_still_zeroes_its_tables(tiny_config: ModernLayoutLmConfig, tmp_path) -> None:
    """
    The other direction: keys genuinely absent from a checkpoint must still be zeroed

    Saving a plain ModernBERT state dict and loading it as ModernLayoutLM is the grafting case,
    and it must produce zeros rather than whatever nn.Embedding defaults to.
    """
    source = ModernLayoutLmForMaskedLM(tiny_config)
    state = {k: v for k, v in source.state_dict().items() if "position_embeddings" not in k}
    source.save_pretrained(tmp_path, state_dict=state)

    grafted = ModernLayoutLmForMaskedLM.from_pretrained(tmp_path)
    assert all(torch.all(table.weight == 0) for table in grafted.model.embeddings.layout_tables())
