"""
Tests for the model: the graft, its heads, the processor, and the pretrain to fine-tune handoff

A model trained on boxes shifted by one word still trains and still reports a falling loss
_init_weights must zero the layout tables only for keys a checkpoint lacks
"""

from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer, ModernBertConfig, ModernBertModel

from src.model import (
    ModernLayoutLmConfig,
    ModernLayoutLmForMaskedLM,
    ModernLayoutLmForSequenceClassification,
    ModernLayoutLmForTokenClassification,
    ModernLayoutLmModel,
    ModernLayoutLmProcessor,
)
from src.model.modeling_modernlayoutlm import LayoutEmbedding
from src.model.processing_modernlayoutlm import CLS_BOX, IGNORE_INDEX, PAD_BOX, SEP_BOX

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

    One state dict loads into both models, so any difference comes from the architecture.
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

    This is deliberately not the same as all-zero boxes, which put every word at the origin.
    That is a degenerate but real layout, and it still adds six embedding lookups.
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

    `Trainer._remove_unused_columns` keeps only the dataset columns this signature names.
    Behind **kwargs the boxes would be dropped, and the model would train as plain ModernBERT.
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

    A freshly grafted ModernLayoutLM must reproduce ModernBERT exactly.
    It stays in the default run, as safetensors memory-maps the weights.
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


WORDS = ["Total", "unbelievability", "42.00"]
BOXES = [[10, 20, 60, 30], [70, 20, 200, 30], [210, 20, 260, 30]]
LABELS = [1, 2, 0]


@pytest.fixture(scope="module")
def processor() -> ModernLayoutLmProcessor:
    """A processor over the pulled ModernBERT tokenizer."""
    backbone = Path("models/backbones/ModernBERT-base")
    if not (backbone / "tokenizer.json").is_file():
        pytest.skip(f"backbone not pulled to {backbone}")
    return ModernLayoutLmProcessor(tokenizer=AutoTokenizer.from_pretrained(backbone))


def test_every_subword_inherits_its_word_box(processor: ModernLayoutLmProcessor) -> None:
    """
    A word split into several subwords must give every piece the same box

    "unbelievability" tokenizes into several pieces, so an off-by-one expansion fails here.
    """
    encoding = processor(words=WORDS, boxes=BOXES, max_length=32, padding=False)
    word_ids = encoding.word_ids(0)
    boxes = encoding["bbox"][0]

    for position, word_id in enumerate(word_ids):
        if word_id is not None:
            assert boxes[position] == BOXES[word_id], f"position {position} has the wrong box"

    multi_subword = [b for b, w in zip(boxes, word_ids, strict=True) if w == 1]
    assert len(multi_subword) > 1, "the test word should split into several subwords"
    assert all(box == BOXES[1] for box in multi_subword)


def test_special_tokens_get_sentinel_boxes(processor: ModernLayoutLmProcessor) -> None:
    """CLS anchors the origin, SEP the far corner, and padding sits at the origin."""
    encoding = processor(words=WORDS, boxes=BOXES, max_length=32, padding="max_length")
    word_ids = encoding.word_ids(0)
    boxes = encoding["bbox"][0]

    assert boxes[0] == list(CLS_BOX)
    last_word = max(i for i, w in enumerate(word_ids) if w is not None)
    assert boxes[last_word + 1] == list(SEP_BOX)
    assert boxes[last_word + 2] == list(PAD_BOX)
    assert boxes[-1] == list(PAD_BOX)


def test_first_subword_only_labelling(processor: ModernLayoutLmProcessor) -> None:
    """By default only a word's first subword carries its label."""
    encoding = processor(words=WORDS, boxes=BOXES, word_labels=LABELS, max_length=32, padding=False)
    word_ids = encoding.word_ids(0)
    labels = encoding["labels"][0]

    seen: set[int] = set()
    for word_id, label in zip(word_ids, labels, strict=True):
        if word_id is None:
            assert label == IGNORE_INDEX
        elif word_id not in seen:
            assert label == LABELS[word_id]
            seen.add(word_id)
        else:
            assert label == IGNORE_INDEX
    assert seen == {0, 1, 2}


def test_label_all_subwords(processor: ModernLayoutLmProcessor) -> None:
    """With the flag set, every subword of a word carries that word's label."""
    labeling = ModernLayoutLmProcessor(
        tokenizer=processor.tokenizer,
        label_all_subwords=True,
    )
    encoding = labeling(words=WORDS, boxes=BOXES, word_labels=LABELS, max_length=32, padding=False)
    for word_id, label in zip(encoding.word_ids(0), encoding["labels"][0], strict=True):
        assert label == (IGNORE_INDEX if word_id is None else LABELS[word_id])


def test_batched_documents_stay_separate(processor: ModernLayoutLmProcessor) -> None:
    """Two documents in one call must not borrow each other's boxes."""
    other_words = ["Date", "2026"]
    other_boxes = [[5, 5, 40, 15], [50, 5, 90, 15]]
    encoding = processor(
        words=[WORDS, other_words],
        boxes=[BOXES, other_boxes],
        max_length=32,
        padding="max_length",
    )
    assert len(encoding["bbox"]) == 2
    for chunk, source in enumerate([BOXES, other_boxes]):
        for position, word_id in enumerate(encoding.word_ids(chunk)):
            if word_id is not None:
                assert encoding["bbox"][chunk][position] == source[word_id]


def test_overflow_chunks_carry_their_own_boxes(processor: ModernLayoutLmProcessor) -> None:
    """
    A page longer than max_length becomes several chunks, each with boxes already aligned

    Without per-chunk alignment the second chunk would silently reuse the first chunk's boxes.
    """
    words = [f"word{i}" for i in range(60)]
    boxes = [[i, i, i + 5, i + 5] for i in range(60)]
    encoding = processor(
        words=words,
        boxes=boxes,
        max_length=16,
        padding="max_length",
        return_overflowing_tokens=True,
    )
    assert len(encoding["input_ids"]) > 1
    for chunk in range(len(encoding["input_ids"])):
        for position, word_id in enumerate(encoding.word_ids(chunk)):
            if word_id is not None:
                assert encoding["bbox"][chunk][position] == boxes[word_id]


def test_pt_tensors_have_matching_shapes(processor: ModernLayoutLmProcessor) -> None:
    """The bbox tensor must line up with input_ids so the model can add them."""
    encoding = processor(
        words=WORDS,
        boxes=BOXES,
        word_labels=LABELS,
        max_length=32,
        return_tensors="pt",
    )
    assert encoding["input_ids"].shape == (1, 32)
    assert encoding["bbox"].shape == (1, 32, 4)
    assert encoding["labels"].shape == (1, 32)


def test_word_and_box_count_mismatch_raises(processor: ModernLayoutLmProcessor) -> None:
    """A missing box is a caller bug that must fail immediately, not train on garbage."""
    with pytest.raises(ValueError, match="words but"):
        processor(words=WORDS, boxes=BOXES[:2], max_length=32)


def test_unnormalized_boxes_raise(processor: ModernLayoutLmProcessor) -> None:
    """Raw pixel coordinates are the most likely mistake, so they must be rejected."""
    with pytest.raises(ValueError, match=r"outside \[0, 1000\]"):
        processor(words=WORDS, boxes=[[0, 0, 2480, 3500]] * 3, max_length=32)


def test_label_count_mismatch_raises(processor: ModernLayoutLmProcessor) -> None:
    """Label misalignment must be caught at the boundary."""
    with pytest.raises(ValueError, match="labels"):
        processor(words=WORDS, boxes=BOXES, word_labels=[1], max_length=32)


def test_layout_tables_survive_into_a_downstream_head(
    tiny_config: ModernLayoutLmConfig,
    tmp_path,
) -> None:
    """
    Tables trained during MLM must arrive intact in a token classification head

    Saved from `ForMaskedLM` and loaded into `ForTokenClassification`, the stage 1 to 2 handoff.
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


def test_sparse_prediction_keeps_the_loss_and_stays_out_of_the_saved_config(
    tiny_config: ModernLayoutLmConfig, batch: dict[str, torch.Tensor], tmp_path
) -> None:
    """
    Pretraining scores masked tokens only, which must neither change the loss nor reach an export

    The flag is set on the instance, so a saved config still promises dense logits to its users.
    """
    torch.manual_seed(0)
    model = ModernLayoutLmForMaskedLM(tiny_config).eval()
    labels = batch["input_ids"].clone()
    labels[:, ::2] = -100
    dense = model(**batch, labels=labels).loss

    model.sparse_prediction = True
    sparse = model(**batch, labels=labels)
    assert torch.allclose(sparse.loss, dense, atol=1e-6)
    assert sparse.logits.shape[0] == int((labels != -100).sum())

    model.save_pretrained(tmp_path)
    assert ModernLayoutLmForMaskedLM.from_pretrained(tmp_path).sparse_prediction is False
