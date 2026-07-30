"""
Tests for box and label alignment across subwords

Misalignment here is the quietest possible failure. A model trained on boxes shifted by one word
still trains, still reports a falling loss, and is simply wrong. So these tests check the
correspondence directly rather than just checking shapes.
"""

from pathlib import Path

import pytest
from transformers import AutoTokenizer

from src.model import ModernLayoutLmProcessor
from src.model.processing_modernlayoutlm import CLS_BOX, IGNORE_INDEX, PAD_BOX, SEP_BOX

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

    "unbelievability" tokenizes into several pieces, so this fails loudly if the expansion is
    off by one rather than genuinely keyed on word_ids.
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
    labelling = ModernLayoutLmProcessor(
        tokenizer=processor.tokenizer,
        label_all_subwords=True,
    )
    encoding = labelling(words=WORDS, boxes=BOXES, word_labels=LABELS, max_length=32, padding=False)
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
