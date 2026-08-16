"""
Tests for sequence packing

Packing may change where a token sits and never which box belongs to it
A packer that drifts boxes by one position would speed training up and train on the wrong layout
"""

import pytest

from src.train.packing import PAD_BOX, SEPARATOR_BOX, pack_encodings, packing_efficiency

SEP, PAD = 2, 0


def page(start: int, length: int) -> dict:
    """Builds a page whose token ids and box x-coordinates both encode their identity."""
    return {
        "input_ids": list(range(start, start + length)),
        "bbox": [[start + i, 1, start + i + 1, 2] for i in range(length)],
    }


def test_pages_are_packed_into_full_sequences() -> None:
    """Several short pages should share one sequence rather than each padding out to length."""
    packed = list(pack_encodings(iter([page(10, 4), page(50, 4)]), 16, SEP, PAD))
    assert len(packed) == 1
    assert packed[0]["attention_mask"].count(1) == 9  # 4 + separator + 4


def test_every_token_keeps_its_own_box() -> None:
    """
    The box at each position must still belong to the token at that position

    Token ids and box x0 are constructed to be equal, so a single misalignment shows up as an
    inequality rather than needing a hand-checked expectation.
    """
    packed = list(pack_encodings(iter([page(10, 5), page(80, 6)]), 32, SEP, PAD))
    row = packed[0]
    for token, box in zip(row["input_ids"], row["bbox"], strict=True):
        if token == SEP:
            assert box == SEPARATOR_BOX
        elif token == PAD:
            assert box == PAD_BOX
        else:
            assert box[0] == token, f"token {token} carries box {box}"


def test_sequences_are_exactly_max_length() -> None:
    """Ragged output would fail to stack in the collator."""
    packed = list(pack_encodings(iter([page(i * 10, 7) for i in range(9)]), 24, SEP, PAD))
    assert len(packed) > 1
    for row in packed:
        assert len(row["input_ids"]) == 24
        assert len(row["attention_mask"]) == 24
        assert len(row["bbox"]) == 24


def test_an_over_length_page_gets_its_own_sequence() -> None:
    """
    A page longer than the window must not straddle a pack boundary

    Splitting it would place half a page beside an unrelated one, which is worse than truncating.
    """
    packed = list(pack_encodings(iter([page(10, 3), page(500, 40)]), 16, SEP, PAD))
    assert len(packed) == 2
    assert packed[0]["input_ids"][:3] == [10, 11, 12]
    # The long page starts its own sequence rather than continuing the first
    assert packed[1]["input_ids"][0] == 500


def test_the_trailing_partial_sequence_is_emitted() -> None:
    """A page must never be dropped just because it landed last."""
    packed = list(pack_encodings(iter([page(10, 4)]), 32, SEP, PAD))
    assert len(packed) == 1
    assert packed[0]["input_ids"][:4] == [10, 11, 12, 13]
    assert packed[0]["attention_mask"][4:] == [0] * 28


def test_padding_is_masked_out() -> None:
    """Padding must not be attended to, or the model learns from filler."""
    row = next(iter(pack_encodings(iter([page(10, 5)]), 20, SEP, PAD)))
    for index, mask in enumerate(row["attention_mask"]):
        if mask == 0:
            assert row["bbox"][index] == PAD_BOX


def test_mismatched_boxes_raise() -> None:
    """A page whose boxes and tokens disagree is a bug upstream and must not be packed."""
    broken = {"input_ids": [1, 2, 3], "bbox": [[0, 0, 1, 1]]}
    with pytest.raises(ValueError, match="tokens but"):
        list(pack_encodings(iter([broken]), 16, SEP, PAD))


def test_empty_pages_are_skipped() -> None:
    """An empty page contributes nothing and must not emit a stray separator."""
    packed = list(pack_encodings(iter([{"input_ids": [], "bbox": []}, page(10, 3)]), 16, SEP, PAD))
    assert len(packed) == 1
    assert packed[0]["input_ids"][0] == 10


def test_efficiency_is_measured_against_overflow_chunking() -> None:
    """The baseline is chunking, not one padded window per page."""
    assert packing_efficiency([512] * 8, 512) == pytest.approx(1.0)
    assert packing_efficiency([380] * 8, 512) == pytest.approx(512 / 380)
    assert packing_efficiency([380] * 8, 1024) == pytest.approx(1024 / 380)


def test_long_pages_barely_benefit_from_packing() -> None:
    """
    A page that overruns the window is chunked into near-full windows already

    This is the correction to the earlier model, which treated every page as one padded window and
    so claimed 2.7x where the real figure on a long-page corpus is a few percent.
    """
    # 1000 tokens at 512 costs two windows, of which only the second is half empty
    assert packing_efficiency([1000] * 8, 512) == pytest.approx(1024 / 1000)
    # Mixing one short page in recovers only that page's padding
    assert packing_efficiency([1000, 100], 512) == pytest.approx((1024 + 512) / 1100)


def test_efficiency_rejects_an_empty_corpus() -> None:
    """No tokens means no ratio to report, rather than a division by zero."""
    with pytest.raises(ValueError):
        packing_efficiency([], 512)
    with pytest.raises(ValueError):
        packing_efficiency([100], 0)
