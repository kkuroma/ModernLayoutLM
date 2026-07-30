"""
Tests for the coordinate primitives

These are the functions every adapter leans on, so a bug here corrupts every dataset at once
rather than just one. They are also pure and cheap, which makes them worth testing thoroughly.
"""

import pytest

from src.data.normalize import (
    COORDINATE_SCALE,
    bio_from_runs,
    clamp_box,
    normalize_box,
    quad_to_xyxy,
    split_segment,
)


def test_clamp_box_bounds_coordinates() -> None:
    """Coordinates outside the scale are pulled to the edge, not rejected."""
    assert clamp_box([-20, -5, 2000, 1500]) == [0, 0, COORDINATE_SCALE, COORDINATE_SCALE]


def test_clamp_box_reorders_inverted_corners() -> None:
    """
    A box with x1 < x0 is repaired rather than dropped

    Rotated and right-to-left text produces these, and dropping the box would lose the word's
    text along with its position.
    """
    assert clamp_box([900, 800, 100, 200]) == [100, 200, 900, 800]


def test_clamp_box_rejects_wrong_arity() -> None:
    """A box that is not four numbers is a bug in the adapter, not bad data."""
    with pytest.raises(ValueError, match="4 coordinates"):
        clamp_box([1, 2, 3])


def test_normalize_box_scales_pixels() -> None:
    """A box spanning half the page lands at half the scale."""
    assert normalize_box([0, 0, 100, 50], 200, 100) == [0, 0, 500, 500]


def test_normalize_box_covers_full_page() -> None:
    """A full-page box maps exactly onto the scale, with no off-by-one."""
    assert normalize_box([0, 0, 762, 1000], 762, 1000) == [0, 0, 1000, 1000]


def test_normalize_box_rejects_degenerate_page() -> None:
    """A zero-width page would divide by zero, so it must be caught."""
    with pytest.raises(ValueError, match="positive"):
        normalize_box([0, 0, 10, 10], 0, 100)


def test_quad_to_xyxy_takes_the_enclosing_box() -> None:
    """A rotated quad collapses to the axis-aligned rectangle that contains it."""
    quad = {"x1": 10, "y1": 20, "x2": 50, "y2": 15, "x3": 55, "y3": 40, "x4": 12, "y4": 45}
    assert quad_to_xyxy(quad) == [10, 15, 55, 45]


def test_quad_to_xyxy_reports_missing_corners() -> None:
    """A truncated quad names what is missing rather than raising a bare KeyError."""
    with pytest.raises(KeyError, match="missing"):
        quad_to_xyxy({"x1": 0, "y1": 0})


def test_split_segment_passes_single_words_through() -> None:
    """One word keeps its box exactly."""
    assert split_segment("Total", [10, 20, 60, 30]) == [("Total", [10, 20, 60, 30])]


def test_split_segment_divides_by_character_count() -> None:
    """
    A multi-word segment is divided along x in proportion to word length

    Word boxes must tile the segment left to right without overlapping, and must stay inside the
    original box. The vertical extent is shared, since the segment is one line.
    """
    pairs = split_segment("ab cd", [0, 10, 100, 20])
    assert [word for word, _ in pairs] == ["ab", "cd"]
    boxes = [box for _, box in pairs]
    assert boxes[0][0] == 0
    assert boxes[-1][2] == 100
    assert boxes[0][2] <= boxes[1][0]
    assert all(box[1] == 10 and box[3] == 20 for box in boxes)


def test_split_segment_handles_no_words() -> None:
    """Whitespace-only OCR output contributes nothing."""
    assert split_segment("   ", [0, 0, 10, 10]) == []


def test_split_segment_survives_zero_width_box() -> None:
    """A degenerate box still yields every word, just stacked at the same place."""
    pairs = split_segment("a b c", [50, 0, 50, 10])
    assert len(pairs) == 3


def test_bio_from_runs_marks_entity_starts() -> None:
    """A contiguous run of one field becomes one entity: B then I."""
    names = ["COMPANY", "DATE", "O"]
    tags, bio_names = bio_from_runs([0, 0, 2, 1, 1, 1], names, outside=2)
    readable = [bio_names[tag] for tag in tags]
    assert readable == ["B-COMPANY", "I-COMPANY", "O", "B-DATE", "I-DATE", "I-DATE"]


def test_bio_from_runs_restarts_after_outside() -> None:
    """A field interrupted by O starts a new entity rather than continuing the old one."""
    names = ["TOTAL", "O"]
    tags, bio_names = bio_from_runs([0, 1, 0], names, outside=1)
    assert [bio_names[tag] for tag in tags] == ["B-TOTAL", "O", "B-TOTAL"]


def test_bio_from_runs_places_outside_first() -> None:
    """O must be label 0 in the output vocabulary regardless of where it sat in the source."""
    _, bio_names = bio_from_runs([], ["COMPANY", "DATE", "ADDRESS", "TOTAL", "O"], outside=4)
    assert bio_names[0] == "O"
    assert bio_names == [
        "O",
        "B-COMPANY",
        "I-COMPANY",
        "B-DATE",
        "I-DATE",
        "B-ADDRESS",
        "I-ADDRESS",
        "B-TOTAL",
        "I-TOTAL",
    ]


def test_bio_from_runs_rejects_unknown_label() -> None:
    """A label id with no name means the source scheme was misread."""
    with pytest.raises(ValueError, match="outside the"):
        bio_from_runs([5], ["A", "O"], outside=1)


def test_bio_from_runs_rejects_bad_outside_index() -> None:
    """An outside index off the end of the names is a caller bug."""
    with pytest.raises(ValueError, match="not a valid index"):
        bio_from_runs([0], ["A", "O"], outside=9)
