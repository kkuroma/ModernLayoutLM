"""
Tests for the per-dataset adapters

Each adapter is parsed against a hand-built record, so they need no network
The point is to pin the awkward part of each source rather than its happy path
"""

import pytest

from src.data.sources import GROUPS, SOURCES, cord, resolve_names, rvlcdip, sroie


def test_every_source_exposes_the_interface() -> None:
    """`build_source` calls exactly these two functions, so every adapter must have them."""
    for name, source in SOURCES.items():
        assert callable(source.describe), name
        assert callable(source.load), name
        assert source.SPLITS, name


def test_every_source_declares_labels_matching_its_task() -> None:
    """
    A scorable task needs a label vocabulary, and a pretraining corpus must not claim one

    Pretraining corpora are exempt because MVLM predicts tokens, not labels. Asserting both
    directions catches a corpus mislabelled as a task, which would otherwise fail much later
    inside the Trainer.
    """
    for name, source in SOURCES.items():
        meta = source.describe()
        assert meta.source == name
        assert meta.task in {"token", "sequence", "pretrain"}
        if meta.task == "pretrain":
            assert not meta.label_names, name
        else:
            assert meta.label_names, name


def test_groups_only_name_registered_sources() -> None:
    """A typo in a group would fail at build time, so catch it here instead."""
    for group, names in GROUPS.items():
        for name in names:
            assert name in SOURCES, f"group {group} names unknown source {name}"


def test_resolve_names_prefers_explicit_names() -> None:
    """Explicit names win over the group, and an unknown name raises."""
    assert resolve_names(["funsd"], "downstream") == ["funsd"]
    assert resolve_names(None, "token") == GROUPS["token"]
    assert resolve_names(None, None) == GROUPS["downstream"]
    with pytest.raises(KeyError, match="Unknown source"):
        resolve_names(["nope"], None)
    with pytest.raises(KeyError, match="Unknown group"):
        resolve_names(None, "nope")


def test_cord_labels_bio_per_line() -> None:
    """
    The first word of a CORD line starts an entity and the rest continue it

    Field names come from the line, not the word, so this is the only place the B and I
    distinction gets made for CORD.
    """
    lines = [
        {
            "category": "menu.nm",
            "words": [
                {"text": "Nasi", "quad": _quad(10, 20, 50, 30)},
                {"text": "Campur", "quad": _quad(60, 20, 120, 30)},
            ],
        }
    ]
    document = cord._build("cord-test-0", lines, width=200, height=100)
    assert document is not None
    names = cord.LABEL_NAMES
    assert document.word_labels is not None
    assert [names[label] for label in document.word_labels] == ["B-menu.nm", "I-menu.nm"]
    assert document.words == ["Nasi", "Campur"]
    # x spans 10..50 of a 200px page, so 50..250 on the 1000 grid
    assert document.bboxes[0] == [50, 200, 250, 300]


def test_cord_rejects_unknown_category() -> None:
    """
    An unlisted category must raise rather than silently becoming O

    This is how the released data's dotted `menu.sub.nm` was caught; the published list writes it
    with an underscore, and falling through to O would have erased the field.
    """
    lines = [{"category": "menu.invented", "words": [{"text": "x", "quad": _quad(0, 0, 1, 1)}]}]
    with pytest.raises(ValueError, match="unknown CORD category"):
        cord._build("cord-test-1", lines, width=10, height=10)


def test_cord_skips_blank_words() -> None:
    """Whitespace-only OCR text contributes no word, and no box either."""
    lines = [
        {
            "category": "menu.nm",
            "words": [
                {"text": "  ", "quad": _quad(0, 0, 5, 5)},
                {"text": "Nasi", "quad": _quad(10, 10, 20, 20)},
            ],
        }
    ]
    document = cord._build("cord-test-2", lines, width=100, height=100)
    assert document is not None
    assert document.words == ["Nasi"]


def test_cord_returns_none_for_a_wordless_receipt() -> None:
    """A receipt whose lines hold nothing usable is dropped, not written as an empty row."""
    assert cord._build("cord-test-3", [], width=100, height=100) is None


def test_sroie_label_scheme_is_the_recovered_one() -> None:
    """
    SROIE's source integers are pinned to the mapping recovered by inspection

    0 is COMPANY and 4 is outside, which is the reverse of the usual convention. If this ever
    silently flips, every SROIE number becomes meaningless, so the mapping is asserted.
    """
    assert sroie.SOURCE_FIELDS == ["COMPANY", "DATE", "ADDRESS", "TOTAL", "O"]
    assert sroie.OUTSIDE_INDEX == 4
    assert sroie.LABEL_NAMES[0] == "O"
    assert sroie.LABEL_NAMES[1] == "B-COMPANY"
    assert len(sroie.LABEL_NAMES) == 9


def test_rvlcdip_splits_segments_into_words() -> None:
    """
    A segment holding several words becomes several words with divided boxes

    The source calls these "word-level annotations" but they are segments, so without splitting
    every word of a phrase would share one wide box.
    """
    entry = {
        "images": [
            {"label": "invoice", "image_width": 1000, "image_height": 1000},
        ],
        "word-level annotations": [
            {
                "ocred_text": ["Plasminogen Activator Activity", "Total"],
                "ocred_boxes": [
                    {"x0": 0, "y0": 100, "x1": 600, "y1": 130},
                    {"x0": 0, "y0": 200, "x1": 100, "y1": 230},
                ],
            }
        ],
    }
    document = rvlcdip._build("rvlcdip-test-0", entry)
    assert document is not None
    assert document.words == ["Plasminogen", "Activator", "Activity", "Total"]
    assert document.doc_label == rvlcdip.CLASS_NAMES.index("invoice")
    assert document.word_labels is None
    # The three split boxes must tile the segment left to right without going outside it
    assert document.bboxes[0][0] == 0
    assert document.bboxes[2][2] == 600
    assert document.bboxes[0][2] <= document.bboxes[1][0]


def test_rvlcdip_drops_text_starved_pages() -> None:
    """A page with almost no OCR carries no signal and only wastes sequence budget."""
    entry = {
        "images": [{"label": "invoice", "image_width": 100, "image_height": 100}],
        "word-level annotations": [
            {"ocred_text": ["a"], "ocred_boxes": [{"x0": 0, "y0": 0, "x1": 5, "y1": 5}]}
        ],
    }
    assert rvlcdip._build("rvlcdip-test-1", entry) is None


def test_rvlcdip_raises_on_unknown_class() -> None:
    """
    An unrecognized class name must raise, not be skipped

    Skipping it is what silently discarded a quarter of the corpus: the mirror spells four
    classes differently from the paper, so `report`, `publication`, `file_folder` and
    `news_article` all fell through and their pages vanished without a word of complaint.
    """
    entry = {
        "images": [{"label": "not a real class", "image_width": 100, "image_height": 100}],
        "word-level annotations": [
            {
                "ocred_text": ["a b c d e"],
                "ocred_boxes": [{"x0": 0, "y0": 0, "x1": 50, "y1": 5}],
            }
        ],
    }
    with pytest.raises(ValueError, match="unknown RVL-CDIP class"):
        rvlcdip._build("rvlcdip-test-2", entry)


def test_rvlcdip_has_sixteen_classes() -> None:
    """RVL-CDIP is a 16-way problem; a short list would silently drop pages."""
    assert len(rvlcdip.CLASS_NAMES) == 16
    assert len(set(rvlcdip.CLASS_NAMES)) == 16
    assert len(rvlcdip.SOURCE_LABELS) == 16
    assert len(set(rvlcdip.SOURCE_LABELS)) == 16


def test_rvlcdip_source_labels_align_with_readable_names() -> None:
    """
    The mirror's spellings and the readable names must stay index for index

    A shift between the two lists would relabel every page while every count still looked right.
    """
    assert len(rvlcdip.SOURCE_LABELS) == len(rvlcdip.CLASS_NAMES)
    for source, readable in zip(rvlcdip.SOURCE_LABELS, rvlcdip.CLASS_NAMES, strict=True):
        assert source.replace("_", " ") in readable, f"{source} does not match {readable}"


def _quad(x0: int, y0: int, x1: int, y1: int) -> dict[str, int]:
    """Builds an axis-aligned CORD quad from a rectangle."""
    return {
        "x1": x0,
        "y1": y0,
        "x2": x1,
        "y2": y0,
        "x3": x1,
        "y3": y1,
        "x4": x0,
        "y4": y1,
    }
