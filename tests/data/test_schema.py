"""
Tests for the canonical schema and its validation

A document whose boxes and words disagree still trains and still shows a falling loss
These check that every way of being wrong actually raises
"""

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.data.schema import DatasetMeta, Document, write_documents


def make_document(**overrides) -> Document:
    """Builds a valid two-word document, with fields optionally replaced."""
    fields = {
        "doc_id": "test-0",
        "source": "test",
        "words": ["Total", "42.00"],
        "bboxes": [[10, 20, 60, 30], [70, 20, 120, 30]],
        "page_width": 800,
        "page_height": 1000,
        "word_labels": [1, 2],
    }
    fields.update(overrides)
    return Document(**fields)


def test_valid_document_passes() -> None:
    """The happy path must not raise."""
    make_document().validate()


def test_word_and_box_mismatch_raises() -> None:
    """One box per word is the core invariant."""
    with pytest.raises(ValueError, match="words but"):
        make_document(bboxes=[[10, 20, 60, 30]]).validate()


def test_label_mismatch_raises() -> None:
    """Labels must line up with words too, or the entities shift."""
    with pytest.raises(ValueError, match="labels"):
        make_document(word_labels=[1]).validate()


def test_empty_document_raises() -> None:
    """A page with no words cannot be trained on and must not reach parquet."""
    with pytest.raises(ValueError, match="no words"):
        make_document(words=[], bboxes=[], word_labels=[]).validate()


def test_out_of_scale_box_raises() -> None:
    """Raw pixels left unnormalized are the most likely adapter bug."""
    with pytest.raises(ValueError, match=r"outside \[0, 1000\]"):
        make_document(bboxes=[[0, 0, 2480, 3500], [1, 1, 2, 2]]).validate()


def test_inverted_box_raises() -> None:
    """Validation runs after clamping, so an inverted box here means clamping was skipped."""
    with pytest.raises(ValueError, match="inverted"):
        make_document(bboxes=[[60, 20, 10, 30], [1, 1, 2, 2]]).validate()


def test_malformed_box_raises() -> None:
    """A box that is not four numbers must be caught before pyarrow sees it."""
    with pytest.raises(ValueError, match="4 coordinates"):
        make_document(bboxes=[[10, 20, 60], [1, 1, 2, 2]]).validate()


def test_roundtrip_through_parquet(tmp_path: Path) -> None:
    """Documents must survive the schema unchanged, boxes included."""
    documents = [make_document(), make_document(doc_id="test-1", doc_label=3)]
    path = tmp_path / "shard.parquet"
    assert write_documents(documents, path) == 2

    table = pq.read_table(path)
    rows = table.to_pylist()
    assert [row["doc_id"] for row in rows] == ["test-0", "test-1"]
    assert rows[0]["bboxes"] == [[10, 20, 60, 30], [70, 20, 120, 30]]
    assert rows[0]["doc_label"] is None
    assert rows[1]["doc_label"] == 3


def test_writing_nothing_raises(tmp_path: Path) -> None:
    """
    An empty parquet file is worse than no file

    It reads back as a valid but empty split, so a failed build would look like a finished one.
    """
    with pytest.raises(ValueError, match="empty parquet"):
        write_documents([], tmp_path / "shard.parquet")


def test_meta_roundtrip(tmp_path: Path) -> None:
    """The label sidecar must survive a write and read unchanged."""
    meta = DatasetMeta(
        source="test",
        task="token",
        label_names=["O", "B-X", "I-X"],
        splits={"train": 10},
        notes="a note",
    )
    meta.write(tmp_path)
    assert DatasetMeta.read(tmp_path) == meta


def test_missing_meta_raises(tmp_path: Path) -> None:
    """A dataset without its sidecar has an unknown label vocabulary and must not load."""
    with pytest.raises(FileNotFoundError, match="rebuild"):
        DatasetMeta.read(tmp_path)
