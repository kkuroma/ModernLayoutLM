"""
FUNSD: form understanding over scanned forms, with per-word boxes and BIO tags

`nielsr/funsd` mirrors the official release, one box per word, which is what the papers score on
The `funsd-layoutlmv3` mirror is segment level and hands the model every entity boundary for free
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

from datasets import Dataset, load_dataset

from ..schema import COORDINATE_SCALE, DatasetMeta, Document, clamp_box

REPO = "nielsr/funsd"
SPLITS = {"train": "train", "test": "test"}


def describe() -> DatasetMeta:
    """
    Returns the label vocabulary and task for FUNSD

    Returns:
        Metadata with the 7 BIO labels in the source's own order.
    """
    return DatasetMeta(
        source="funsd",
        task="token",
        label_names=[
            "O",
            "B-HEADER",
            "I-HEADER",
            "B-QUESTION",
            "I-QUESTION",
            "B-ANSWER",
            "I-ANSWER",
        ],
        notes="Official word-level release, 1.00 words per box. Boxes arrive on 0-1000, verified.",
    )


def load(split: str, cache_dir: str | None = None) -> Iterator[Document]:
    """
    Yields FUNSD documents in the canonical schema

    Args:
        split: One of the keys of `SPLITS`.
        cache_dir: Hugging Face cache directory, or None for the default.

    Yields:
        One Document per form.

    Raises:
        KeyError: If `split` is not a known split.
        ValueError: If the boxes turn out not to be on the expected 0-1000 scale.
    """
    dataset = cast(Dataset, load_dataset(REPO, split=SPLITS[split], cache_dir=cache_dir))
    for raw in dataset:
        row = cast(dict[str, Any], raw)
        width, height = row["image"].size
        boxes = [clamp_box(box, COORDINATE_SCALE) for box in row["bboxes"]]
        _assert_normalized(row["bboxes"], row["id"], width, height)
        yield Document(
            doc_id=f"funsd-{split}-{row['id']}",
            source="funsd",
            words=list(row["words"]),
            bboxes=boxes,
            page_width=width,
            page_height=height,
            word_labels=list(row["ner_tags"]),
        )


def _assert_normalized(boxes: list[list[int]], doc_id: str, width: int, height: int) -> None:
    """
    Confirms the source's boxes really are on the 0-1000 scale and not in pixels

    The two are easy to confuse, as FUNSD pages happen to be about 1000 pixels tall.
    A box beyond the scale is the giveaway, and a silent misread would shift every coordinate.

    Args:
        boxes: The source boxes for one page.
        doc_id: Identifier, for the error message.
        width: Page width in pixels.
        height: Page height in pixels.

    Raises:
        ValueError: If any coordinate exceeds the coordinate scale.
    """
    for box in boxes:
        if max(box) > COORDINATE_SCALE:
            raise ValueError(
                f"funsd {doc_id}: box {box} exceeds {COORDINATE_SCALE}, so the source is "
                f"probably in pixels for a {width}x{height} page. Renormalize instead."
            )
