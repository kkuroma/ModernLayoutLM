"""
SROIE: receipt information extraction, ICDAR 2019 task 3

Labels arrive as bare integers whose mapping is recovered by reading the rows, and 0 is COMPANY
Fields carry no B and I, so BIO is reconstructed from contiguous runs
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

from datasets import Dataset, load_dataset

from ..schema import COORDINATE_SCALE, DatasetMeta, Document, bio_from_runs, clamp_box

REPO = "Ssunbell/SROIE_layoutlmv2_sequence"
SPLITS = {"train": "train", "validation": "val", "test": "test"}

# Ordered by the source's label ids, recovered by inspecting the rows
SOURCE_FIELDS = ["COMPANY", "DATE", "ADDRESS", "TOTAL", "O"]
OUTSIDE_INDEX = SOURCE_FIELDS.index("O")

LABEL_NAMES = bio_from_runs([], SOURCE_FIELDS, OUTSIDE_INDEX)[1]


def describe() -> DatasetMeta:
    """
    Returns the label vocabulary and task for SROIE

    Returns:
        Metadata with the 9 BIO labels over COMPANY, DATE, ADDRESS and TOTAL.
    """
    return DatasetMeta(
        source="sroie",
        task="token",
        label_names=LABEL_NAMES,
        notes=(
            "Source labels are unnamed integers where 0 is COMPANY and 4 is outside, recovered "
            "by inspection. BIO is reconstructed from contiguous runs, so two adjacent entities "
            "of the same field merge. Numbers are therefore not strictly comparable to published "
            "SROIE results, which use the original annotations."
        ),
    )


def load(split: str, cache_dir: str | None = None) -> Iterator[Document]:
    """
    Yields SROIE receipts in the canonical schema

    Args:
        split: One of the keys of `SPLITS`.
        cache_dir: Hugging Face cache directory, or None for the default.

    Yields:
        One Document per receipt.

    Raises:
        KeyError: If `split` is not a known split.
        ValueError: If the source boxes are not on the expected 0-1000 scale.
    """
    dataset = cast(Dataset, load_dataset(REPO, split=SPLITS[split], cache_dir=cache_dir))
    # The image column is a raw nested uint8 array, so it is dropped before iterating
    columns = [
        name
        for name in ("words", "labels", "boxes", "actual_bboxes")
        if name in dataset.column_names
    ]
    dataset = dataset.select_columns(columns)

    for index, raw in enumerate(dataset):
        row = cast(dict[str, Any], raw)
        boxes = [clamp_box(box, COORDINATE_SCALE) for box in row["boxes"]]
        for box in row["boxes"]:
            if max(box) > COORDINATE_SCALE:
                raise ValueError(
                    f"sroie {index}: box {box} exceeds {COORDINATE_SCALE}; the mirror is "
                    f"expected to store 0-1000 boxes in `boxes` and pixels in `actual_bboxes`."
                )
        tags, _ = bio_from_runs(list(row["labels"]), SOURCE_FIELDS, OUTSIDE_INDEX)
        width, height = _page_size(row.get("actual_bboxes"))
        yield Document(
            doc_id=f"sroie-{split}-{index}",
            source="sroie",
            words=list(row["words"]),
            bboxes=boxes,
            page_width=width,
            page_height=height,
            word_labels=tags,
        )


def _page_size(actual_bboxes: list[list[int]] | None) -> tuple[int, int]:
    """
    Recovers an approximate page size from the pixel boxes

    The mirror does not record page dimensions usably, so the furthest pixel box corner stands in.
    Only rendering and debugging use these numbers; the normalized boxes carry the real geometry.

    Args:
        actual_bboxes: Pixel boxes for the page, or None.

    Returns:
        A (width, height) pair, falling back to zeros when unavailable.
    """
    if not actual_bboxes:
        return 0, 0
    return (
        max(box[2] for box in actual_bboxes),
        max(box[3] for box in actual_bboxes),
    )
