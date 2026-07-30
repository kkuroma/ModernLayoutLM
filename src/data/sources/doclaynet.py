"""
DocLayNet v1.2: 80k human-annotated pages across six document domains

Used for layout-aware MLM, not for its layout labels. The value here is domain diversity: finance,
law, manuals, patents, tenders and science, where PubTables-1M is PubMed only.

Two things make this adapter unusual.

The dataset is 40GB because every row bundles the page image and the source PDF alongside the text.
We need neither. Reading the remote parquet through pyarrow with column projection skips those
column chunks entirely, and `image` plus `pdf` are 99 percent of the bytes, so the transfer drops
from 40GB to a few hundred megabytes.

And `pdf_cells[].bbox` is COCO `(x, y, width, height)` in a 1025-square space, not `(x0, y0, x1,
y1)`. Reading it as the latter violates the box invariants on 20568 of 20569 sampled cells, which
is how the format was pinned down.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from ..normalize import normalize_box, split_segment
from ..schema import DatasetMeta, Document

REPO = "docling-project/DocLayNet-v1.2"
GLOB = f"datasets/{REPO}/**/*.parquet"

# Parquet shards are named by split, so the split is read off the filename.
SPLITS = {"train": "train-", "validation": "validation-", "test": "test-"}

# Only these two columns are fetched. The rest, image and pdf above all, are never transferred.
COLUMNS = ["pdf_cells", "metadata"]

# Rows per read. Larger batches mean fewer round trips against the remote file.
BATCH_SIZE = 64

# A page with almost no extractable text is not worth a training sequence.
MIN_WORDS = 8


def describe() -> DatasetMeta:
    """
    Returns the task for DocLayNet as used here

    Returns:
        Metadata marking this a pretraining corpus with no label vocabulary.
    """
    return DatasetMeta(
        source="doclaynet",
        task="pretrain",
        label_names=[],
        notes=(
            "Text and boxes come from pdf_cells, which is line level, so cells are split into "
            "words with boxes divided along x. Boxes are COCO (x, y, w, h) in a 1025-square "
            "space, converted to xyxy and normalized. Images and PDFs are skipped by parquet "
            "column projection, which is what keeps this a few hundred MB instead of 40GB."
        ),
    )


def load(split: str, cache_dir: str | None = None) -> Iterator[Document]:
    """
    Yields DocLayNet pages in the canonical schema

    Args:
        split: One of the keys of `SPLITS`.
        cache_dir: Unused. Reads stream from the hub rather than through the dataset cache,
            because caching would defeat the column projection that makes this affordable.

    Yields:
        One Document per page with enough extractable text.

    Raises:
        KeyError: If `split` is not a known split.
        FileNotFoundError: If the repository exposes no parquet for this split.
    """
    prefix = SPLITS[split]
    filesystem = HfFileSystem()
    paths = sorted(
        path for path in filesystem.glob(GLOB) if path.rsplit("/", 1)[-1].startswith(prefix)
    )
    if not paths:
        raise FileNotFoundError(f"No {prefix}*.parquet under {REPO}")

    for path in paths:
        with filesystem.open(path) as handle:
            parquet = pq.ParquetFile(handle)
            for batch in parquet.iter_batches(batch_size=BATCH_SIZE, columns=COLUMNS):
                for index, row in enumerate(batch.to_pylist()):
                    document = _build(f"doclaynet-{split}-{_stem(path)}-{index}", row)
                    if document is not None:
                        yield document


def _build(doc_id: str, row: dict[str, Any]) -> Document | None:
    """
    Assembles one page from its pdf_cells

    Cells are flattened out of their annotation groups, deduplicated, and sorted into reading
    order, because a cell can appear under more than one overlapping region.

    Args:
        doc_id: Identifier for the resulting document.
        row: One projected parquet row, holding `pdf_cells` and `metadata`.

    Returns:
        The Document, or None when the page has too little text.
    """
    meta = row["metadata"] or {}
    width = int(meta.get("coco_width") or 0)
    height = int(meta.get("coco_height") or 0)
    if width <= 0 or height <= 0:
        return None

    cells = _unique_cells(row["pdf_cells"] or [])
    words: list[str] = []
    boxes: list[list[int]] = []
    for box, text in cells:
        for word, split_box in split_segment(text, box):
            words.append(word)
            boxes.append(normalize_box(split_box, width, height))

    if len(words) < MIN_WORDS:
        return None
    return Document(
        doc_id=doc_id,
        source="doclaynet",
        words=words,
        bboxes=boxes,
        page_width=width,
        page_height=height,
    )


def _unique_cells(groups: list[Any]) -> list[tuple[list[float], str]]:
    """
    Flattens, deduplicates and orders the cells of one page

    Args:
        groups: The nested `pdf_cells` value, a list of annotation groups of cells.

    Returns:
        (box, text) pairs in reading order, with each box converted from COCO xywh to xyxy.
    """
    seen: set[tuple[float, float, float, float, str]] = set()
    cells: list[tuple[list[float], str]] = []
    for group in groups:
        for cell in group or []:
            text = (cell.get("text") or "").strip()
            raw = cell.get("bbox")
            if not text or not raw or len(raw) != 4:
                continue
            x, y, box_width, box_height = (float(value) for value in raw)
            key = (x, y, box_width, box_height, text)
            if key in seen:
                continue
            seen.add(key)
            cells.append(([x, y, x + box_width, y + box_height], text))

    # Reading order top to bottom then left to right, which is the order the model's 1D positions
    # will imply. Without it, cells arrive grouped by region and jump around the page.
    cells.sort(key=lambda pair: (round(pair[0][1], 1), pair[0][0]))
    return cells


def _stem(path: str) -> str:
    """
    Extracts a short shard identifier from a parquet path

    Args:
        path: The full remote path.

    Returns:
        The filename without its extension.
    """
    return path.rsplit("/", 1)[-1].removesuffix(".parquet")
