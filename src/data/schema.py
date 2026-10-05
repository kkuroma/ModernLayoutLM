"""
The canonical document schema, and the primitives that bring source coordinates onto its grid

Every source is reduced to this one shape before anything downstream sees it
The label vocabulary lives in a sidecar so training never infers it from whichever rows are present
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

META_FILENAME = "dataset_meta.json"

# Boxes are integers on this canonical scale
COORDINATE_SCALE = 1000

# A four-corner quad as CORD stores it: corners 1 to 4 clockwise from the top left
QUAD_KEYS = ("x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4")

# int16 comfortably holds [0, 1000] and halves the size of the box column against int32
DOCUMENT_SCHEMA = pa.schema(
    [
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("words", pa.list_(pa.string()), nullable=False),
        pa.field("bboxes", pa.list_(pa.list_(pa.int16(), 4)), nullable=False),
        pa.field("page_width", pa.int32(), nullable=False),
        pa.field("page_height", pa.int32(), nullable=False),
        pa.field("word_labels", pa.list_(pa.int16()), nullable=True),
        pa.field("doc_label", pa.int16(), nullable=True),
    ]
)

Task = Literal["token", "pretrain"]


@dataclass
class Document:
    """
    One page of text with a box per word

    Args:
        doc_id: Identifier unique within the source.
        source: The dataset name this came from.
        words: The page's words in reading order as the source gave it.
        bboxes: One box per word, normalized to `[0, COORDINATE_SCALE]`.
        page_width: Original page width in pixels, kept for rendering and debugging.
        page_height: Original page height in pixels.
        word_labels: One label per word for token classification, else None.
        doc_label: One label for the whole page for sequence classification, else None.
    """

    doc_id: str
    source: str
    words: list[str]
    bboxes: list[list[int]]
    page_width: int
    page_height: int
    word_labels: list[int] | None = None
    doc_label: int | None = None

    def validate(self, scale: int = COORDINATE_SCALE) -> None:
        """
        Checks the invariants the rest of the pipeline relies on

        Called on every document at build time.
        A violation missed here surfaces later as a model that trains fine and scores nonsense.

        Args:
            scale: The coordinate scale boxes must lie within.

        Raises:
            ValueError: On any count mismatch, malformed box, or out of range coordinate.
        """
        if len(self.words) != len(self.bboxes):
            raise ValueError(f"{self.doc_id}: {len(self.words)} words but {len(self.bboxes)} boxes")
        if not self.words:
            raise ValueError(f"{self.doc_id}: no words")
        if self.word_labels is not None and len(self.word_labels) != len(self.words):
            raise ValueError(
                f"{self.doc_id}: {len(self.words)} words but {len(self.word_labels)} labels"
            )
        for box in self.bboxes:
            if len(box) != 4:
                raise ValueError(f"{self.doc_id}: box {box} does not have 4 coordinates")
            if any(not 0 <= value <= scale for value in box):
                raise ValueError(f"{self.doc_id}: box {box} is outside [0, {scale}]")
            if box[0] > box[2] or box[1] > box[3]:
                raise ValueError(f"{self.doc_id}: box {box} has inverted corners")


@dataclass
class DatasetMeta:
    """
    What a processed dataset is and what its labels mean

    Args:
        source: The dataset name.
        task: Whether labels are per word, per document, or absent.
        label_names: The label vocabulary indexed by label id, empty for pretraining corpora.
        splits: Row count per split.
        notes: What a reader needs to interpret the numbers, such as a reconstructed label scheme.
    """

    source: str
    task: Task
    label_names: list[str] = field(default_factory=list)
    splits: dict[str, int] = field(default_factory=dict)
    notes: str = ""

    def write(self, directory: Path) -> None:
        """
        Writes the sidecar into a processed dataset directory

        Args:
            directory: The dataset's processed directory.
        """
        directory.mkdir(parents=True, exist_ok=True)
        (directory / META_FILENAME).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def read(cls, directory: Path) -> DatasetMeta:
        """
        Reads the sidecar from a processed dataset directory

        Args:
            directory: The dataset's processed directory.

        Returns:
            The parsed metadata.

        Raises:
            FileNotFoundError: If the sidecar is absent, meaning the build never completed.
        """
        path = directory / META_FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"No {META_FILENAME} in {directory}; rebuild the dataset")
        return cls(**json.loads(path.read_text()))


def write_documents(documents: list[Document], path: Path) -> int:
    """
    Writes documents to one parquet file in the canonical schema

    Args:
        documents: The documents to write, at least one.
        path: Destination parquet file, whose parent directories are created.

    Returns:
        The number of rows written.

    Raises:
        ValueError: If `documents` is empty.
    """
    if not documents:
        raise ValueError(f"Refusing to write an empty parquet file to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([vars(document) for document in documents], DOCUMENT_SCHEMA)
    pq.write_table(table, path, compression="zstd")
    return table.num_rows


def clamp_box(box: Sequence[float], scale: int = COORDINATE_SCALE) -> list[int]:
    """
    Clamps a box into `[0, scale]` and orders its corners

    Sources emit coordinates a pixel outside the page, and `x1 < x0` for rotated text.
    Both are repaired rather than rejected, as dropping the word would lose its text too.

    Args:
        box: Four coordinates as (x0, y0, x1, y1).
        scale: Upper bound for every coordinate.

    Returns:
        A box with `0 <= x0 <= x1 <= scale` and `0 <= y0 <= y1 <= scale`.

    Raises:
        ValueError: If `box` does not hold exactly four coordinates.
    """
    if len(box) != 4:
        raise ValueError(f"Expected 4 coordinates, got {len(box)}")
    x0, y0, x1, y1 = (max(0, min(int(value), scale)) for value in box)
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def clamp_boxes(boxes: np.ndarray, scale: int = COORDINATE_SCALE) -> list[list[int]]:
    """
    Applies `clamp_box` to every row of an (N, 4) array in one vectorized pass

    Args:
        boxes: Coordinates of shape (N, 4) as (x0, y0, x1, y1).
        scale: Upper bound for every coordinate.

    Returns:
        One box per row, identical to what `clamp_box` returns for that row.

    Raises:
        ValueError: If any coordinate is not finite, which `int()` would refuse too.
    """
    if not np.isfinite(boxes).all():
        raise ValueError("Box coordinates must be finite")
    clamped = np.clip(np.trunc(boxes), 0, scale).astype(np.int64)
    low = np.minimum(clamped[:, :2], clamped[:, 2:])
    high = np.maximum(clamped[:, :2], clamped[:, 2:])
    return np.concatenate([low, high], axis=1).tolist()


def normalize_box(
    box: Sequence[float],
    page_width: float,
    page_height: float,
    scale: int = COORDINATE_SCALE,
) -> list[int]:
    """
    Scales a pixel box onto the canonical `[0, scale]` grid

    Args:
        box: Four pixel coordinates as (x0, y0, x1, y1).
        page_width: Page width in the same units as the box.
        page_height: Page height in the same units as the box.
        scale: Target upper bound.

    Returns:
        The clamped, normalized box as integers.

    Raises:
        ValueError: If the page has a non-positive dimension, or the box is malformed.
    """
    if page_width <= 0 or page_height <= 0:
        raise ValueError(f"Page dimensions must be positive, got {page_width}x{page_height}")
    if len(box) != 4:
        raise ValueError(f"Expected 4 coordinates, got {len(box)}")
    scaled = [
        box[0] / page_width * scale,
        box[1] / page_height * scale,
        box[2] / page_width * scale,
        box[3] / page_height * scale,
    ]
    return clamp_box([int(round(value)) for value in scaled], scale)


def quad_to_xyxy(quad: Mapping[str, float]) -> list[float]:
    """
    Collapses a four-corner quad into an axis-aligned box

    CORD stores each word as four corners, which supports rotated text.
    The enclosing axis-aligned rectangle is taken, as the embedding tables are axis-aligned.

    Args:
        quad: Corner coordinates keyed `x1`..`x4` and `y1`..`y4`.

    Returns:
        The enclosing box as (x0, y0, x1, y1), in the quad's own units.

    Raises:
        KeyError: If any of the eight corner keys is missing.
    """
    missing = [key for key in QUAD_KEYS if key not in quad]
    if missing:
        raise KeyError(f"Quad is missing {missing}")
    xs = [quad["x1"], quad["x2"], quad["x3"], quad["x4"]]
    ys = [quad["y1"], quad["y2"], quad["y3"], quad["y4"]]
    return [min(xs), min(ys), max(xs), max(ys)]


def split_segment(text: str, box: Sequence[float]) -> list[tuple[str, list[float]]]:
    """
    Splits a multi-word segment into words, dividing its box along the x axis

    Several sources only box a whole line or phrase.
    Word extents are estimated by character count, assuming roughly monospaced horizontal text.
    That is far better than giving every word of a line the same wide box.

    Args:
        text: The segment's text, possibly several whitespace-separated words.
        box: The segment's box as (x0, y0, x1, y1), in any units.

    Returns:
        One (word, box) pair per word left to right, empty when `text` has no words.
    """
    words = text.split()
    if not words:
        return []
    if len(words) == 1:
        return [(words[0], list(box))]

    x0, y0, x1, y1 = box
    span = x1 - x0
    # Character budget includes one space between each pair of words, matching the source text
    total = sum(len(word) for word in words) + (len(words) - 1)
    if span <= 0 or total <= 0:
        return [(word, list(box)) for word in words]

    pairs: list[tuple[str, list[float]]] = []
    cursor = 0
    for index, word in enumerate(words):
        start = x0 + span * cursor / total
        end = x0 + span * (cursor + len(word)) / total
        pairs.append((word, [start, y0, end, y1]))
        cursor += len(word) + 1 if index < len(words) - 1 else len(word)
    return pairs


def bio_from_runs(
    labels: Sequence[int], names: Sequence[str], outside: int
) -> tuple[list[int], list[str]]:
    """
    Converts flat per-word field labels into BIO tags

    Some sources tag each word with its field but never mark where one entity ends.
    Contiguous runs of one field become one entity, the first word `B-` and the rest `I-`.
    Two adjacent entities of one field merge, the standard cost of recovering BIO afterwards.

    Args:
        labels: One field index per word, indexing into `names`.
        names: Field names, including the outside class.
        outside: The index in `names` meaning "not part of any field".

    Returns:
        The BIO label indices per word, and the BIO label names they index into.

    Raises:
        ValueError: If a label falls outside `names`, or `outside` is not a valid index.
    """
    if not 0 <= outside < len(names):
        raise ValueError(f"outside index {outside} is not a valid index into {len(names)} names")
    fields = [name for index, name in enumerate(names) if index != outside]
    bio_names = ["O"] + [f"{prefix}-{field}" for field in fields for prefix in ("B", "I")]
    bio_index = {name: index for index, name in enumerate(bio_names)}

    tags: list[int] = []
    previous = outside
    for label in labels:
        if not 0 <= label < len(names):
            raise ValueError(f"Label {label} is outside the {len(names)} known names")
        if label == outside:
            tags.append(0)
        else:
            prefix = "I" if label == previous else "B"
            tags.append(bio_index[f"{prefix}-{names[label]}"])
        previous = label
    return tags, bio_names
