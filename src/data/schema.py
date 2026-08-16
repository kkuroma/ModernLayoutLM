"""
The canonical document schema

Every source is reduced to this one shape before anything downstream sees it
The label vocabulary lives in a sidecar so training never infers it from whichever rows are present
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq

from .normalize import COORDINATE_SCALE

META_FILENAME = "dataset_meta.json"

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

Task = Literal["token", "sequence", "pretrain"]


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

        Called on every document at build time. A silent violation here surfaces much later as a
        model that trains without complaint and scores nonsense, so it is worth the cost.

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
        label_names: The label vocabulary, indexed by label id. Empty for pretraining corpora.
        splits: Row count per split.
        notes: Anything a reader needs to know to interpret the numbers, such as a label scheme
            that was reconstructed rather than given.
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
        documents: The documents to write. Must be non-empty.
        path: Destination parquet file. Parent directories are created.

    Returns:
        The number of rows written.

    Raises:
        ValueError: If `documents` is empty.
    """
    if not documents:
        raise ValueError(f"Refusing to write an empty parquet file to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([asdict(document) for document in documents], DOCUMENT_SCHEMA)
    pq.write_table(table, path, compression="zstd")
    return table.num_rows
