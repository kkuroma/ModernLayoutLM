"""
PDFA: born-digital PDF pages read from the JSON in each webdataset shard, never from the PDF itself

Boxes arrive as normalized (x, y, width, height) rather than corners
Low-confidence words are dropped and a deterministic tail of shards is held out
"""

from __future__ import annotations

import json
import tarfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..schema import COORDINATE_SCALE, DatasetMeta, Document, clamp_box

REPO = "pixparse/pdfa-eng-wds"

# Hold out a deterministic tail of shards as no validation split ships
SPLITS = {"train": "train", "validation": "validation", "test": "test"}
VALIDATION_SHARDS = 4
TEST_SHARDS = 4

# OCR confidence below this is usually a misread glyph rather than a word
MIN_SCORE = 0.5

# A page with almost no extractable text earns no training sequence
MIN_WORDS = 8


def describe() -> DatasetMeta:
    """
    Returns the task for PDFA as used here

    Returns:
        Metadata marking this a pretraining corpus with no label vocabulary.
    """
    return DatasetMeta(
        source="pdfa",
        task="pretrain",
        label_names=[],
        notes=(
            "Born-digital PDFs from the SafeDocs CommonCrawl corpus, word level text and boxes, "
            "no OCR. Boxes arrive as normalized (x, y, w, h) in [0, 1] and are converted to "
            "xyxy on the 0-1000 grid. Only the .json members of each webdataset shard are read, "
            "so the bundled PDFs are never decoded. Words below "
            f"score {MIN_SCORE} are dropped."
        ),
    )


def shard_paths(cache_dir: str | None = None) -> list[Path]:
    """
    Finds the downloaded PDFA shards on disk

    Args:
        cache_dir: Hugging Face cache directory, or None for the default.

    Returns:
        Every `.tar` shard found, sorted by name.
    """
    root = Path(cache_dir or "data/raw/hf-cache")
    return sorted(root.glob("**/datasets--pixparse--pdfa-eng-wds/**/*.tar"))


def load(split: str, cache_dir: str | None = None) -> Iterator[Document]:
    """
    Yields PDFA pages in the canonical schema

    Args:
        split: One of the keys of `SPLITS`.
        cache_dir: Hugging Face cache directory, or None for the default.

    Yields:
        One Document per page with enough usable text.

    Raises:
        KeyError: If `split` is not a known split.
        FileNotFoundError: If no shards have been downloaded yet.
    """
    if split not in SPLITS:
        raise KeyError(f"Unknown split {split!r}")
    shards = shard_paths(cache_dir)
    if not shards:
        raise FileNotFoundError(
            "No PDFA shards found. Download pixparse/pdfa-eng-wds into the raw cache first."
        )

    # Held out by shard so a document never straddles two splits
    if split == "validation":
        chosen = shards[-(VALIDATION_SHARDS + TEST_SHARDS) : -TEST_SHARDS]
    elif split == "test":
        chosen = shards[-TEST_SHARDS:]
    else:
        chosen = shards[: -(VALIDATION_SHARDS + TEST_SHARDS)]

    for shard in chosen:
        yield from _read_shard(shard)


def _read_shard(shard: Path) -> Iterator[Document]:
    """
    Streams one webdataset shard, reading only its JSON members

    A corrupt shard is skipped rather than killing a build millions of pages in.

    Args:
        shard: Path to a `.tar` shard.

    Yields:
        One Document per usable page.
    """
    try:
        archive = tarfile.open(shard, "r")
    except tarfile.TarError, OSError:
        return
    with archive:
        try:
            for member in archive:
                if not member.name.endswith(".json"):
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                try:
                    payload = json.load(handle)
                except json.JSONDecodeError, UnicodeDecodeError:
                    continue
                stem = member.name.rsplit("/", 1)[-1].removesuffix(".json")
                for index, page in enumerate(payload.get("pages") or []):
                    document = _build(f"pdfa-{shard.stem}-{stem}-{index}", page)
                    if document is not None:
                        yield document
        except tarfile.TarError, OSError:
            return


def _build(doc_id: str, page: dict[str, Any]) -> Document | None:
    """
    Assembles one page from its word records

    Args:
        doc_id: Identifier for the resulting document.
        page: One entry of the JSON `pages` list.

    Returns:
        The Document, or None when the page has too little usable text.
    """
    words_field = page.get("words") or {}
    texts = words_field.get("text") or []
    boxes_field = words_field.get("bbox") or []
    scores = words_field.get("score") or []
    if not texts or len(texts) != len(boxes_field):
        return None

    words: list[str] = []
    boxes: list[list[int]] = []
    for position, (text, raw) in enumerate(zip(texts, boxes_field, strict=False)):
        cleaned = (text or "").strip()
        if not cleaned or len(raw) != 4:
            continue
        if position < len(scores) and scores[position] < MIN_SCORE:
            continue
        x, y, width, height = (float(value) for value in raw)
        # Normalized xywh in [0, 1] to xyxy on the canonical grid
        box = [
            x * COORDINATE_SCALE,
            y * COORDINATE_SCALE,
            (x + width) * COORDINATE_SCALE,
            (y + height) * COORDINATE_SCALE,
        ]
        words.append(cleaned)
        boxes.append(clamp_box(box, COORDINATE_SCALE))

    if len(words) < MIN_WORDS:
        return None
    return Document(
        doc_id=doc_id,
        source="pdfa",
        words=words,
        bboxes=boxes,
        # Use the grid as the page since boxes arrive normalized
        page_width=COORDINATE_SCALE,
        page_height=COORDINATE_SCALE,
    )
