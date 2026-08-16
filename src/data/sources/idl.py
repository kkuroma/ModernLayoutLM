"""
IDL: scanned pages read from the IMDB release, which carries only words and boxes

Boxes are already (left, top, right, bottom) normalized to [0, 1]
Entry zero of each .npy shard is a header rather than a page
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ..contamination import Fingerprints
from ..normalize import COORDINATE_SCALE, clamp_box
from ..schema import DatasetMeta, Document

# Keep separate from cache_dir as the HF cache globs RVL-CDIP's .npy files
DEFAULT_ROOT = Path(os.environ.get("IDL_IMDB_ROOT", "data/raw/idl/imdbs"))

# Match the prefix so another dataset's .npy is not read as ours
SHARD_GLOB = "imdb_*.npy"

# Stride the shards as they are ordered by collection
SHARD_STRIDE = max(1, int(os.environ.get("IDL_SHARD_STRIDE", "1")))

logger = logging.getLogger(__name__)

SPLITS = {"train": "train", "validation": "validation", "test": "test"}

# Files held out, taken from the end so the split is stable as more files are extracted
VALIDATION_FILES = 1
TEST_FILES = 1

# A page with almost no OCR text is not worth a training sequence
MIN_WORDS = 8

# Drop benchmark Bates numbers as IDL shares a library with FUNSD and RVL-CDIP
_FINGERPRINTS: Fingerprints | None = None
_DROPPED = 0


def dropped_for_contamination() -> int:
    """
    Reports how many pages this process skipped as benchmark overlap

    Returns:
        The running count, for the build report.
    """
    return _DROPPED


def describe() -> DatasetMeta:
    """
    Returns the task for IDL as used here

    Returns:
        Metadata marking this a pretraining corpus with no label vocabulary.
    """
    return DatasetMeta(
        source="idl",
        task="pretrain",
        label_names=[],
        notes=(
            "UCSF Industry Documents Library, the corpus IIT-CDIP and therefore LayoutLM v1 were "
            "drawn from. Words and boxes come from the IMDB distillation of Amazon Textract "
            "output, 127GB against 1.34TB for the raw annotations and 2.6TB for the webdataset "
            "mirror, because only words and boxes are needed. Boxes are normalized xyxy in [0, 1] "
            "and are rescaled to the 0-1000 grid."
        ),
    )


def shard_files(root: Path | None = None) -> list[Path]:
    """
    Lists the extracted IMDB shards

    Args:
        root: Directory holding the `.npy` files, or None for the default.

    Returns:
        Every `SHARD_STRIDE`th `.npy` shard found, sorted by name.
    """
    found = sorted((root or DEFAULT_ROOT).glob(f"**/{SHARD_GLOB}"))
    return found[::SHARD_STRIDE]


def load(split: str, cache_dir: str | None = None) -> Iterator[Document]:
    """
    Yields IDL pages in the canonical schema

    Args:
        split: One of the keys of `SPLITS`.
        cache_dir: Ignored. IDL is not fetched through the Hugging Face cache, and the argument
            exists only to match the adapter interface. Set `IDL_IMDB_ROOT` to relocate the shards.

    Yields:
        One Document per page with enough OCR text.

    Raises:
        KeyError: If `split` is not a known split.
        FileNotFoundError: If no shards have been extracted yet.
    """
    if split not in SPLITS:
        raise KeyError(f"Unknown split {split!r}")
    del cache_dir  # see the docstring: the shard root is IDL_IMDB_ROOT, not the HF cache
    files = shard_files()
    if not files:
        raise FileNotFoundError(
            f"No {SHARD_GLOB} under {DEFAULT_ROOT}. Extract imdbs_v2.tar.gz there, "
            f"or set IDL_IMDB_ROOT."
        )

    # Held out by file, so a document never straddles two splits
    if len(files) <= VALIDATION_FILES + TEST_FILES:
        chosen = files
    elif split == "validation":
        chosen = files[-(VALIDATION_FILES + TEST_FILES) : -TEST_FILES]
    elif split == "test":
        chosen = files[-TEST_FILES:]
    else:
        chosen = files[: -(VALIDATION_FILES + TEST_FILES)]

    global _FINGERPRINTS
    if _FINGERPRINTS is None:
        _FINGERPRINTS = Fingerprints.read(DEFAULT_ROOT.parent)
        logger.info("contamination filter: %d benchmark fingerprints", len(_FINGERPRINTS))

    for path in chosen:
        yield from _read_shard(path)


def _read_shard(path: Path) -> Iterator[Document]:
    """
    Streams one IMDB shard

    The array is memory-mapped where possible so a 3.4GB shard does not have to be resident all at
    once, and a shard that fails to unpickle is skipped rather than killing a build that may be
    part way through millions of pages.

    Args:
        path: Path to one `.npy` shard.

    Yields:
        One Document per usable page.
    """
    try:
        entries = np.load(path, allow_pickle=True)
    except OSError, ValueError, EOFError:
        return
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or "ocr_tokens" not in entry:
            continue  # entry zero is the shard header
        if _FINGERPRINTS is not None:
            hit = _FINGERPRINTS.contaminated(entry.get("ocr_tokens") or [])
            if hit is not None:
                global _DROPPED
                _DROPPED += 1
                continue
        document = _build(f"idl-{path.stem}-{index}", entry)
        if document is not None:
            yield document


def _build(doc_id: str, entry: dict[str, Any]) -> Document | None:
    """
    Assembles one page from its IMDB entry

    Args:
        doc_id: Identifier for the resulting document.
        entry: One page record holding `ocr_tokens` and `ocr_normalized_boxes`.

    Returns:
        The Document, or None when the page has too little text or malformed boxes.
    """
    tokens = entry.get("ocr_tokens") or []
    raw_boxes = entry.get("ocr_normalized_boxes")
    if raw_boxes is None or len(tokens) != len(raw_boxes):
        return None

    words: list[str] = []
    boxes: list[list[int]] = []
    for text, raw in zip(tokens, raw_boxes, strict=False):
        cleaned = str(text).strip()
        if not cleaned or len(raw) != 4:
            continue
        # Already (left, top, right, bottom) in [0, 1]; only the scale changes
        words.append(cleaned)
        boxes.append(clamp_box([float(value) * COORDINATE_SCALE for value in raw]))

    if len(words) < MIN_WORDS:
        return None
    return Document(
        doc_id=doc_id,
        source="idl",
        words=words,
        bboxes=boxes,
        # Use the canonical grid as the page since the release records no pixel size
        page_width=COORDINATE_SCALE,
        page_height=COORDINATE_SCALE,
    )
