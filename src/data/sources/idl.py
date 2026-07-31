"""
IDL: 27M scanned pages from the UCSF Industry Documents Library

The domain the original LayoutLM was built on. IIT-CDIP, LayoutLM v1's 11M-page corpus, is drawn
from this same library, and RVL-CDIP and FUNSD descend from it in turn. Adding it is what closes
the domain gap our PubMed-heavy corpus leaves open.

What makes it affordable is using the IMDB release rather than the raw one. Amazon Textract's full
output for these pages is 1.34TB and the webdataset mirror bundling PDFs and TIFFs is 2.6TB, but
neither is needed: this model has no image tower and wants only words and boxes. The IMDB bundle
distills exactly that into 127GB, and its boxes are already `(left, top, right, bottom)` normalized
to `[0, 1]`, which is our own convention modulo the 1000x scale.

Each `.npy` is a pickled object array of 500,001 entries. Entry zero is a header describing the
shard, and the rest are pages, so the header is skipped by shape rather than by index.
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

# Where the extracted imdb_*.npy files live. The bundle is one tar.gz of 54 such files.
# Overridable for testing; deliberately NOT taken from the adapter's `cache_dir` argument, which
# every other source uses to mean the Hugging Face cache. Pointing this at that cache made the
# glob pick up RVL-CDIP's unrelated .npy files and silently yield nothing.
DEFAULT_ROOT = Path(os.environ.get("IDL_IMDB_ROOT", "data/raw/idl/imdbs"))

# IMDB shards are named imdb_<split>_p<N>.npy. Matching the prefix stops a stray .npy from
# another dataset being read as if it were one of ours.
SHARD_GLOB = "imdb_*.npy"

logger = logging.getLogger(__name__)

SPLITS = {"train": "train", "validation": "validation", "test": "test"}

# Files held out, taken from the end so the split is stable as more files are extracted.
VALIDATION_FILES = 1
TEST_FILES = 1

# A page with almost no OCR text is not worth a training sequence.
MIN_WORDS = 8

# Loaded once per process. IDL descends from the same library as FUNSD and RVL-CDIP, so any page
# carrying a benchmark Bates number is dropped before it can reach pretraining.
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
        Every `.npy` shard found, sorted by name.
    """
    return sorted((root or DEFAULT_ROOT).glob(f"**/{SHARD_GLOB}"))


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

    # Held out by file, so a document never straddles two splits.
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
        # Already (left, top, right, bottom) in [0, 1]; only the scale changes.
        words.append(cleaned)
        boxes.append(clamp_box([float(value) * COORDINATE_SCALE for value in raw]))

    if len(words) < MIN_WORDS:
        return None
    return Document(
        doc_id=doc_id,
        source="idl",
        words=words,
        bboxes=boxes,
        # The release records no pixel dimensions, and the boxes arrive normalized, so the
        # canonical grid stands in as the page.
        page_width=COORDINATE_SCALE,
        page_height=COORDINATE_SCALE,
    )
