"""
IDL: scanned pages read from the IMDB release, which carries only words and boxes

Boxes are already (left, top, right, bottom) normalized to [0, 1], and entry zero is a header
The `Fresh` and `Docs` classes serve shards earlier corpora never read, `Docs` as whole documents
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ..contamination import Fingerprints
from ..schema import COORDINATE_SCALE, DatasetMeta, Document, clamp_boxes

# The shards live outside the HF cache, relocatable with IDL_IMDB_ROOT
DEFAULT_ROOT = Path(os.environ.get("IDL_IMDB_ROOT", "data/raw/idl/imdbs"))

# Match the prefix so another dataset's .npy is not read as ours
SHARD_GLOB = "imdb_*.npy"

# The corpus reads every ninth shard, a spread across collections as the shards are ordered by one
SHARD_STRIDE = 9

SPLITS = {"train": "train", "validation": "validation", "test": "test"}

# Files held out, taken from the end so the split is stable as more files are extracted
VALIDATION_FILES = 1
TEST_FILES = 1

# A page with almost no OCR text earns no training sequence
MIN_WORDS = 8

logger = logging.getLogger(__name__)

# Drop benchmark Bates numbers as IDL shares a library with FUNSD and RVL-CDIP
_FINGERPRINTS: Fingerprints | None = None


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


def split_files(files: list[Path], split: str) -> list[Path]:
    """
    Picks one split's shards, holding the last files out so a page never straddles two splits

    Args:
        files: The shards to split, in a stable order.
        split: One of the keys of `SPLITS`.

    Returns:
        The shards belonging to `split`.

    Raises:
        KeyError: If `split` is not a known split.
        FileNotFoundError: If `files` is empty.
    """
    if split not in SPLITS:
        raise KeyError(f"Unknown split {split!r}")
    if not files:
        raise FileNotFoundError(f"No {SHARD_GLOB} under {DEFAULT_ROOT}, or set IDL_IMDB_ROOT")
    held = VALIDATION_FILES + TEST_FILES
    if len(files) <= held:
        return files
    if split == "validation":
        return files[-held:-TEST_FILES]
    if split == "test":
        return files[-TEST_FILES:]
    return files[:-held]


def parts(split: str) -> list[Path]:
    """
    Lists the shards of one split, the unit the build spreads over processes

    Args:
        split: One of the keys of `SPLITS`.

    Returns:
        The split's shard files.
    """
    return split_files(shard_files(), split)


def load(split: str, cache_dir: str | None = None) -> Iterator[Document]:
    """
    Yields IDL pages in the canonical schema

    Args:
        split: One of the keys of `SPLITS`.
        cache_dir: Ignored, as the shards live under `IDL_IMDB_ROOT` rather than the HF cache.

    Yields:
        One Document per page with enough OCR text.
    """
    del cache_dir
    for path in parts(split):
        yield from load_part(path)


def load_part(path: Path) -> Iterator[Document]:
    """
    Streams one shard through the contamination filter

    Args:
        path: Path to one `.npy` shard.

    Yields:
        One Document per usable page.
    """
    for index, entry in _pages(path):
        document = _build(f"idl-{path.stem}-{index}", entry)
        if document is not None:
            yield document


def load_documents(path: Path) -> Iterator[Document]:
    """
    Streams one shard as whole documents, their pages in order and each on its own grid

    The release stores a document's pages together and in page order, so a new image id ends one.

    Args:
        path: Path to one `.npy` shard.

    Yields:
        One Document per document with at least one usable page.
    """
    current, pages = None, []
    for index, entry in _pages(path):
        if entry.get("image_id") != current:
            if pages:
                yield _join(f"idl-{current}", pages)
            current, pages = entry.get("image_id"), []
        page = _build(f"idl-{current}-{index}", entry)
        if page is not None:
            pages.append(page)
    if pages:
        yield _join(f"idl-{current}", pages)


def _pages(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """
    Yields the page entries of one shard that pass the contamination filter

    Args:
        path: Path to one `.npy` shard.

    Yields:
        Each entry's index in the shard and the entry.
    """
    global _FINGERPRINTS
    if _FINGERPRINTS is None:
        _FINGERPRINTS = Fingerprints.read(DEFAULT_ROOT.parent)
        logger.info("contamination filter: %d benchmark fingerprints", len(_FINGERPRINTS))
    try:
        entries = np.load(path, allow_pickle=True)
    except OSError, ValueError, EOFError:
        return
    for index, entry in enumerate(entries):
        # Release each page as it is read so a worker's memory shrinks through the shard
        entries[index] = None
        if not isinstance(entry, dict) or "ocr_tokens" not in entry:
            continue
        if _FINGERPRINTS.contaminated(entry.get("ocr_tokens") or []) is not None:
            continue
        yield index, entry


def _join(doc_id: str, pages: list[Document]) -> Document:
    """
    Concatenates a document's pages, leaving each page's boxes on its own grid

    Args:
        doc_id: Identifier for the joined document.
        pages: The document's pages in order.

    Returns:
        One Document holding every page's words and boxes in order.
    """
    return Document(
        doc_id=doc_id,
        source="idl",
        words=[word for page in pages for word in page.words],
        bboxes=[box for page in pages for box in page.bboxes],
        page_width=COORDINATE_SCALE,
        page_height=COORDINATE_SCALE,
    )


def _build(doc_id: str, entry: dict[str, Any]) -> Document | None:
    """
    Assembles one page from its IMDB entry, dropping blank words

    Args:
        doc_id: Identifier for the resulting document.
        entry: One page record holding `ocr_tokens` and an (N, 4) `ocr_normalized_boxes`.

    Returns:
        The Document, or None when the page has too little text or mismatched boxes.
    """
    tokens = entry.get("ocr_tokens") or []
    raw = entry.get("ocr_normalized_boxes")
    if raw is None or len(tokens) != len(raw):
        return None
    words = [str(text).strip() for text in tokens]
    keep = [index for index, word in enumerate(words) if word]
    if len(keep) < MIN_WORDS:
        return None
    return Document(
        doc_id=doc_id,
        source="idl",
        words=[words[index] for index in keep],
        bboxes=clamp_boxes(np.asarray(raw, dtype=np.float64)[keep] * COORDINATE_SCALE),
        # Use the canonical grid as the page since the release records no pixel size
        page_width=COORDINATE_SCALE,
        page_height=COORDINATE_SCALE,
    )


class Fresh:
    """
    The `idl-fresh` source: shards the first corpus left unread, for continued pretraining

    It excludes every shard the idl corpus reads, so no page is seen twice
    """

    SPLITS = SPLITS

    # 19 train shards of ~480k pages cover the 8.8M fresh pages an 11M-page pass needs
    TRAIN_FILES = 19

    load_part = staticmethod(load_part)

    @staticmethod
    def describe() -> DatasetMeta:
        """
        Returns the task for the fresh shards

        Returns:
            Metadata marking this a pretraining corpus with no label vocabulary.
        """
        return DatasetMeta(
            source="idl-fresh",
            task="pretrain",
            label_names=[],
            notes=(
                "The IDL IMDB shards left unread by the idl corpus, for continuing pretraining "
                "past its 2.19M pages without repeating one. Same reader, filter and box handling "
                "as idl."
            ),
        )

    @staticmethod
    def shard_files(root: Path | None = None) -> list[Path]:
        """
        Lists the unread shards, enough for the train split and the two held out

        Args:
            root: Directory holding the `.npy` files, or None for the default.

        Returns:
            The unread shards sorted by name, truncated to what the splits need.
        """
        found = sorted((root or DEFAULT_ROOT).glob(f"**/{SHARD_GLOB}"))
        used = set(found[::SHARD_STRIDE])
        fresh = [path for path in found if path not in used]
        return fresh[: Fresh.TRAIN_FILES + VALIDATION_FILES + TEST_FILES]

    @staticmethod
    def parts(split: str) -> list[Path]:
        """
        Lists the shards of one split, the unit the build spreads over processes

        Args:
            split: One of the keys of `SPLITS`.

        Returns:
            The split's shard files.
        """
        return split_files(Fresh.shard_files(), split)

    @staticmethod
    def load(split: str, cache_dir: str | None = None) -> Iterator[Document]:
        """
        Yields fresh IDL pages in the canonical schema

        Args:
            split: One of the keys of `SPLITS`.
            cache_dir: Ignored, as for idl.

        Yields:
            One Document per page with enough OCR text.
        """
        del cache_dir
        for path in Fresh.parts(split):
            yield from load_part(path)


class Docs:
    """
    The `idl-docs` source: whole multi-page documents from shards neither earlier corpus read

    It feeds the long-window phase, so one window holds consecutive pages of one document
    """

    SPLITS = SPLITS

    # 5 train shards hold about 1.1B tokens, enough for the long phase without a repeat
    TRAIN_FILES = 5

    load_part = staticmethod(load_documents)

    @staticmethod
    def describe() -> DatasetMeta:
        """
        Returns the task for the document shards

        Returns:
            Metadata marking this a pretraining corpus with no label vocabulary.
        """
        return DatasetMeta(
            source="idl-docs",
            task="pretrain",
            label_names=[],
            notes=(
                "IDL IMDB shards read by neither idl nor idl-fresh, one row per document rather "
                "than per page. Pages keep their order and each keeps its own 0-1000 grid, so "
                "page order is carried by token position alone. Same filter as idl."
            ),
        )

    @staticmethod
    def shard_files(root: Path | None = None) -> list[Path]:
        """
        Lists the shards no earlier corpus read, enough for the train split and the two held out

        Args:
            root: Directory holding the `.npy` files, or None for the default.

        Returns:
            The unread shards sorted by name, truncated to what the splits need.
        """
        found = sorted((root or DEFAULT_ROOT).glob(f"**/{SHARD_GLOB}"))
        used = set(found[::SHARD_STRIDE]) | set(Fresh.shard_files(root))
        unread = [path for path in found if path not in used]
        return unread[: Docs.TRAIN_FILES + VALIDATION_FILES + TEST_FILES]

    @staticmethod
    def parts(split: str) -> list[Path]:
        """
        Lists the shards of one split, the unit the build spreads over processes

        Args:
            split: One of the keys of `SPLITS`.

        Returns:
            The split's shard files.
        """
        return split_files(Docs.shard_files(), split)

    @staticmethod
    def load(split: str, cache_dir: str | None = None) -> Iterator[Document]:
        """
        Yields whole IDL documents in the canonical schema

        Args:
            split: One of the keys of `SPLITS`.
            cache_dir: Ignored, as for idl.

        Yields:
            One Document per document with at least one usable page.
        """
        del cache_dir
        for path in Docs.parts(split):
            yield from load_documents(path)
