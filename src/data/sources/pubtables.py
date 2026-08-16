"""
PubTables-1M: PubMed pages of words and boxes, the main pretraining corpus

Only the word and filelist tarballs are fetched as this model has no image tower
Boxes are word level and live in image_rect pixel space rather than pdf_page_rect points
"""

from __future__ import annotations

import json
import tarfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from ..normalize import normalize_box
from ..schema import DatasetMeta, Document

REPO = "bsmock/pubtables-1m"
WORDS_ARCHIVE = "PubTables-1M-Detection_Page_Words.tar.gz"
FILELIST_ARCHIVE = "PubTables-1M-Detection_Filelists.tar.gz"

SPLITS = {
    "train": "train_filelist.txt",
    "validation": "val_filelist.txt",
    "test": "test_filelist.txt",
}

# A page with almost no extractable text is not worth a training sequence
MIN_WORDS = 8


def describe() -> DatasetMeta:
    """
    Returns the task for PubTables-1M as used here

    Returns:
        Metadata marking this a pretraining corpus with no label vocabulary.
    """
    return DatasetMeta(
        source="pubtables",
        task="pretrain",
        label_names=[],
        notes=(
            "Words and boxes come from the Detection_Page_Words tarball, extracted from the "
            "source PDFs, so no OCR is involved and the boxes are word level. Coordinates are in "
            "image_rect pixel space, not pdf_page_rect points. The 90GB of image tarballs are "
            "never downloaded."
        ),
    )


def load(split: str, cache_dir: str | None = None) -> Iterator[Document]:
    """
    Yields PubTables-1M pages in the canonical schema

    The words archive is streamed once from front to back and each page is matched against the
    split's file list, because a 6.5GB gzip stream cannot be seeked into cheaply.

    Args:
        split: One of the keys of `SPLITS`.
        cache_dir: Hugging Face cache directory, or None for the default.

    Yields:
        One Document per page of this split that has enough text.

    Raises:
        KeyError: If `split` is not a known split.
    """
    wanted = _split_page_ids(split, cache_dir)
    words_path = Path(
        hf_hub_download(REPO, WORDS_ARCHIVE, repo_type="dataset", cache_dir=cache_dir)
    )

    with tarfile.open(words_path, "r:gz") as archive:
        for member in archive:
            page_id = _page_id(member.name)
            if page_id is None or page_id not in wanted:
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            document = _build(f"pubtables-{split}-{page_id}", json.load(handle))
            if document is not None:
                yield document


def _split_page_ids(split: str, cache_dir: str | None) -> set[str]:
    """
    Reads the page ids belonging to one split

    Args:
        split: One of the keys of `SPLITS`.
        cache_dir: Hugging Face cache directory, or None for the default.

    Returns:
        Page ids such as "PMC6589332_11".

    Raises:
        KeyError: If `split` is not a known split.
        FileNotFoundError: If the archive does not contain the split's file list.
    """
    filename = SPLITS[split]
    path = Path(hf_hub_download(REPO, FILELIST_ARCHIVE, repo_type="dataset", cache_dir=cache_dir))
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if not member.name.endswith(filename):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            lines = handle.read().decode().splitlines()
            return {Path(line.strip()).stem for line in lines if line.strip()}
    raise FileNotFoundError(f"{filename} not found inside {FILELIST_ARCHIVE}")


def _page_id(name: str) -> str | None:
    """
    Extracts a page id from a words filename

    Args:
        name: An archive member name such as "PMC5836426_5_words.json".

    Returns:
        The page id, or None if the member is not a words file.
    """
    stem = name.rsplit("/", 1)[-1]
    if not stem.endswith("_words.json"):
        return None
    return stem.removesuffix("_words.json")


def _build(doc_id: str, payload: dict[str, Any]) -> Document | None:
    """
    Assembles one page from its words file

    Args:
        doc_id: Identifier for the resulting document.
        payload: The parsed JSON, holding `image_rect` and `words`.

    Returns:
        The Document, or None when the page has too little text or no usable page rectangle.
    """
    rect = payload.get("image_rect") or []
    if len(rect) != 4:
        return None
    width = int(rect[2] - rect[0])
    height = int(rect[3] - rect[1])
    if width <= 0 or height <= 0:
        return None

    words: list[str] = []
    boxes: list[list[int]] = []
    for entry in payload.get("words") or []:
        text = (entry.get("text") or "").strip()
        box = entry.get("bbox")
        if not text or not box or len(box) != 4:
            continue
        words.append(text)
        boxes.append(normalize_box(box, width, height))

    if len(words) < MIN_WORDS:
        return None
    return Document(
        doc_id=doc_id,
        source="pubtables",
        words=words,
        bboxes=boxes,
        page_width=width,
        page_height=height,
    )
