"""
RVL-CDIP: document image classification, 400k pages, 16 classes

The standard release ships images only, and OCR-ing 400k scans is a day of compute we do not need
to spend. `jordyvl/rvl_cdip_easyocr` already carries EasyOCR output for every page, so this adapter
reads that instead.

Two things about it need handling. It is a legacy loading-script dataset, so the `.npy` payloads are
downloaded and unpickled directly rather than going through `load_dataset`. And its "word-level"
annotations are really segment level: an entry's text is often a phrase like
"Plasminogen Activator Activity in" sharing one box, so segments are split into words with their
box divided along x.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download

from ..normalize import normalize_box, split_segment
from ..schema import DatasetMeta, Document

REPO = "jordyvl/rvl_cdip_easyocr"
SPLITS = {
    "train": "data/Easy_Train_Data.npy",
    "validation": "data/Easy_Valid_Data.npy",
    "test": "data/Easy_Test_Data.npy",
}

"""
The 16 RVL-CDIP classes, in the canonical order the original label ids use

Two lists, because the mirror does not spell the classes the way the dataset paper does. It writes
`report` for "scientific report", `publication` for "scientific publication", and underscores
`file_folder` and `news_article`. `SOURCE_LABELS` holds what the mirror actually emits and
`CLASS_NAMES` holds the readable names used in reports, index for index.
"""
SOURCE_LABELS = [
    "letter",
    "form",
    "email",
    "handwritten",
    "advertisement",
    "report",
    "publication",
    "specification",
    "file_folder",
    "news_article",
    "budget",
    "invoice",
    "presentation",
    "questionnaire",
    "resume",
    "memo",
]
CLASS_NAMES = [
    "letter",
    "form",
    "email",
    "handwritten",
    "advertisement",
    "scientific report",
    "scientific publication",
    "specification",
    "file folder",
    "news article",
    "budget",
    "invoice",
    "presentation",
    "questionnaire",
    "resume",
    "memo",
]
_CLASS_INDEX = {name: index for index, name in enumerate(SOURCE_LABELS)}

# Pages with almost no OCR text carry no signal and only waste sequence budget.
MIN_WORDS = 4


def describe() -> DatasetMeta:
    """
    Returns the label vocabulary and task for RVL-CDIP

    Returns:
        Metadata for 16-way document classification.
    """
    return DatasetMeta(
        source="rvlcdip",
        task="sequence",
        label_names=CLASS_NAMES,
        notes=(
            "Text and boxes are precomputed EasyOCR from jordyvl/rvl_cdip_easyocr, not the "
            "Tesseract output the original LayoutLM papers used, so absolute numbers differ. "
            "Segment-level OCR entries are split into words with boxes divided along x. "
            f"Pages with fewer than {MIN_WORDS} words are dropped."
        ),
    )


def load(split: str, cache_dir: str | None = None) -> Iterator[Document]:
    """
    Yields RVL-CDIP pages in the canonical schema

    Args:
        split: One of the keys of `SPLITS`.
        cache_dir: Hugging Face cache directory, or None for the default.

    Yields:
        One Document per page that has enough OCR text.

    Raises:
        KeyError: If `split` is not a known split.
    """
    path = Path(hf_hub_download(REPO, SPLITS[split], repo_type="dataset", cache_dir=cache_dir))
    entries = np.load(path, allow_pickle=True)
    for index, entry in enumerate(entries):
        document = _build(f"rvlcdip-{split}-{index}", entry)
        if document is not None:
            yield document


def _build(doc_id: str, entry: dict[str, Any]) -> Document | None:
    """
    Assembles one page from its OCR entry

    Args:
        doc_id: Identifier for the resulting document.
        entry: One unpickled record, holding `images` and `word-level annotations`.

    Returns:
        The Document, or None when the page has too little text.

    Raises:
        ValueError: If the page's class is not one of the 16 known ones. Skipping it instead
            silently discarded a quarter of the corpus when four class names turned out to be
            spelled differently by the mirror.
    """
    image = entry["images"][0]
    name = str(image["label"]).strip()
    label = _CLASS_INDEX.get(name)
    if label is None:
        raise ValueError(
            f"{doc_id}: unknown RVL-CDIP class {name!r}. Known: {SOURCE_LABELS}. "
            f"Fix SOURCE_LABELS rather than dropping the page."
        )

    width = int(image["image_width"])
    height = int(image["image_height"])
    if width <= 0 or height <= 0:
        return None

    annotations = entry.get("word-level annotations") or []
    if not annotations:
        return None
    texts = annotations[0].get("ocred_text") or []
    raw_boxes = annotations[0].get("ocred_boxes") or []

    words: list[str] = []
    boxes: list[list[int]] = []
    for text, raw_box in zip(texts, raw_boxes, strict=False):
        pixel_box = [raw_box["x0"], raw_box["y0"], raw_box["x1"], raw_box["y1"]]
        for word, split_box in split_segment(str(text), pixel_box):
            words.append(word)
            boxes.append(normalize_box(split_box, width, height))

    if len(words) < MIN_WORDS:
        return None
    return Document(
        doc_id=doc_id,
        source="rvlcdip",
        words=words,
        bboxes=boxes,
        page_width=width,
        page_height=height,
        doc_label=label,
    )
