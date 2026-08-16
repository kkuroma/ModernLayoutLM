"""
CORD: receipt key information extraction over Indonesian receipts

Words are buried in one JSON string per receipt, as four-corner quads in pixels
Field names come from the line rather than the word, so BIO is derived from each line's word run
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, cast

from datasets import Dataset, load_dataset

from ..normalize import normalize_box, quad_to_xyxy
from ..schema import DatasetMeta, Document

REPO = "naver-clova-ix/cord-v2"
SPLITS = {"train": "train", "validation": "validation", "test": "test"}

"""
The 29 CORD field categories, enumerated from all three splits rather than from the paper

The published category list writes the sub-menu fields with underscores, as `menu.sub_nm`. The
released data uses dots, as `menu.sub.nm`, and has no `menu.sub.etc` at all. Fixing the list here
rather than discovering it per split means a split missing a rare field still produces the same
label ids, which is what makes the numbers comparable across splits.
"""
CATEGORIES = [
    "menu.cnt",
    "menu.discountprice",
    "menu.etc",
    "menu.itemsubtotal",
    "menu.nm",
    "menu.num",
    "menu.price",
    "menu.sub.cnt",
    "menu.sub.nm",
    "menu.sub.price",
    "menu.sub.unitprice",
    "menu.unitprice",
    "menu.vatyn",
    "sub_total.discount_price",
    "sub_total.etc",
    "sub_total.othersvc_price",
    "sub_total.service_price",
    "sub_total.subtotal_price",
    "sub_total.tax_price",
    "total.cashprice",
    "total.changeprice",
    "total.creditcardprice",
    "total.emoneyprice",
    "total.menuqty_cnt",
    "total.menutype_cnt",
    "total.total_etc",
    "total.total_price",
    "void_menu.nm",
    "void_menu.price",
]

LABEL_NAMES = ["O"] + [f"{prefix}-{c}" for c in CATEGORIES for prefix in ("B", "I")]
_LABEL_INDEX = {name: index for index, name in enumerate(LABEL_NAMES)}


def describe() -> DatasetMeta:
    """
    Returns the label vocabulary and task for CORD

    Returns:
        Metadata with 59 BIO labels over the 29 CORD categories.
    """
    return DatasetMeta(
        source="cord",
        task="token",
        label_names=LABEL_NAMES,
        notes=(
            "Words parsed out of the ground_truth JSON; four-corner quads reduced to "
            "axis-aligned boxes. BIO derived per valid_line, so one line is one entity. "
            "Categories absent from the source are still reserved label ids."
        ),
    )


def load(split: str, cache_dir: str | None = None) -> Iterator[Document]:
    """
    Yields CORD receipts in the canonical schema

    Args:
        split: One of the keys of `SPLITS`.
        cache_dir: Hugging Face cache directory, or None for the default.

    Yields:
        One Document per receipt. Receipts whose lines hold no usable words are skipped.

    Raises:
        KeyError: If `split` is not a known split.
    """
    dataset = cast(Dataset, load_dataset(REPO, split=SPLITS[split], cache_dir=cache_dir))
    for index, raw in enumerate(dataset):
        row = cast(dict[str, Any], raw)
        parsed = json.loads(row["ground_truth"])
        size = parsed["meta"]["image_size"]
        document = _build(
            doc_id=f"cord-{split}-{index}",
            lines=parsed.get("valid_line", []),
            width=size["width"],
            height=size["height"],
        )
        if document is not None:
            yield document


def _build(
    doc_id: str,
    lines: list[dict],
    width: int,
    height: int,
) -> Document | None:
    """
    Assembles one receipt from its annotated lines

    Args:
        doc_id: Identifier for the resulting document.
        lines: The `valid_line` entries.
        width: Page width in pixels.
        height: Page height in pixels.

    Returns:
        The Document, or None when no line yielded a word.

    Raises:
        ValueError: If a line carries a category absent from `CATEGORIES`. Mapping it to O
            instead would silently erase an entire field from the training signal.
    """
    words: list[str] = []
    boxes: list[list[int]] = []
    labels: list[int] = []

    for line in lines:
        category = line.get("category")
        if category not in CATEGORIES:
            raise ValueError(
                f"{doc_id}: unknown CORD category {category!r}. Add it to CATEGORIES rather "
                f"than letting it fall through to O."
            )
        for position, word in enumerate(line.get("words", [])):
            text = (word.get("text") or "").strip()
            if not text:
                continue
            box = normalize_box(quad_to_xyxy(word["quad"]), width, height)
            prefix = "B" if position == 0 else "I"
            label = _LABEL_INDEX.get(f"{prefix}-{category}", 0)
            words.append(text)
            boxes.append(box)
            labels.append(label)

    if not words:
        return None
    return Document(
        doc_id=doc_id,
        source="cord",
        words=words,
        bboxes=boxes,
        page_width=width,
        page_height=height,
        word_labels=labels,
    )
