"""
RealKIE: long enterprise documents labeled as character spans over the release's own OCR

A span tags every OCR word it overlaps, the first `B-` and the rest `I-`, so entity edges are exact
Each page keeps its own 0-1000 grid, and FCC Invoices, NDA and Charities share this one reader
"""

from __future__ import annotations

import csv
import gzip
import json
import logging
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..schema import COORDINATE_SCALE, DatasetMeta, Document, normalize_box

ROOT = Path("data/raw/realkie")

SPLITS = {"train": "train", "validation": "val", "test": "test"}

# The label set each dataset ships, so an unrecognized label fails the build
FIELDS = {
    "fcc_invoices": [
        "Advertiser",
        "Agency",
        "Agency Commission",
        "Gross Total",
        "Line Item - Days",
        "Line Item - Description",
        "Line Item - End Date",
        "Line Item - Rate",
        "Line Item - Start Date",
        "Net Amount Due",
        "Payment Terms",
    ],
    "nda": ["Effective Date", "Jurisdiction", "Party"],
    "charities": [
        "Accounting Basis",
        "Bank Name",
        "Cash In Hand at Current Year End",
        "Cash In Hand at Previous Year End",
        "Charity Name",
        "Charity Registered Number",
        "Company Number",
        "Event Name",
        "Examination Date",
        "Independent Examiner City",
        "Independent Examiner Company",
        "Independent Examiner Name",
        "Independent Examiner Postal Code",
        "Independent Examiner Street Address",
        "Named Donor",
        "Named Employee",
        "Net Assets at Current Year End",
        "Net Assets at Previous Year End",
        "Net Income at Current Year End",
        "Net Income at Previous Year End",
        "Objectives and Activities",
        "Principal Office City",
        "Principal Office Postal Code",
        "Principal Office Street Address",
        "Project Name",
        "Trustee Name",
        "Trustee Title",
        "Year Ended",
    ],
}

logger = logging.getLogger(__name__)


def slug(label: str) -> str:
    """
    Turns a RealKIE label into a tag-safe name, such as `line_item_days`

    Args:
        label: The label as the release spells it.

    Returns:
        The lowercase name with every run of other characters as one underscore.
    """
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


class RealKie:
    """
    One RealKIE dataset as a source

    Args:
        name: The dataset's directory under the RealKIE root, a key of `FIELDS`.
    """

    SPLITS = SPLITS

    def __init__(self, name: str) -> None:
        self.name = name
        self.source = f"realkie-{name.removesuffix('_invoices')}"
        self.fields = FIELDS[name]
        self.label_names = ["O"] + [
            f"{prefix}-{slug(field)}" for field in self.fields for prefix in ("B", "I")
        ]

    def describe(self) -> DatasetMeta:
        """
        Returns the label vocabulary and task

        Returns:
            Metadata with one B and one I label per field.
        """
        return DatasetMeta(
            source=self.source,
            task="token",
            label_names=self.label_names,
            notes=(
                "RealKIE (arXiv 2403.20101), CC BY-NC 4.0. Labels are character spans over "
                "the release's OCR text, mapped onto every OCR word they overlap, so entity "
                "boundaries are exact rather than reconstructed. Each page keeps its own 0-1000 "
                "grid, and page order is carried by word order alone."
            ),
        )

    def load(self, split: str, cache_dir: str | None = None) -> Iterator[Document]:
        """
        Yields one split's documents in the canonical schema

        Args:
            split: One of the keys of `SPLITS`.
            cache_dir: Ignored, as the release lives under the raw directory.

        Yields:
            One Document per document with any OCR text.
        """
        del cache_dir
        # The OCR text of a long report overruns csv's default field limit
        csv.field_size_limit(sys.maxsize)
        empty = 0
        with (ROOT / self.name / f"{SPLITS[split]}.csv").open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                document = self._document(row)
                if document is None:
                    empty += 1
                    continue
                yield document
        if empty:
            logger.info("%s/%s: %d documents without OCR words skipped", self.name, split, empty)

    def _document(self, row: dict[str, str]) -> Document | None:
        """
        Assembles one document from its CSV row and its OCR file

        Args:
            row: The CSV row, holding the span labels and the OCR path.

        Returns:
            The Document, or None when the OCR holds no words.

        Raises:
            ValueError: If a span carries a label outside the dataset's field list.
        """
        index = {field: position for position, field in enumerate(self.fields)}
        spans = sorted(json.loads(row["labels"]), key=lambda span: (span["start"], span["end"]))
        for span in spans:
            if span["label"] not in index:
                raise ValueError(f"{self.name}: unknown label {span['label']!r}")
        with gzip.open(ROOT / row["ocr"], "rt", encoding="utf-8") as handle:
            pages = json.load(handle)

        words: list[str] = []
        boxes: list[list[int]] = []
        tags: list[int] = []
        cursor, previous = 0, None
        for page in pages:
            size = page["pages"][0]["size"]
            for token in page["tokens"]:
                text = token["text"].strip()
                if not text:
                    continue
                start, end = token["doc_offset"]["start"], token["doc_offset"]["end"]
                while cursor < len(spans) and spans[cursor]["end"] <= start:
                    cursor += 1
                span = cursor if cursor < len(spans) and spans[cursor]["start"] < end else None
                if span is None:
                    tags.append(0)
                else:
                    field = index[spans[span]["label"]]
                    tags.append(1 + 2 * field + (span == previous))
                previous = span
                words.append(text)
                boxes.append(_box(token["position"], size))
        if not words:
            return None
        return Document(
            doc_id=f"{self.source}-{Path(row['document_path']).stem}",
            source=self.source,
            words=words,
            bboxes=boxes,
            page_width=COORDINATE_SCALE,
            page_height=COORDINATE_SCALE,
            word_labels=tags,
        )


def _box(position: dict[str, Any], size: dict[str, Any]) -> list[int]:
    """
    Places a pixel box on its page's 0-1000 grid, corners put in order

    Args:
        position: The OCR token's `left`, `top`, `right` and `bottom` in pixels.
        size: The page's `width` and `height` in pixels.

    Returns:
        The normalized box as (x0, y0, x1, y1).
    """
    x0, x1 = sorted((position["left"], position["right"]))
    y0, y1 = sorted((position["top"], position["bottom"]))
    return normalize_box([x0, y0, x1, y1], size["width"], size["height"])


FCC = RealKie("fcc_invoices")
NDA = RealKie("nda")
CHARITIES = RealKie("charities")
