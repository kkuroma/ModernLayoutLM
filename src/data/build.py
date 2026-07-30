"""
Turning a raw source into validated canonical parquet

One entrypoint, `build_source`, drives any adapter: stream its documents, validate every one,
write parquet per split, then write the label sidecar. Validation is deliberately fatal. A source
that produces a misaligned document is a source we do not understand yet, and training on it would
waste a GPU day to produce a number nobody can interpret.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .schema import DatasetMeta, Document, write_documents
from .sources import get_source

logger = logging.getLogger(__name__)

# Documents per parquet file. Keeps single files readable while avoiding thousands of tiny ones.
SHARD_SIZE = 20_000


@dataclass
class BuildReport:
    """
    What a build actually produced

    Args:
        source: The dataset name.
        splits: Rows written per split.
        skipped: Documents the adapter declined to emit, per split.
        label_counts: How often each label id appears, for spotting a collapsed label scheme.
    """

    source: str
    splits: dict[str, int]
    skipped: dict[str, int]
    label_counts: dict[int, int]

    def summary(self) -> str:
        """
        Renders a one-line-per-split human summary

        Returns:
            A printable multi-line string.
        """
        lines = [f"{self.source}:"]
        for split, count in self.splits.items():
            skipped = self.skipped.get(split, 0)
            suffix = f", {skipped} skipped" if skipped else ""
            lines.append(f"  {split}: {count} documents{suffix}")
        used = len(self.label_counts)
        lines.append(f"  labels in use: {used}")
        return "\n".join(lines)


def build_source(
    name: str,
    processed_dir: Path,
    cache_dir: Path | None = None,
    limit: int | None = None,
) -> BuildReport:
    """
    Builds every split of one source into canonical parquet

    Args:
        name: The dataset name, a key of `SOURCES`.
        processed_dir: Root of the processed tree. Output goes to `processed_dir / name`.
        cache_dir: Hugging Face cache directory, or None for the default.
        limit: Stop after this many documents per split. For smoke tests.

    Returns:
        A report of what was written.

    Raises:
        KeyError: If the source is not registered.
        ValueError: If any document fails validation.
    """
    source = get_source(name)
    meta = source.describe()
    out_dir = processed_dir / name

    splits: dict[str, int] = {}
    skipped: dict[str, int] = {}
    label_counts: dict[int, int] = {}

    for split in source.SPLITS:
        logger.info("building %s/%s", name, split)
        documents = source.load(split, cache_dir=str(cache_dir) if cache_dir else None)
        written, dropped = _write_split(
            documents=_limited(documents, limit),
            split_dir=out_dir / split,
            label_counts=label_counts,
        )
        splits[split] = written
        skipped[split] = dropped

    meta.splits = splits
    meta.write(out_dir)
    report = BuildReport(source=name, splits=splits, skipped=skipped, label_counts=label_counts)
    _warn_on_collapsed_labels(meta, report)
    return report


def _limited(documents: Iterator[Document], limit: int | None) -> Iterator[Document]:
    """
    Truncates a document stream

    Args:
        documents: The source stream.
        limit: Maximum documents to pass through, or None for all.

    Yields:
        At most `limit` documents.
    """
    if limit is None:
        yield from documents
        return
    for index, document in enumerate(documents):
        if index >= limit:
            return
        yield document


def _write_split(
    documents: Iterator[Document],
    split_dir: Path,
    label_counts: dict[int, int],
) -> tuple[int, int]:
    """
    Validates and shards one split to parquet

    Args:
        documents: Canonical documents for this split.
        split_dir: Directory to write shards into. Cleared of stale shards first.
        label_counts: Accumulator for label frequency, mutated in place.

    Returns:
        A (written, skipped) pair, where skipped counts documents dropped for being empty.

    Raises:
        ValueError: If a document fails its invariants.
    """
    split_dir.mkdir(parents=True, exist_ok=True)
    for stale in split_dir.glob("*.parquet"):
        stale.unlink()

    written = 0
    skipped = 0
    shard_index = 0
    buffer: list[Document] = []

    for document in documents:
        if not document.words:
            skipped += 1
            continue
        document.validate()
        _count_labels(document, label_counts)
        buffer.append(document)
        if len(buffer) >= SHARD_SIZE:
            written += write_documents(buffer, split_dir / f"{shard_index:05d}.parquet")
            shard_index += 1
            buffer = []

    if buffer:
        written += write_documents(buffer, split_dir / f"{shard_index:05d}.parquet")
    return written, skipped


def _count_labels(document: Document, label_counts: dict[int, int]) -> None:
    """
    Adds one document's labels to the running frequency count

    Args:
        document: The document to count.
        label_counts: Accumulator, mutated in place.
    """
    if document.word_labels is not None:
        for label in document.word_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
    if document.doc_label is not None:
        label_counts[document.doc_label] = label_counts.get(document.doc_label, 0) + 1


def _warn_on_collapsed_labels(meta: DatasetMeta, report: BuildReport) -> None:
    """
    Logs a warning when most of the declared label vocabulary never appears

    A label scheme misread at the source shows up here: every word lands on one or two ids while
    the rest of the vocabulary sits empty. Cheap to check, and it catches the failure that is
    otherwise invisible until the F1 comes back meaningless.

    Args:
        meta: The dataset metadata, holding the declared label names.
        report: The build report, holding observed label frequencies.
    """
    declared = len(meta.label_names)
    if declared == 0:
        return
    used = len(report.label_counts)
    if used <= 2 < declared:
        logger.warning(
            "%s uses only %d of %d declared labels. Check the source label scheme.",
            meta.source,
            used,
            declared,
        )
