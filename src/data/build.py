"""
Turning a raw source into validated canonical parquet

One entrypoint drives any adapter: stream, validate, write parquet, write the sidecar
Validation is deliberately fatal as a source producing a misaligned document is not understood yet
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, cast

from .schema import DatasetMeta, Document, write_documents
from .sources import get_source

logger = logging.getLogger(__name__)

# Keeps a single parquet file readable while avoiding thousands of tiny ones
SHARD_SIZE = 20_000


@dataclass
class BuildReport:
    """
    What a build actually produced

    Args:
        source: The dataset name.
        splits: Rows written per split.
        skipped: Documents dropped for having no words, per split.
        label_counts: How often each label id appears, for spotting a collapsed label scheme.
    """

    source: str
    splits: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)
    label_counts: Counter[int] = field(default_factory=Counter)

    def summary(self) -> str:
        """
        Renders a one-line-per-split human summary

        Returns:
            A printable multi-line string.
        """
        lines = [f"{self.source}:"]
        for split, count in self.splits.items():
            skipped = self.skipped.get(split, 0)
            lines.append(
                f"  {split}: {count} documents" + (f", {skipped} skipped" if skipped else "")
            )
        lines.append(f"  labels in use: {len(self.label_counts)}")
        return "\n".join(lines)


def build_source(
    name: str,
    processed_dir: Path,
    cache_dir: Path | None = None,
    limit: int | None = None,
    workers: int = 1,
) -> BuildReport:
    """
    Builds every split of one source into canonical parquet

    A source exposing `parts` and `load_part` builds each part in its own process

    Args:
        name: The dataset name, a key of `SOURCES`.
        processed_dir: Root of the processed tree, with output under `processed_dir / name`.
        cache_dir: Hugging Face cache directory, or None for the default.
        limit: Stop after this many documents per split for a smoke test, which builds serially.
        workers: Processes for a source with parts, each holding one part in memory at a time.

    Returns:
        A report of what was written.

    Raises:
        KeyError: If the source is not registered.
        ValueError: If any document fails validation.
    """
    source = get_source(name)
    meta = source.describe()
    report = BuildReport(source=name)
    parts = getattr(source, "parts", None)

    for split in source.SPLITS:
        logger.info("building %s/%s", name, split)
        split_dir = processed_dir / name / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for stale in split_dir.glob("*.parquet"):
            stale.unlink()

        if workers > 1 and limit is None and parts is not None:
            jobs = [
                (name, part, split_dir / f"{index:03d}") for index, part in enumerate(parts(split))
            ]
            # A fresh process per part returns each part's memory before the next one loads
            with ProcessPoolExecutor(workers, max_tasks_per_child=1) as pool:
                results = list(pool.map(_build_part, jobs))
        else:
            documents = source.load(split, cache_dir=str(cache_dir) if cache_dir else None)
            results = [_write(islice(documents, limit), split_dir / "000")]

        report.splits[split] = sum(written for written, _, _ in results)
        report.skipped[split] = sum(skipped for _, skipped, _ in results)
        for _, _, counts in results:
            report.label_counts.update(counts)

    meta.splits = report.splits
    meta.write(processed_dir / name)
    _warn_on_collapsed_labels(meta, report)
    return report


def _build_part(job: tuple[str, Any, Path]) -> tuple[int, int, Counter[int]]:
    """
    Builds one part of a split inside a worker process

    Args:
        job: The source name, the part as the source's `parts` listed it, and the file prefix.

    Returns:
        Rows written, documents skipped, and label counts for this part.
    """
    name, part, prefix = job
    return _write(cast(Any, get_source(name)).load_part(part), prefix)


def _write(documents: Iterator[Document], prefix: Path) -> tuple[int, int, Counter[int]]:
    """
    Validates documents and writes them as numbered parquet files next to `prefix`

    Args:
        documents: Canonical documents for one split or one part of it.
        prefix: Path stem the numbered files share, such as `split_dir / "003"`.

    Returns:
        Rows written, documents skipped for having no words, and label counts.

    Raises:
        ValueError: If a document fails its invariants.
    """
    written, skipped, counts = 0, 0, Counter[int]()
    buffer: list[Document] = []

    def flush() -> int:
        path = prefix.with_name(f"{prefix.name}-{written // SHARD_SIZE:05d}.parquet")
        return write_documents(buffer, path)

    for document in documents:
        if not document.words:
            skipped += 1
            continue
        document.validate()
        counts.update(document.word_labels or [])
        if document.doc_label is not None:
            counts[document.doc_label] += 1
        buffer.append(document)
        if len(buffer) == SHARD_SIZE:
            written += flush()
            buffer = []
    if buffer:
        written += flush()
    return written, skipped, counts


def _warn_on_collapsed_labels(meta: DatasetMeta, report: BuildReport) -> None:
    """
    Logs a warning when most of the declared label vocabulary never appears

    A misread label scheme lands every word on one or two ids, invisible until the F1 is nonsense.

    Args:
        meta: The dataset metadata, holding the declared label names.
        report: The build report, holding observed label frequencies.
    """
    declared, used = len(meta.label_names), len(report.label_counts)
    if used <= 2 < declared:
        logger.warning(
            "%s uses %d of %d declared labels, check its scheme", meta.source, used, declared
        )
