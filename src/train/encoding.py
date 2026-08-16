"""
Turning canonical parquet into tokenized model inputs

The first stage that may touch both src.data and src.model, so the processor lives here
Long pages are split into overlapping chunks so a long receipt keeps its tail
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk

from src.data import DatasetMeta
from src.model import ModernLayoutLmProcessor
from src.train.packing import pack_encodings

logger = logging.getLogger(__name__)

# Drop everything else or a stray column becomes a forward argument
MODEL_COLUMNS = ["input_ids", "attention_mask", "bbox", "labels"]

# Token overlap between consecutive chunks of an over-length page
DEFAULT_STRIDE = 128

# Which page each chunk came from, so scoring can stitch a chunked page back into one sequence
SAMPLE_INDEX = "sample_index"

"""
Pages handed to one packing call

Packing is sequential by nature, so it is parallelized by packing blocks of pages independently
rather than by splitting a block. Each block flushes a partial sequence at its end, which is the
whole cost of the parallelism: at this block size that is roughly one padded tail per eight hundred
packed sequences, under a tenth of a percent, against a single-threaded pass that left fifteen
cores idle for forty minutes.
"""
PACK_BATCH_PAGES = 2000


def load_split(processed_dir: Path, name: str, split: str) -> Dataset:
    """
    Loads one split of a processed dataset from parquet

    Args:
        processed_dir: Root of the processed tree.
        name: The dataset name.
        split: The split name.

    Returns:
        The raw canonical dataset, untokenized.

    Raises:
        FileNotFoundError: If the split has no parquet shards.
    """
    shards = sorted((processed_dir / name / split).glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"No parquet in {processed_dir / name / split}. "
            f"Run scripts/data/build_datasets.py first."
        )
    return load_dataset("parquet", data_files=[str(path) for path in shards], split="train")


def read_meta(processed_dir: Path, name: str) -> DatasetMeta:
    """
    Reads a processed dataset's label sidecar

    Args:
        processed_dir: Root of the processed tree.
        name: The dataset name.

    Returns:
        The dataset metadata.
    """
    return DatasetMeta.read(processed_dir / name)


def tokenize_split(
    dataset: Dataset,
    processor: ModernLayoutLmProcessor,
    task: str,
    max_length: int,
    stride: int = DEFAULT_STRIDE,
    num_proc: int | None = None,
    keep_sample_index: bool = False,
) -> Dataset:
    """
    Tokenizes a canonical split into model inputs, chunking pages that overrun

    A page longer than `max_length` becomes several rows. For token classification each chunk keeps
    its own slice of the labels. For sequence classification every chunk inherits the page's label,
    so a long document contributes several training examples and is scored per chunk.

    Args:
        dataset: A canonical split as loaded by `load_split`.
        processor: The processor pairing a tokenizer with box alignment.
        task: Either "token", "sequence" or "pretrain".
        max_length: Tokens per chunk.
        stride: Token overlap between consecutive chunks.
        num_proc: Worker processes for the map, or None for one.
        keep_sample_index: Emit a `SAMPLE_INDEX` column naming each chunk's source page.

    Returns:
        A dataset whose columns are exactly `MODEL_COLUMNS`, minus labels for pretraining.

    Raises:
        ValueError: If `task` is not recognized.
    """
    if task not in {"token", "sequence", "pretrain"}:
        raise ValueError(f"Unknown task {task!r}")

    def encode(batch: dict[str, list[Any]], indices: list[int]) -> dict[str, list[Any]]:
        """Encodes one batch of canonical rows into chunked model inputs."""
        word_labels = batch["word_labels"] if task == "token" else None
        encoding = processor(
            words=batch["words"],
            boxes=batch["bboxes"],
            word_labels=word_labels,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_overflowing_tokens=True,
            stride=stride,
        )
        mapping = encoding["overflow_to_sample_mapping"]
        result: dict[str, list[Any]] = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "bbox": encoding["bbox"],
        }
        if task == "token":
            result["labels"] = encoding["labels"]
        elif task == "sequence":
            # Label every chunk so long pages are not truncated away
            result["labels"] = [batch["doc_label"][index] for index in mapping]
        if keep_sample_index:
            # The row index in the split, not the offset within this batch
            result[SAMPLE_INDEX] = [indices[index] for index in mapping]
        return result

    return dataset.map(
        encode,
        batched=True,
        batch_size=16,
        with_indices=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc="tokenizing",
    )


def prepare_task_splits(
    processed_dir: Path,
    name: str,
    processor: ModernLayoutLmProcessor,
    max_length: int,
    stride: int = DEFAULT_STRIDE,
    num_proc: int | None = None,
    max_train_pages: int | None = None,
    seed: int = 42,
) -> tuple[dict[str, Dataset], DatasetMeta]:
    """
    Loads and tokenizes every split of one downstream dataset

    Args:
        processed_dir: Root of the processed tree.
        name: The dataset name.
        processor: The processor to encode with.
        max_length: Tokens per chunk.
        stride: Token overlap between chunks.
        num_proc: Worker processes for tokenization.
        max_train_pages: Cap on training pages, applied before tokenizing so the cap bounds the
            tokenization cost too. Evaluation splits are never capped. The subsample is a fixed
            shuffle, so every ablation variant sees exactly the same pages.
        seed: Seed for that subsample.

    Returns:
        A mapping from split name to tokenized dataset, plus the dataset metadata.
    """
    meta = read_meta(processed_dir, name)
    splits: dict[str, Dataset] = {}
    for split in meta.splits:
        raw = load_split(processed_dir, name, split)
        if split == "train" and max_train_pages is not None and max_train_pages < len(raw):
            raw = raw.shuffle(seed=seed).select(range(max_train_pages))
        # Overlapping chunks would score their shared tokens twice, so evaluation splits abut
        scoring = split != "train"
        splits[split] = tokenize_split(
            dataset=raw,
            processor=processor,
            task=meta.task,
            max_length=max_length,
            stride=0 if scoring else stride,
            num_proc=num_proc,
            keep_sample_index=scoring,
        )
    return splits, meta


def build_packed_corpus(
    processed_dir: Path,
    names: list[str],
    split: str,
    processor: ModernLayoutLmProcessor,
    max_length: int,
    cache_dir: Path,
    num_proc: int | None = None,
    max_pages: int | None = None,
    seed: int = 42,
) -> Dataset:
    """
    Tokenizes and packs several pretraining corpora into one dataset, cached on disk

    Packing removes the padding that otherwise dominates pretraining: at 1024 tokens a typical
    page fills barely a third of the window. The result is cached because tokenizing tens of
    millions of pages takes hours, and a multi-day run will be restarted more than once.

    A capped corpus is shuffled before it is cut. Taking the first `max_pages` rows instead would
    take whichever shards happen to sort first, and the shards are grouped by source: PubTables is
    ordered by PMC accession and RVL-CDIP by class, so the head of the file is not a sample of the
    corpus. That matters most for a run that will be stopped before it finishes an epoch.

    Args:
        processed_dir: Root of the processed tree.
        names: Corpus names to concatenate, in mixing order.
        split: The split to build.
        processor: The processor to encode with.
        max_length: Target packed sequence length.
        cache_dir: Where to store the packed arrow dataset.
        num_proc: Worker processes for tokenization.
        max_pages: Cap on pages per corpus, for probes.
        seed: Seed for that subsample, so the cache key and the pages agree.

    Returns:
        A dataset of `input_ids`, `attention_mask` and `bbox`, every row exactly `max_length` long.

    Raises:
        ValueError: If `names` is empty.
    """
    if not names:
        raise ValueError("Packing needs at least one corpus")

    key = f"packed-{split}-{max_length}-{'_'.join(names)}-{max_pages or 'all'}"
    target = cache_dir / key
    if (target / "dataset_info.json").is_file():
        logger.info("reusing cached pack at %s", target)
        return cast(Dataset, load_from_disk(str(target)))

    separator = processor.tokenizer.sep_token_id
    pad = processor.tokenizer.pad_token_id

    def unpadded(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """Encodes a batch of pages without padding, so packing sees their true lengths."""
        encoding = processor(
            words=batch["words"],
            boxes=batch["bboxes"],
            max_length=max_length,
            padding=False,
            truncation=True,
        )
        return {"input_ids": encoding["input_ids"], "bbox": encoding["bbox"]}

    def packer(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """Packs one block of tokenized pages into full-length sequences."""
        pages = cast(
            Iterable[Mapping[str, Any]],
            [
                {"input_ids": ids, "bbox": boxes}
                for ids, boxes in zip(batch["input_ids"], batch["bbox"], strict=True)
            ],
        )
        sequences = list(pack_encodings(pages, max_length, separator, pad))
        return {
            "input_ids": [s["input_ids"] for s in sequences],
            "attention_mask": [s["attention_mask"] for s in sequences],
            "bbox": [s["bbox"] for s in sequences],
        }

    parts: list[Dataset] = []
    for name in names:
        raw = load_split(processed_dir, name, split)
        if max_pages is not None and max_pages < len(raw):
            raw = raw.shuffle(seed=seed).select(range(max_pages))
        tokenized = raw.map(
            unpadded,
            batched=True,
            batch_size=64,
            remove_columns=raw.column_names,
            num_proc=num_proc,
            desc=f"tokenizing {name}/{split}",
        )
        parts.append(
            tokenized.map(
                packer,
                batched=True,
                batch_size=PACK_BATCH_PAGES,
                remove_columns=tokenized.column_names,
                num_proc=num_proc,
                desc=f"packing {name}/{split}",
            )
        )

    packed = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    packed.save_to_disk(str(target))
    logger.info("packed %d sequences into %s", len(packed), target)
    return cast(Dataset, packed)
