"""
Turning canonical parquet into model batches: tokenize, chunk or pack, then mask

Long pages split into overlapping chunks for fine-tuning, and pretraining packs pages end to end
The stock MLM collator drops columns it does not know, so this one masks text and keeps the boxes
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from datasets import (
    Dataset,
    Features,
    List,
    Value,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src.data import DatasetMeta
from src.model import ModernLayoutLmProcessor

logger = logging.getLogger(__name__)

# Token overlap between consecutive chunks of an over-length page
DEFAULT_STRIDE = 128

# Which page each chunk came from, so scoring can stitch a chunked page back into one sequence
SAMPLE_INDEX = "sample_index"

# Packed columns at their narrowest, as bbox in int64 tripled the pack
PACK_FEATURES = Features(
    {
        "input_ids": List(Value("int32")),
        "attention_mask": List(Value("int8")),
        "bbox": List(List(Value("int16"), length=4)),
    }
)

# Boxes for the separator between two packed pages, and for trailing padding
SEPARATOR_BOX = [1000, 1000, 1000, 1000]
PAD_BOX = [0, 0, 0, 0]

# Keep BERT's 80/10/10 split of selected tokens as ModernBERT raises only the overall rate
REPLACE_FRACTION = 0.8
RANDOM_FRACTION = 0.5
IGNORE_INDEX = -100


def load_split(processed_dir: Path, name: str, split: str) -> Dataset:
    """
    Loads one split of a processed dataset from its parquet shards

    Args:
        processed_dir: Root of the processed tree.
        name: The dataset name.
        split: The split name.

    Returns:
        The canonical dataset, untokenized.

    Raises:
        FileNotFoundError: If the split has no parquet shards.
    """
    shards = sorted((processed_dir / name / split).glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet in {processed_dir / name / split}, build it first")
    return load_dataset("parquet", data_files=[str(path) for path in shards], split="train")


def tokenize_split(
    dataset: Dataset,
    processor: ModernLayoutLmProcessor,
    max_length: int,
    stride: int = DEFAULT_STRIDE,
    num_proc: int | None = None,
    keep_sample_index: bool = False,
) -> Dataset:
    """
    Tokenizes a labeled split into padded model inputs, chunking pages that overrun

    Each chunk keeps its own slice of the word labels.

    Args:
        dataset: A canonical split as loaded by `load_split`.
        processor: The processor pairing a tokenizer with box alignment.
        max_length: Tokens per chunk.
        stride: Token overlap between consecutive chunks.
        num_proc: Worker processes for the map, or None for one.
        keep_sample_index: Emit a `SAMPLE_INDEX` column naming each chunk's source page.

    Returns:
        A dataset of `input_ids`, `attention_mask`, `bbox` and `labels`.
    """

    def encode(batch: dict[str, list[Any]], indices: list[int]) -> dict[str, list[Any]]:
        """Encodes one batch of canonical rows into chunked model inputs."""
        encoding = processor(
            words=batch["words"],
            boxes=batch["bboxes"],
            word_labels=batch["word_labels"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_overflowing_tokens=True,
            stride=stride,
        )
        mapping = encoding["overflow_to_sample_mapping"]
        result = {key: encoding[key] for key in ("input_ids", "attention_mask", "bbox", "labels")}
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
) -> tuple[dict[str, Dataset], DatasetMeta]:
    """
    Loads and tokenizes every split of one downstream dataset

    Evaluation splits abut rather than overlap, as shared tokens would be scored twice.

    Args:
        processed_dir: Root of the processed tree.
        name: The dataset name.
        processor: The processor to encode with.
        max_length: Tokens per chunk.
        stride: Token overlap between training chunks.
        num_proc: Worker processes for tokenization.

    Returns:
        A mapping from split name to tokenized dataset, plus the dataset metadata.
    """
    meta = DatasetMeta.read(processed_dir / name)
    splits: dict[str, Dataset] = {}
    for split in meta.splits:
        scoring = split != "train"
        splits[split] = tokenize_split(
            load_split(processed_dir, name, split),
            processor,
            max_length,
            0 if scoring else stride,
            num_proc,
            scoring,
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
    batch_rows: int = 2000,
) -> Dataset:
    """
    Tokenizes and packs pretraining corpora into full-length sequences, cached on disk

    A capped corpus is subsampled at random, as its shards are grouped by collection.
    A row longer than the window becomes consecutive windows rather than losing its tail.

    Args:
        processed_dir: Root of the processed tree.
        names: Corpus names to concatenate, in mixing order.
        split: The split to build.
        processor: The processor to encode with.
        max_length: Target packed sequence length.
        cache_dir: Where to store the packed arrow dataset.
        num_proc: Worker processes for tokenizing and packing.
        max_pages: Cap on pages per corpus.
        seed: Seed for that subsample, so the cache key and the pages agree.
        batch_rows: Rows per packing call, each held in memory by one process.

    Returns:
        A dataset of `input_ids`, `attention_mask` and `bbox`, every row `max_length` long.

    Raises:
        ValueError: If `names` is empty.
    """
    if not names:
        raise ValueError("Packing needs at least one corpus")
    target = cache_dir / f"packed-{split}-{max_length}-{'_'.join(names)}-{max_pages or 'all'}"
    if (target / "dataset_info.json").is_file():
        logger.info("reusing cached pack at %s", target)
        return cast(Dataset, load_from_disk(str(target)))
    separator, pad = processor.tokenizer.sep_token_id, processor.tokenizer.pad_token_id

    def packer(batch: dict[str, list[Any]], indices: list[int]) -> dict[str, list[Any]]:
        """Encodes one block of rows without padding and packs them in a seeded order."""
        encoding = processor(
            words=batch["words"],
            boxes=batch["bboxes"],
            max_length=max_length,
            padding=False,
            truncation=True,
            return_overflowing_tokens=True,
        )
        # Storage order keeps a document's pages together, so the block is mixed before packing
        order = np.random.default_rng(seed + indices[0]).permutation(len(indices))
        ids, boxes = encoding["input_ids"], encoding["bbox"]
        rows = chunk_order(encoding["overflow_to_sample_mapping"], order)
        pages = ({"input_ids": ids[i], "bbox": boxes[i]} for i in rows)
        sequences = list(pack_encodings(pages, max_length, separator, pad))
        return {key: [s[key] for s in sequences] for key in ("input_ids", "attention_mask", "bbox")}

    parts: list[Dataset] = []
    for name in names:
        raw = load_split(processed_dir, name, split)
        if max_pages is not None and max_pages < len(raw):
            # The pages a seeded shuffle would cut, kept in storage order so the disk reads stream
            raw = raw.select(np.sort(np.random.default_rng(seed).permutation(len(raw))[:max_pages]))
        parts.append(
            raw.map(
                packer,
                batched=True,
                batch_size=batch_rows,
                with_indices=True,
                remove_columns=raw.column_names,
                features=PACK_FEATURES,
                num_proc=num_proc,
                desc=f"packing {name}/{split}",
            )
        )
    packed = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    packed.save_to_disk(str(target))
    logger.info("packed %d sequences into %s", len(packed), target)
    return cast(Dataset, packed)


def chunk_order(sample_map: list[int], order: Iterable[int]) -> list[int]:
    """
    Lists encoded chunks by their source row in the given order, a row's chunks kept in sequence

    Args:
        sample_map: The source row of each chunk, as the tokenizer's overflow mapping gives it.
        order: Source rows in the order they should be packed.

    Returns:
        Chunk indices, every chunk of one row adjacent and in its original order.
    """
    chunks: dict[int, list[int]] = {}
    for chunk, sample in enumerate(sample_map):
        chunks.setdefault(sample, []).append(chunk)
    return [chunk for sample in order for chunk in chunks.get(sample, [])]


def pack_encodings(
    encodings: Iterable[Mapping[str, Any]],
    max_length: int,
    separator_id: int,
    pad_id: int,
) -> Iterator[dict[str, list[Any]]]:
    """
    Concatenates unpadded pages into sequences of exactly `max_length` tokens

    A page at or over `max_length` is truncated and emitted alone.
    The final partial sequence is padded, so no page is dropped.

    Args:
        encodings: Per-page dicts holding `input_ids` and `bbox`.
        max_length: Target sequence length.
        separator_id: Token id placed between two packed pages.
        pad_id: Token id used for trailing padding.

    Yields:
        Dicts of `input_ids`, `attention_mask` and `bbox`, each exactly `max_length` long.

    Raises:
        ValueError: If a page's token and box counts disagree.
    """
    ids: list[int] = []
    boxes: list[list[int]] = []
    for encoding in encodings:
        page_ids = list(encoding["input_ids"])
        page_boxes = [list(box) for box in encoding["bbox"]]
        if len(page_ids) != len(page_boxes):
            raise ValueError(f"page has {len(page_ids)} tokens but {len(page_boxes)} boxes")
        if not page_ids:
            continue
        if len(page_ids) >= max_length:
            if ids:
                yield _padded(ids, boxes, max_length, pad_id)
                ids, boxes = [], []
            yield _padded(page_ids[:max_length], page_boxes[:max_length], max_length, pad_id)
            continue
        if ids and len(ids) + 1 + len(page_ids) > max_length:
            yield _padded(ids, boxes, max_length, pad_id)
            ids, boxes = [], []
        if ids:
            ids.append(separator_id)
            boxes.append(list(SEPARATOR_BOX))
        ids.extend(page_ids)
        boxes.extend(page_boxes)
    if ids:
        yield _padded(ids, boxes, max_length, pad_id)


def _padded(
    ids: list[int], boxes: list[list[int]], max_length: int, pad_id: int
) -> dict[str, list[Any]]:
    """
    Pads one packed sequence out to the target length

    Args:
        ids: Token ids accumulated so far.
        boxes: Boxes accumulated so far, one per token.
        max_length: Target sequence length.
        pad_id: Token id for padding.

    Returns:
        A dict of `input_ids`, `attention_mask` and `bbox`, each `max_length` long.
    """
    short = max_length - len(ids)
    return {
        "input_ids": ids + [pad_id] * short,
        "attention_mask": [1] * len(ids) + [0] * short,
        "bbox": boxes + [list(PAD_BOX) for _ in range(short)],
    }


@dataclass
class LayoutMlmCollator:
    """
    Masks tokens for layout MLM while passing boxes through unchanged

    Args:
        tokenizer: Supplies the mask token id, the vocabulary size and the special token mask.
        mlm_probability: Fraction of tokens selected for prediction, 0.3 in ModernBERT's recipe.
    """

    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 0.3

    def __post_init__(self) -> None:
        if not isinstance(self.tokenizer.mask_token_id, int):
            raise ValueError("The tokenizer has no single mask token, so MLM is impossible")
        self.mask_token_id = self.tokenizer.mask_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Collates encoded rows into a masked batch

        Args:
            features: Encoded rows, each holding `input_ids`, `attention_mask` and `bbox`.

        Returns:
            `input_ids`, `attention_mask`, `bbox` and `labels`, the last `IGNORE_INDEX` off masks.

        Raises:
            KeyError: If a row has no `bbox`, which would train a layout model on no layout.
            ValueError: If the rows are ragged.
        """
        if any("bbox" not in feature for feature in features):
            raise KeyError("A feature has no bbox, so the tokenizing step dropped the boxes")
        batch = {}
        for key in ("input_ids", "attention_mask", "bbox"):
            if len({len(feature[key]) for feature in features}) != 1:
                raise ValueError(f"Column {key!r} has ragged lengths, pad first")
            batch[key] = torch.tensor([feature[key] for feature in features], dtype=torch.long)
        batch["input_ids"], batch["labels"] = self.mask_tokens(
            batch["input_ids"], batch["attention_mask"]
        )
        return batch

    def mask_tokens(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Chooses tokens to predict and corrupts them, never selecting special tokens or padding

        Of the selected tokens 80% become the mask, half the rest a random token, the rest stay.

        Args:
            input_ids: Token ids of shape (batch, seq_len).
            attention_mask: Mask of shape (batch, seq_len), zero on padding.

        Returns:
            The corrupted input ids, and labels that are `IGNORE_INDEX` off the selected positions.
        """
        labels = input_ids.clone()
        probability = torch.full(labels.shape, self.mlm_probability)
        special = [
            self.tokenizer.get_special_tokens_mask(row, already_has_special_tokens=True)
            for row in labels.tolist()
        ]
        probability.masked_fill_(torch.tensor(special, dtype=torch.bool), value=0.0)
        probability.masked_fill_(attention_mask == 0, value=0.0)
        selected = torch.bernoulli(probability).bool()
        labels[~selected] = IGNORE_INDEX
        replaced = torch.bernoulli(torch.full(labels.shape, REPLACE_FRACTION)).bool() & selected
        input_ids = input_ids.clone()
        input_ids[replaced] = self.mask_token_id
        randomized = torch.bernoulli(torch.full(labels.shape, RANDOM_FRACTION)).bool()
        randomized &= selected & ~replaced
        random_ids = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[randomized] = random_ids[randomized]
        return input_ids, labels
