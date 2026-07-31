"""
Packing pages into full sequences, so padding stops burning GPU time

Pretraining pads every chunk to `max_length`. Pages average roughly 380 tokens, so at 512 about a
quarter of every batch is padding and at 1024 nearly two thirds of it is. That padding costs the
same attention and MLP work as real text and teaches the model nothing.

Packing concatenates consecutive pages until a sequence is full, so almost every position carries
a real token. The measured saving is the padding fraction: about 1.35x at 512 and 2.7x at 1024,
which is the difference between a five day pretraining run and a two day one.

Pages are separated by the tokenizer's separator, and each page keeps its own boxes. Attention is
not masked between packed pages, which is the standard trade in pretraining pipelines: a rare
spurious cross-page attention costs far less than throwing away two thirds of the compute.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

# Boxes for the separator between two packed pages, and for trailing padding.
SEPARATOR_BOX = [1000, 1000, 1000, 1000]
PAD_BOX = [0, 0, 0, 0]


def packing_efficiency(page_tokens: float, max_length: int) -> float:
    """
    Reports the speedup packing buys at a given page length and sequence length

    Args:
        page_tokens: Mean tokens per page.
        max_length: Sequence length being trained at.

    Returns:
        The ratio of padded cost to packed cost, so 2.0 means packing halves the work.

    Raises:
        ValueError: If either argument is not positive.
    """
    if page_tokens <= 0 or max_length <= 0:
        raise ValueError("page_tokens and max_length must both be positive")
    return max_length / min(page_tokens, max_length)


def pack_encodings(
    encodings: Iterable[Mapping[str, Any]],
    max_length: int,
    separator_id: int,
    pad_id: int,
) -> Iterator[dict[str, list[Any]]]:
    """
    Concatenates per-page encodings into sequences of exactly `max_length` tokens

    A page longer than `max_length` is emitted on its own, truncated, rather than being allowed to
    straddle two packed sequences and lose its box alignment. The final partial sequence is padded
    and emitted, so no page is silently dropped.

    Args:
        encodings: Per-page dicts holding `input_ids` and `bbox`, unpadded and untruncated.
        max_length: Target sequence length.
        separator_id: Token id placed between two packed pages.
        pad_id: Token id used for trailing padding in the final sequence.

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
            raise ValueError(
                f"page has {len(page_ids)} tokens but {len(page_boxes)} boxes; "
                f"the tokenizer and the box expansion disagree"
            )
        if not page_ids:
            continue

        # An over-length page is its own sequence. Splitting it across a pack boundary would put
        # half a page next to an unrelated one, which is worse than truncating.
        if len(page_ids) >= max_length:
            if ids:
                yield _finish(ids, boxes, max_length, pad_id)
                ids, boxes = [], []
            yield _finish(page_ids[:max_length], page_boxes[:max_length], max_length, pad_id)
            continue

        if ids and len(ids) + 1 + len(page_ids) > max_length:
            yield _finish(ids, boxes, max_length, pad_id)
            ids, boxes = [], []

        if ids:
            ids.append(separator_id)
            boxes.append(list(SEPARATOR_BOX))
        ids.extend(page_ids)
        boxes.extend(page_boxes)

    if ids:
        yield _finish(ids, boxes, max_length, pad_id)


def _finish(
    ids: list[int],
    boxes: list[list[int]],
    max_length: int,
    pad_id: int,
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
    real = len(ids)
    short = max_length - real
    return {
        "input_ids": ids + [pad_id] * short,
        "attention_mask": [1] * real + [0] * short,
        "bbox": boxes + [list(PAD_BOX) for _ in range(short)],
    }
