"""
Pre-pack pages into full sequence length to avoid wasting GPU time

The time saved scales up with context length
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from math import ceil
from typing import Any

# Boxes for the separator between two packed pages, and for trailing padding
SEPARATOR_BOX = [1000, 1000, 1000, 1000]
PAD_BOX = [0, 0, 0, 0]


def packing_efficiency(page_tokens: Iterable[int], max_length: int) -> float:
    """
    Reports the speedup packing buys over overflow chunking, from a length distribution

    The comparison is against what the pipeline actually does without packing, which is to chunk an
    over-length page into several full windows rather than truncating it. A page of 900 tokens at
    512 costs two windows, not one, and the first of those is completely full. Taking the mean page
    length and dividing it into `max_length` ignores that and overstates the gain, which is the
    error behind the retracted two-day estimate.

    Args:
        page_tokens: Token count of every page, before padding or chunking.
        max_length: Sequence length being trained at.

    Returns:
        The ratio of chunked cost to packed cost, so 2.0 means packing halves the work.

    Raises:
        ValueError: If `max_length` is not positive, or no page has any tokens.
    """
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    real = 0
    chunked = 0
    for tokens in page_tokens:
        if tokens <= 0:
            continue
        real += tokens
        chunked += max_length * ceil(tokens / max_length)
    if real == 0:
        raise ValueError("no page carried any tokens")
    return chunked / real


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

        # Leave over-length pages independent to avoid contamination
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
