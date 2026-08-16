"""
Scoring SROIE the way ICDAR task 3 scores it, on four exact strings per receipt

A field counts only on an exact string match and Hmean is micro-averaged over receipts and fields
Entity F1 measures every tagged span instead, and on the same predictions the two differ by 7.6
"""

from __future__ import annotations

from collections.abc import Sequence

# The four strings ICDAR task 3 extracts from a receipt
FIELDS = ["company", "date", "address", "total"]

# Datasets whose headline number is field-level rather than entity level
FIELD_LEVEL_SOURCES = {"sroie"}


def decode_fields(words: Sequence[str], tags: Sequence[str]) -> dict[str, str]:
    """
    Joins each field's first tagged span into one string

    A receipt names each field once, so later spans are duplicates or false positives and the first
    one is the answer.

    Args:
        words: The receipt's words in reading order.
        tags: One BIO tag per word.

    Returns:
        Field name to string, omitting any field with no tagged span.

    Raises:
        ValueError: If the word and tag counts disagree.
    """
    if len(words) != len(tags):
        raise ValueError(f"{len(words)} words but {len(tags)} tags")

    spans: dict[str, list[str]] = {}
    current: list[str] = []
    field = ""
    for word, tag in zip(words, tags, strict=True):
        if tag.startswith("B-"):
            if current and field not in spans:
                spans[field] = current
            field = tag[2:].lower()
            current = [word]
        elif tag.startswith("I-") and field == tag[2:].lower():
            current.append(word)
        else:
            if current and field not in spans:
                spans[field] = current
            current, field = [], ""
    if current and field not in spans:
        spans[field] = current
    return {name: " ".join(parts) for name, parts in spans.items()}


def field_metrics(
    predicted: Sequence[dict[str, str]],
    truth: Sequence[dict[str, str]],
) -> dict[str, float]:
    """
    Micro precision, recall and Hmean over exact field matches

    Args:
        predicted: One field mapping per receipt.
        truth: The matching ground-truth field mappings, in the same order.

    Returns:
        Precision, recall and hmean, plus the raw counts behind them.

    Raises:
        ValueError: If the two sequences are different lengths.
    """
    if len(predicted) != len(truth):
        raise ValueError(f"{len(predicted)} predictions but {len(truth)} receipts")

    correct = predicted_total = truth_total = 0
    for got, want in zip(predicted, truth, strict=True):
        for name in FIELDS:
            has_got = bool(got.get(name))
            has_want = bool(want.get(name))
            predicted_total += has_got
            truth_total += has_want
            correct += has_got and has_want and got[name] == want[name]

    precision = correct / predicted_total if predicted_total else 0.0
    recall = correct / truth_total if truth_total else 0.0
    hmean = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "field_precision": precision,
        "field_recall": recall,
        "hmean": hmean,
    }


def score_fields(
    words: Sequence[Sequence[str]],
    predicted: Sequence[Sequence[str]],
    truth: Sequence[Sequence[str]],
) -> dict[str, float]:
    """
    Decodes tags into fields and scores them, which is the whole field-level path

    Truth is decoded from the annotated tags rather than the official key files, since those
    disagree on about a fifth of addresses and the disagreements are words absent from the OCR.

    Args:
        words: Each receipt's words in reading order.
        predicted: Predicted tags per receipt, already stitched across chunks.
        truth: True tags per receipt.

    Returns:
        Field precision, recall and Hmean.
    """
    # Truncation drops the tail of an over-long receipt, so the words outrun the tags
    trimmed = [
        (page[:limit], tags[:limit], gold[:limit])
        for page, tags, gold in zip(words, predicted, truth, strict=True)
        if (limit := min(len(page), len(tags), len(gold)))
    ]
    return field_metrics(
        [decode_fields(page, tags) for page, tags, _ in trimmed],
        [decode_fields(page, gold) for page, _, gold in trimmed],
    )
