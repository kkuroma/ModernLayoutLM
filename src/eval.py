"""
Scoring predictions as pure functions of predictions and labels, below src.train in the layers

Token tasks are scored per entity with seqeval, as per-token accuracy is dominated by the O class
SROIE is scored as ICDAR task 3 does, on exact strings per field micro-averaged over receipts
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score

# Positions the loss ignored, which must also be excluded from scoring
IGNORE_INDEX = -100

# Widen here as seqeval ships no stubs and pyright infers a string
ZERO_DIVISION: Any = 0

# The four strings ICDAR task 3 extracts from a receipt
FIELDS = ["company", "date", "address", "total"]

# Datasets whose headline number is field-level rather than entity-level
FIELD_LEVEL_SOURCES = {"sroie"}


def decode_token_predictions(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_names: Sequence[str],
) -> tuple[list[list[str]], list[list[str]]]:
    """
    Turns logits and label ids into per-sequence BIO tags, dropping positions the loss ignored

    Args:
        predictions: Logits of shape (batch, seq_len, num_labels), or ids of shape (batch, seq_len).
        labels: Label ids of shape (batch, seq_len).
        label_names: The BIO vocabulary, indexed by label id.

    Returns:
        Predicted and true tag sequences, one list per input sequence.
    """
    if predictions.ndim == 3:
        predictions = predictions.argmax(axis=-1)
    predicted, truth = [], []
    for row_predictions, row_labels in zip(predictions, labels, strict=True):
        keep = row_labels != IGNORE_INDEX
        predicted.append([label_names[int(index)] for index in row_predictions[keep]])
        truth.append([label_names[int(index)] for index in row_labels[keep]])
    return predicted, truth


def stitch_chunks(
    predicted: list[list[str]],
    truth: list[list[str]],
    groups: Sequence[int],
) -> tuple[list[list[str]], list[list[str]]]:
    """
    Concatenates the chunks of each page back into one tag sequence

    Otherwise an entity split by a chunk boundary scores as two, penalizing the shorter window.

    Args:
        predicted: Predicted tags, one list per chunk.
        truth: True tags, one list per chunk.
        groups: The source page of each chunk, in scoring order.

    Returns:
        Predicted and true tag sequences, one list per page in page order.

    Raises:
        ValueError: If the group count does not match the chunk count.
    """
    if not len(groups) == len(predicted) == len(truth):
        raise ValueError(f"{len(groups)} groups against {len(predicted)} chunks")
    pages: dict[int, tuple[list[str], list[str]]] = {}
    for group, chunk_predicted, chunk_truth in zip(groups, predicted, truth, strict=True):
        page_predicted, page_truth = pages.setdefault(int(group), ([], []))
        page_predicted.extend(chunk_predicted)
        page_truth.extend(chunk_truth)
    ordered = [pages[key] for key in sorted(pages)]
    return [p for p, _ in ordered], [t for _, t in ordered]


def token_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_names: Sequence[str],
    groups: Sequence[int] | None = None,
) -> dict[str, float]:
    """
    Computes entity-level precision, recall and F1 for a token classification task

    Args:
        predictions: Logits or predicted ids.
        labels: Label ids, with `IGNORE_INDEX` where the loss was masked.
        label_names: The BIO vocabulary.
        groups: The source page of each row, which stitches a chunked page before scoring.

    Returns:
        Entity precision, recall and F1, plus token accuracy for reference.
    """
    predicted, truth = decode_token_predictions(predictions, labels, label_names)
    if groups is not None:
        predicted, truth = stitch_chunks(predicted, truth, groups)
    # Sequences emptied by the ignore mask would make seqeval divide by zero
    pairs = [(p, t) for p, t in zip(predicted, truth, strict=True) if t]
    if not pairs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "token_accuracy": 0.0}
    predicted, truth = [p for p, _ in pairs], [t for _, t in pairs]
    return {
        "precision": _scalar(precision_score(truth, predicted, zero_division=ZERO_DIVISION)),
        "recall": _scalar(recall_score(truth, predicted, zero_division=ZERO_DIVISION)),
        "f1": _scalar(f1_score(truth, predicted, zero_division=ZERO_DIVISION)),
        "token_accuracy": _scalar(accuracy_score(truth, predicted)),
    }


def decode_fields(words: Sequence[str], tags: Sequence[str]) -> dict[str, str]:
    """
    Joins each field's first tagged span into one string, as a receipt names each field once

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
        if tag.startswith("I-") and field == tag[2:].lower():
            current.append(word)
            continue
        if current and field not in spans:
            spans[field] = current
        field, current = (tag[2:].lower(), [word]) if tag.startswith("B-") else ("", [])
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
        Field precision, field recall and Hmean.

    Raises:
        ValueError: If the two sequences are different lengths.
    """
    if len(predicted) != len(truth):
        raise ValueError(f"{len(predicted)} predictions but {len(truth)} receipts")
    correct = guessed = actual = 0
    for got, want in zip(predicted, truth, strict=True):
        for name in FIELDS:
            guessed += bool(got.get(name))
            actual += bool(want.get(name))
            correct += bool(got.get(name)) and bool(want.get(name)) and got[name] == want[name]
    precision = correct / guessed if guessed else 0.0
    recall = correct / actual if actual else 0.0
    hmean = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"field_precision": precision, "field_recall": recall, "hmean": hmean}


def score_fields(
    words: Sequence[Sequence[str]],
    predicted: Sequence[Sequence[str]],
    truth: Sequence[Sequence[str]],
) -> dict[str, float]:
    """
    Decodes tags into fields and scores them, the whole field-level path

    Truth comes from the tags, as the official keys disagree on a fifth of addresses over OCR gaps.

    Args:
        words: Each receipt's words in reading order.
        predicted: Predicted tags per receipt, already stitched across chunks.
        truth: True tags per receipt.

    Returns:
        Field precision, field recall and Hmean.
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


def _scalar(value: Any) -> float:
    """
    Narrows a seqeval score to one float, averaging if a per-class list came back

    Args:
        value: A seqeval score.

    Returns:
        The score as a float.
    """
    if isinstance(value, list | tuple):
        return sum(float(item) for item in value) / len(value) if value else 0.0
    return float(value)
