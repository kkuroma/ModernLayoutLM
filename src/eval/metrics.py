"""
Scoring token and document classification

Token tasks are scored at the entity level with seqeval rather than per token
Per-token accuracy is dominated by the O class and entity F1 is what the literature reports
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


def decode_token_predictions(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_names: Sequence[str],
) -> tuple[list[list[str]], list[list[str]]]:
    """
    Turns logits and label ids into per-sequence BIO tag strings

    Positions labeled `IGNORE_INDEX` are dropped from both sides, so padding, special tokens and
    the non-first subwords of a word never reach the metric.

    Args:
        predictions: Logits of shape (batch, seq_len, num_labels), or ids of shape
            (batch, seq_len).
        labels: Label ids of shape (batch, seq_len).
        label_names: The BIO vocabulary, indexed by label id.

    Returns:
        Predicted and true tag sequences, one list per input sequence.
    """
    if predictions.ndim == 3:
        predictions = predictions.argmax(axis=-1)

    predicted: list[list[str]] = []
    truth: list[list[str]] = []
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
    Concatenates the chunks of one page back into a single tag sequence

    An entity split across a chunk boundary is otherwise scored as two truncated entities, which
    penalizes whichever model has the shorter window and so produced more chunks.

    Args:
        predicted: Predicted tags, one list per chunk.
        truth: True tags, one list per chunk.
        groups: The source page of each chunk, in the order the chunks were scored.

    Returns:
        Predicted and true tag sequences, one list per page.

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
    predicted = [p for p, _ in pairs]
    truth = [t for _, t in pairs]

    return {
        "precision": _scalar(precision_score(truth, predicted, zero_division=ZERO_DIVISION)),
        "recall": _scalar(recall_score(truth, predicted, zero_division=ZERO_DIVISION)),
        "f1": _scalar(f1_score(truth, predicted, zero_division=ZERO_DIVISION)),
        "token_accuracy": _scalar(accuracy_score(truth, predicted)),
    }


def _scalar(value: Any) -> float:
    """
    Narrows a seqeval score to one float

    seqeval returns a list of per-class scores when `average=None` and a single number otherwise.
    We always use an average, but the signature admits both, so this collapses the union rather
    than asserting it away.

    Args:
        value: A seqeval score.

    Returns:
        The score as a float, averaging if a list came back.
    """
    if isinstance(value, (list, tuple)):
        return sum(float(item) for item in value) / len(value) if value else 0.0
    return float(value)


def sequence_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """
    Computes accuracy and macro F1 for a document classification task

    Macro F1 sits beside accuracy because a model that ignores the rarer classes can still post a
    respectable accuracy on a 16-way problem.

    Args:
        predictions: Logits of shape (batch, num_labels), or predicted ids of shape (batch,).
        labels: True ids of shape (batch,).

    Returns:
        Accuracy and macro F1.
    """
    if predictions.ndim == 2:
        predictions = predictions.argmax(axis=-1)
    correct = float((predictions == labels).mean()) if labels.size else 0.0
    return {"accuracy": correct, "macro_f1": _macro_f1(predictions, labels)}


def _macro_f1(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Computes unweighted mean F1 across the classes present in the labels

    Args:
        predictions: Predicted class ids.
        labels: True class ids.

    Returns:
        The macro F1, or 0.0 when there are no labels.
    """
    classes = np.unique(labels)
    if classes.size == 0:
        return 0.0
    scores = []
    for value in classes:
        true_positive = float(np.count_nonzero((predictions == value) & (labels == value)))
        predicted_positive = float(np.count_nonzero(predictions == value))
        actual_positive = float(np.count_nonzero(labels == value))
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)
