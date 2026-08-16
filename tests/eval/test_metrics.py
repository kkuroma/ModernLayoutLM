"""
Tests for entity scoring and for stitching a chunked page back together

Stitching is the point: an entity cut by a chunk boundary otherwise scores as two truncated ones
That penalizes whichever model has the shorter window, which reversed the SROIE ranking once
"""

import numpy as np
import pytest

from src.eval.metrics import stitch_chunks, token_metrics

LABELS = ["O", "B-TOTAL", "I-TOTAL"]


def ids(tags):
    """Turns tag names into label ids."""
    return [LABELS.index(tag) for tag in tags]


def test_stitch_joins_the_chunks_of_one_page():
    predicted, truth = stitch_chunks(
        [["B-TOTAL"], ["I-TOTAL"], ["O"]],
        [["B-TOTAL"], ["I-TOTAL"], ["O"]],
        [0, 0, 1],
    )
    assert predicted == [["B-TOTAL", "I-TOTAL"], ["O"]]
    assert truth == [["B-TOTAL", "I-TOTAL"], ["O"]]


def test_stitch_orders_pages_by_index():
    predicted, _ = stitch_chunks([["O"], ["B-TOTAL"]], [["O"], ["B-TOTAL"]], [1, 0])
    assert predicted == [["B-TOTAL"], ["O"]]


def test_stitch_rejects_a_group_count_mismatch():
    with pytest.raises(ValueError):
        stitch_chunks([["O"]], [["O"]], [0, 1])


def test_split_entity_scores_as_one_once_stitched():
    # One TOTAL span cut across two chunks, predicted exactly right
    predictions = np.array(ids(["B-TOTAL", "I-TOTAL"] * 2)).reshape(2, 2)
    labels = predictions.copy()

    chunked = token_metrics(predictions, labels, LABELS)
    stitched = token_metrics(predictions, labels, LABELS, groups=[0, 0])

    # Chunked sees two spans and stitched sees one, and both are fully correct
    assert chunked["f1"] == pytest.approx(1.0)
    assert stitched["f1"] == pytest.approx(1.0)


def test_stitching_changes_the_score_when_the_boundary_hides_an_error():
    # Truth is one span over four words, the model restarts it at the boundary
    labels = np.array(ids(["B-TOTAL", "I-TOTAL", "I-TOTAL", "I-TOTAL"])).reshape(2, 2)
    predictions = np.array(ids(["B-TOTAL", "I-TOTAL", "B-TOTAL", "I-TOTAL"])).reshape(2, 2)

    chunked = token_metrics(predictions, labels, LABELS)
    stitched = token_metrics(predictions, labels, LABELS, groups=[0, 0])

    assert chunked["f1"] > stitched["f1"]


def test_ignored_positions_never_reach_the_metric():
    predictions = np.array([ids(["B-TOTAL", "O"])])
    labels = np.array([[LABELS.index("B-TOTAL"), -100]])
    assert token_metrics(predictions, labels, LABELS, groups=[0])["f1"] == pytest.approx(1.0)
