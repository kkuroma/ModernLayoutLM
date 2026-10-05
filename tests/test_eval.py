"""
Tests for scoring, entity F1 with stitched chunks and SROIE's field-level Hmean
"""

import numpy as np
import pytest

from src.eval import decode_fields, field_metrics, score_fields, stitch_chunks, token_metrics

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


def test_decode_joins_a_span_into_one_string():
    words = ["ACME", "SDN", "BHD", "2018-01-01"]
    tags = ["B-COMPANY", "I-COMPANY", "I-COMPANY", "B-DATE"]
    assert decode_fields(words, tags) == {"company": "ACME SDN BHD", "date": "2018-01-01"}


def test_decode_keeps_only_the_first_span_of_a_field():
    words = ["9.00", "1.00"]
    tags = ["B-TOTAL", "B-TOTAL"]
    assert decode_fields(words, tags) == {"total": "9.00"}


def test_decode_ignores_an_orphan_continuation():
    assert decode_fields(["a", "b"], ["O", "I-TOTAL"]) == {}


def test_decode_rejects_a_length_mismatch():
    with pytest.raises(ValueError):
        decode_fields(["a"], ["O", "O"])


def test_an_exact_match_scores_and_a_near_miss_does_not():
    exact = field_metrics([{"total": "9.00"}], [{"total": "9.00"}])
    near = field_metrics([{"total": "9.0"}], [{"total": "9.00"}])
    assert exact["hmean"] == pytest.approx(1.0)
    assert near["hmean"] == pytest.approx(0.0)


def test_a_missing_prediction_costs_recall_not_precision():
    scores = field_metrics([{"total": "9.00"}], [{"total": "9.00", "date": "2018-01-01"}])
    assert scores["field_precision"] == pytest.approx(1.0)
    assert scores["field_recall"] == pytest.approx(0.5)


def test_metrics_reject_a_receipt_count_mismatch():
    with pytest.raises(ValueError):
        field_metrics([{}], [{}, {}])


def test_score_fields_trims_words_the_tags_do_not_cover():
    # Truncation dropped the tail, so the receipt has more words than tags
    scores = score_fields(
        [["ACME", "BHD", "dropped"]],
        [["B-COMPANY", "I-COMPANY"]],
        [["B-COMPANY", "I-COMPANY"]],
    )
    assert scores["hmean"] == pytest.approx(1.0)
