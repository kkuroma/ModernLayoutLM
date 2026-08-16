"""
Tests for SROIE field-level scoring

The competition extracts four exact strings per receipt, so a near miss is worth nothing
Entity F1 gives partial credit for the same prediction, which is why both are reported
"""

import pytest

from src.eval.fields import decode_fields, field_metrics, score_fields


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
