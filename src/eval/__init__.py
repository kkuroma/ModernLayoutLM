"""
Scoring predictions

Pure functions of predictions and labels, so this sits below src.train in the layer contract
Token tasks are scored at the entity level as per-token accuracy is dominated by the O class
"""

from .fields import FIELD_LEVEL_SOURCES, decode_fields, field_metrics, score_fields
from .metrics import decode_token_predictions, sequence_metrics, stitch_chunks, token_metrics

__all__ = [
    "FIELD_LEVEL_SOURCES",
    "decode_fields",
    "decode_token_predictions",
    "field_metrics",
    "score_fields",
    "sequence_metrics",
    "stitch_chunks",
    "token_metrics",
]
