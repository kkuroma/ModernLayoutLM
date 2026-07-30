"""
Scoring predictions

Pure functions of predictions and labels, which is why this sits below `src.train` in the layer
contract rather than above it: training consumes them for its own compute_metrics callback. The
result tables are assembled in `scripts/eval/`.

Token tasks are scored at the entity level, because per-token accuracy on these datasets is
dominated by the O class and looks impressive while saying nothing.
"""

from .metrics import decode_token_predictions, sequence_metrics, token_metrics

__all__ = ["decode_token_predictions", "sequence_metrics", "token_metrics"]
