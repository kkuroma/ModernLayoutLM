# eval

Scoring. Pure functions of predictions and labels, with no model or dataset dependency.

## Files

```
metrics.py   entity-level F1 for token tasks, accuracy and macro F1 for document classification
```

## Why this sits below train

The layer contract puts `src.eval` beneath `src.train`, which reads backwards until you notice that
training needs the metrics for its own `compute_metrics` callback. Scoring is a utility; it is the
result *reporting* that comes last, and that lives in `scripts/eval/`.

## Entity F1, not token accuracy

Token tasks are scored at the entity level with seqeval. Per-token accuracy on FUNSD, CORD and
SROIE is dominated by the `O` class: a model that predicts `O` everywhere scores well and has found
nothing. Entity F1 is also the number the LayoutLM literature reports, so it is the only one worth
comparing against.

Positions labelled `-100` are dropped before scoring, which excludes padding, special tokens and
the non-first subwords of a word. Those are exactly the positions the loss ignored, so including
them would score the model on predictions it was never trained to make.

Macro F1 accompanies accuracy for RVL-CDIP, because on a 16-way problem a model that quietly
ignores the rarer classes can still post a respectable accuracy.
