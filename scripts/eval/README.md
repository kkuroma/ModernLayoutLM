# eval scripts

Turning finished runs into a results table.

## Files

```
evaluate.py   collect every run's result.json into a markdown and JSON table
```

## Running

```sh
uv run python scripts/eval/evaluate.py
```

Reads `data/checkpoints/finetune/*/result.json` and writes `notebooks/results/results.{md,json}`.

## What the table is for

Four variants per task, because one number in isolation proves nothing:

| variant | what it isolates |
|---|---|
| `layout` | ModernLayoutLM after layout-MLM, the real thing |
| `nobbox` | the same recipe with boxes withheld, isolating the 2D embeddings |
| `skipmlm` | fine-tuned straight off ModernBERT, isolating the pretraining stage |
| `layoutlm` | `microsoft/layoutlm-base-uncased` on the identical processed data |

The script states plainly whether the layout run beat each control, including when it did not. A
null result is a result, and the FUNSD skip-MLM comparison already produced one.
