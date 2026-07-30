# src

Importable code for ModernLayoutLM, split into four pipeline stages plus one shared config module.

## Stages

A stage imports only from the stages below it. The `import-linter` layers contract in
`pyproject.toml` enforces the order, which makes import cycles impossible.

```
src/
  config.py   typed access to config.toml, no stage of its own
  data/       acquire raw datasets, normalize to one canonical schema
  model/      config, modeling and processor. Self contained and exportable
  eval/       scoring: entity F1 and accuracy, pure functions of predictions and labels
  train/      collators, param groups, training loops
```

`eval` sits below `train`, not above it, because training needs the metrics for its own
`compute_metrics` callback. Scoring is a pure function; it is the result *reporting* that comes
last, and that lives in `scripts/eval/`.

Nothing here reads a path or hyperparameter off disk. `scripts` loads `config.toml` via
`src.config.load_config` and passes the values in as arguments.

## The constraint on src/model

`src/model` is copied verbatim into the exported Hugging Face model directory by transformers'
`custom_object_save`, so that `from_pretrained(..., trust_remote_code=True)` works on a machine
that has never seen this repo. Those files may therefore import **only** torch and transformers,
and may reference each other only by relative import. A single `from src...` line in there breaks
every export.
