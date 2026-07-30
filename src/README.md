# src

Importable code for ModernLayoutLM, split into four pipeline stages plus one shared config module.

## Stages

A stage imports only from the stages below it. The `import-linter` layers contract in
`pyproject.toml` enforces the order, which makes import cycles impossible.

```
src/
  config.py   typed access to config.toml, no stage of its own
  data/       stage 1: acquire raw datasets, normalize to one canonical schema
  model/      stage 2: config, modeling and processor. Self contained and exportable
  train/      stage 3: collators, param groups, training loops
  eval/       stage 4: metrics and result reporting
```

Nothing here reads a path or hyperparameter off disk. `scripts` loads `config.toml` via
`src.config.load_config` and passes the values in as arguments.

## The constraint on src/model

`src/model` is copied verbatim into the exported Hugging Face model directory by transformers'
`custom_object_save`, so that `from_pretrained(..., trust_remote_code=True)` works on a machine
that has never seen this repo. Those files may therefore import **only** torch and transformers,
and may reference each other only by relative import. A single `from src...` line in there breaks
every export.
