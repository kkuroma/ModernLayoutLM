# src

Importable code for ModernLayoutLM, four pipeline stages plus one config module.
A stage imports only from the stages below it, and the `import-linter` layers contract in `pyproject.toml` enforces the order, which makes import cycles impossible.

```
src/
  config.py   typed access to config.toml
  data/       acquire raw datasets, reduce them to one canonical schema
  model/      config, modeling and processor, self contained and exportable
  eval.py     entity F1, SROIE field Hmean and accuracy, pure functions of predictions
  train/      batching, pretraining and fine-tuning
```

`eval` sits below `train` because training needs the metrics for its own `compute_metrics`.
Result reporting comes last, in `scripts/evaluate.py`.

Nothing here reads a path or a hyperparameter off disk.
Scripts load `config.toml` through `src.config.load_config` and pass the values in.

`src/model` is copied verbatim into every exported model directory by transformers' `custom_object_save`, so `from_pretrained(..., trust_remote_code=True)` works on a machine that has never seen this repo.
Its three HF modules may import only torch and transformers, and each other only by relative import.
A single `from src...` line there breaks every export.
