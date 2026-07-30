# scripts

Runnable entrypoints. Each one loads `config.toml`, then calls into `src`. Scripts import from
`src` and are never imported themselves.

## Tasks

```
scripts/
  data/    acquire and normalize datasets
  train/   pretraining and fine-tuning runs
  eval/    scoring and result tables
  export_hf.py    write a loadable Hugging Face model directory
  push_to_hub.py  upload it, not run by default
```

## Running

Every script takes its defaults from `config.toml` and accepts flags to override them.

```sh
uv run python scripts/data/build_datasets.py --group downstream
uv run python scripts/train/finetune.py --dataset funsd
uv run python scripts/eval/evaluate.py
```

Pass `--help` to any of them for the full flag list.
