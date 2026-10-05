# scripts

Runnable entrypoints, one per job.
Each loads `config.toml`, calls into `src`, and is never imported itself.

```
build_datasets.py      download, normalize, validate and shard any registered dataset
build_fingerprints.py  collect the benchmark Bates numbers the contamination filter matches
pretrain.py            layout-aware MLM in a 512 then a 2048 phase, or only their packs with --pack-only
finetune.py            any cells of the variant matrix, or quick probes of checkpoints
evaluate.py            pool the finished runs into the results tables
export.py              write a loadable Hugging Face directory, or push one with --push
```

```sh
uv run python scripts/build_datasets.py --group downstream
uv run python scripts/build_datasets.py --dataset idl-fresh --workers 3
uv run python scripts/build_datasets.py --dataset idl-docs --workers 3
uv run python scripts/pretrain.py --max-pages 2000 --max-steps 50        # probe first
uv run python scripts/pretrain.py                                         # both phases from ModernBERT
uv run python scripts/pretrain.py --backbone data/checkpoints/fp32 --phase long --run-name fp32
uv run python scripts/finetune.py --dataset funsd --variant layout layoutlm --seed 42 43 44
uv run python scripts/finetune.py --dataset funsd --variant layout --set max_length=512 --tag len512
uv run python scripts/finetune.py --probe data/checkpoints/fp32/kept/* --dataset funsd cord
uv run python scripts/evaluate.py
uv run python scripts/export.py --checkpoint data/checkpoints/fp32
```

`--workers` gives each shard of a sharded source its own process, which holds about 10GB while it unpacks.
`--limit` on `build_datasets.py` overwrites that dataset's full build, so rebuild without it afterwards.

`finetune.py` names runs `task-variant-sSEED` and skips any whose `result.json` exists.
A `--tag` suffix keeps an override apart, and `evaluate.py` reads only the untagged names, so exploratory runs never average into a reported cell.
Probe runs land under `data/checkpoints/probes` and append to `notebooks/results/checkpoint_series.json`.

| variant | what it isolates |
|---|---|
| `layout` | ModernLayoutLM after layout-MLM, the real thing |
| `nobbox` | the same weights with every box zeroed, isolating the 2D embeddings |
| `skipmlm` | fine-tuned straight off ModernBERT, isolating the pretraining stage |
| `plain` | ModernBERT with neither, the floor both stages are measured against |
| `layoutlm` | `microsoft/layoutlm-base-uncased` on the identical processed data |
