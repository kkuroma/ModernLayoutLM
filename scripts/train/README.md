# train scripts

The two training stages.

## Files

```
pretrain.py   stage 1, layout-aware MLM over the pretraining corpora
finetune.py   stage 2, downstream fine-tuning, including the ablation controls
```

## Running

```sh
# probe throughput first: the real pretraining run is an overnight job
uv run python scripts/train/pretrain.py --max-pages 2000 --max-steps 50
uv run python scripts/train/pretrain.py

uv run python scripts/train/finetune.py --dataset funsd
uv run python scripts/train/finetune.py --dataset funsd --no-bbox
uv run python scripts/train/finetune.py --dataset cord --backbone data/checkpoints/pretrain-base
```

Every ablation runs through the same `finetune` code path as the real thing, so a control is never
an approximation of the recipe it is controlling for.
