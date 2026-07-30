# ModernLayoutLM

LayoutLM's 2D layout embeddings grafted onto a ModernBERT backbone, pretrained with layout-aware
masked language modelling and fine-tuned on document understanding tasks.

## What this is

LayoutLM made BERT layout-aware by adding four 2D position embedding tables and summing them into
the token embeddings. That backbone is now six years old. ModernBERT is the current encoder of
record (RoPE, alternating local and global attention, GeGLU, 8192 native context, 2T training
tokens) but knows nothing about where words sit on a page.

ModernLayoutLM applies the LayoutLM v1 technique to ModernBERT. Coordinates are normalized to
`[0, 1000]` and looked up in four tables, where `x0` and `x1` share table X and `y0` and `y1`
share table Y, exactly as in the original. The sum lands on the token embeddings before the
embedding LayerNorm.

Nothing touches the attention mechanism. A LayoutLMv3-style 2D relative attention bias would be
stronger in principle, but an arbitrary attention bias is incompatible with ModernBERT's fused
attention kernels and its sliding-window masks, so it was ruled out deliberately.

The four new tables are **zero-initialized**. Adding seven randomly initialized vectors to every
token embedding would wreck the pretrained representation before the first optimizer step. Zero
init means the model starts numerically identical to ModernBERT and grows layout sensitivity from
there. `tests/model/` asserts that parity.

## Layout

```
data -> /mnt/Vault-Storage/research/layout-datasets   symlink, gitignored
  raw/          datasets exactly as downloaded
  processed/    canonical parquet, our schema
  checkpoints/  intermediate training checkpoints
models/         gitignored: pulled backbones and final exports
src/            importable code, four layered stages
scripts/        runnable entrypoints, grouped by task
tests/          fast by construction
notebooks/      experiment reports
config.toml     every path and hyperparameter
```

See `src/README.md` and `scripts/README.md` for the module trees.

## Setup

`direnv` plus the nix flake pin the interpreter and tools, `uv` owns the packages.

```sh
uv sync
```

PaddleOCR is intentionally not a dependency: `paddlepaddle` ships no cp314 wheels, and no dataset
in the pipeline needs OCR. See `src/data/README.md` if that ever changes.

## Running it

```sh
# 1. pull the backbone and the small downstream datasets
uv run python scripts/data/download_raw.py --group downstream
uv run python scripts/data/build_downstream.py

# 2. the layout-MLM corpus, ~46GB of downloads
uv run python scripts/data/download_raw.py --group pretrain
uv run python scripts/data/build_pretrain.py

# 3. layout-aware MLM, then downstream SFT
uv run python scripts/train/pretrain_mlm.py
uv run python scripts/train/finetune_token.py --dataset funsd

# 4. score everything and export
uv run python scripts/eval/evaluate.py
uv run python scripts/export_hf.py
```

## Results

Pending. The results table lands in `notebooks/results/` and covers four rows per task:
ModernLayoutLM, a no-bbox control, a skip-MLM control, and `microsoft/layoutlm-base-uncased`
fine-tuned on the identical processed data.

## Style

Code follows [the Kuroma manual of coding style](/home/kuroma/Documents/coding-style/README.md).
