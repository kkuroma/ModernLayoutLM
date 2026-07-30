# CLAUDE.md

Guidance for Claude Code working in this repository.

Code style follows [the Kuroma manual](/home/kuroma/Documents/coding-style/README.md). Read it
before writing anything here.

## What this project is

LayoutLM's 2D layout embeddings grafted onto a ModernBERT backbone, pretrained with layout-aware
MLM and fine-tuned on document understanding tasks. The deliverable is a directory that
`from_pretrained(..., trust_remote_code=True)` loads.

`README.md` has the overview. Each directory under `src/` and `scripts/` has its own README with
the details that matter for that stage.

## Environment

- Python 3.14, torch 2.13.0+cu130, **transformers 5.14.1**. This is the v5 API, not v4.
- `direnv` plus the nix flake pin the tools, `uv` owns the packages. `uv sync`, then `uv run`.
- One RTX 3090, 24GB.
- `data` is a symlink to `/mnt/Vault-Storage/research/layout-datasets`. `models/` is local disk.
  Both gitignored, along with `scratchpad/`.

### Two environment traps

**Triton on NixOS.** torch 2.13 routes ModernBERT's RoPE through a triton kernel. Triton finds
libcuda by shelling out to `/sbin/ldconfig`, which NixOS does not have, and ships a `ptxas` built
against an interpreter NixOS cannot execute. The flake sets `TRITON_LIBCUDA_PATH` and
`TRITON_PTXAS_PATH` to fix both. Outside the direnv shell, export them or every CUDA run dies on
its first step.

**PaddleOCR is not installable.** `paddlepaddle` publishes no cp314 wheels, and declaring it even
as an optional extra makes the whole lockfile unresolvable. Nothing in the pipeline needs OCR.

## Layout

```
src/data/    acquire datasets, normalize to one canonical schema
src/model/   the graft, the encoder, three heads, the processor
src/eval/    scoring, pure functions of predictions and labels
src/train/   collators, param groups, the two training stages
scripts/     runnable entrypoints, grouped by task
```

`import-linter` enforces the layer order `src.train` > `src.eval` > `src.model` > `src.data`.
`eval` sits below `train` because training needs the metrics for its own `compute_metrics`.

## Rules that are not obvious

**Never let `src/model` import from `src`.** Those three modules are copied verbatim into exported
model directories by `custom_object_save`. They may import only torch and transformers, and may
reference each other only by relative import. One `from src...` line breaks every export.

**Boxes disappear silently, so both routes are guarded.** `Trainer._remove_unused_columns` prunes
any column not named in the model's forward signature, and `DataCollatorForLanguageModeling` drops
any column it does not recognize. Hence `bbox` is named explicitly in every head's forward, and
`LayoutMlmCollator` replaces the stock collator. A run that loses the boxes still trains and still
shows a falling loss.

**`bbox` goes last in a forward signature.** Inserting it earlier shifts the positional order away
from ModernBERT's, so a positional `position_ids` would silently rebind into `bbox`.

**Unknown labels and failed validation are fatal, deliberately.** `Document.validate` runs on every
document at build time, and adapters raise on an unrecognized label rather than mapping it to the
outside class. That rule caught CORD's dotted `menu.sub.nm` and RVL-CDIP's four misspelled classes,
the latter of which had silently discarded a quarter of the corpus.

**Zero-init is load bearing.** The four layout tables start at zero so a fresh graft is bit-identical
to its backbone. `test_real_backbone_zero_init_parity` asserts exactly that against the real
checkpoint. `LayoutEmbedding` exists only as a marker type, because `_init_weights` sees leaf
modules and a plain `nn.Embedding` would be indistinguishable from the token table.

## Verifying a change

```sh
uv run pytest              # under a second
ruff check src scripts tests && black --check src scripts tests
pyright                    # standard mode, expected clean
uv run lint-imports        # the layer contract
```

For anything touching the data stage, also render the boxes. Validation catches coordinates outside
the grid; only a picture catches coordinates that are in range and still wrong.

```sh
uv run python scripts/data/inspect_dataset.py --dataset cord --render scratchpad/cord.png
```

## Running the pipeline

```sh
uv run python scripts/data/build_datasets.py --group downstream
uv run python scripts/data/build_datasets.py --group pretrain
uv run python scripts/train/pretrain.py --max-pages 2000 --max-steps 50   # probe first
uv run python scripts/train/pretrain.py
uv run python scripts/train/finetune.py --dataset funsd
uv run python scripts/eval/evaluate.py
uv run python scripts/export_hf.py --checkpoint data/checkpoints/pretrain-base
```

Measured throughput is 54 sequences per second at length 512, so one epoch over the 528k-page
corpus takes about 7 hours. Probe before committing to a long run, and monitor it every 30 minutes
per the manual rather than watching it.

## Status

Model, data, training, scoring and export are all in and tested. The layout-MLM run and the full
ablation table are the remaining work; `README.md` carries the current results.
