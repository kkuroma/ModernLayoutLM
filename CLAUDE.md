# CLAUDE.md

Guidance for Claude Code working in this repository.

Code style follows [the Kuroma manual](/home/kuroma/Documents/coding-style/README.md).
Read it before writing anything here.

## What this project is

LayoutLM's 2D layout embeddings grafted onto a ModernBERT backbone, pretrained with layout-aware MLM and fine-tuned on document understanding tasks.
The deliverable is a directory that `from_pretrained(..., trust_remote_code=True)` loads.

`README.md` has the overview.
`src/`, its stage directories and `scripts/` each have a README with the details that matter for that stage.

## Environment

- Python 3.14, torch 2.13.0+cu130, **transformers 5.14.1**.
  This is the v5 API, not v4.
- `direnv` plus the nix flake pin the tools, `uv` owns the packages.
  `uv sync`, then `uv run`.
- One RTX 3090, 24GB.
- `data` is a symlink to `/mnt/Vault-Storage/research/layout-datasets`, and holds the checkpoints and the exports.
  `models/` is local disk and holds only the upstream backbones.
  `packed/` is local disk and holds the tokenized pretraining packs, because training reads them in random order and the HDD cannot keep the GPU fed.
  All three gitignored, along with `scratchpad/`.

### Three environment traps

**Triton on NixOS.** torch 2.13 routes ModernBERT's RoPE through a triton kernel.
Triton finds libcuda by shelling out to `/sbin/ldconfig`, which NixOS does not have, and ships a `ptxas` built against an interpreter NixOS cannot execute.
The flake sets `TRITON_LIBCUDA_PATH` and `TRITON_PTXAS_PATH` to fix both.
Outside the direnv shell, export them or every CUDA run dies on its first step.

**Hugging Face caches default to the SSD.** Dataset downloads pass an explicit `cache_dir`, but hub weights and the arrow conversions `load_dataset` builds from `data/processed` do not, and their default is `~/.cache/huggingface`.
That reached 325GB before anyone looked.
The flake exports `HF_HOME` into `data/raw/hf-cache`, which puts the hub, datasets and modules caches on the vault.
Outside the direnv shell, export it, or the next long run fills the SSD instead.

**PaddleOCR is not installable.** `paddlepaddle` publishes no cp314 wheels, and declaring it even as an optional extra makes the whole lockfile unresolvable.
Nothing in the pipeline needs OCR.

### Sharing the machine

The GPU also serves the user's desktop and a llama.cpp router, whose chat model takes about 19GB.
`curl localhost:11434/router/restart` unloads it, which a pretraining run needs, as the fast stack peaks at 13.3GB.
The card runs at a 270W power cap against a 370W default, so training is power-bound rather than compute-bound.
RAM is 61GB and each `build_datasets.py --workers` process holds one unpickled IDL shard of about 10GB, so keep it at 3.
`data` sits on ZFS with auto-snapshots up to monthly, so deleting from it frees space only once the snapshots holding the data expire.

## Layout

```
src/data/    acquire datasets, normalize to one canonical schema
src/model/   the graft, the encoder, three heads, the processor
src/eval.py  scoring, pure functions of predictions and labels
src/train/   batching, param groups, the two training stages
scripts/     runnable entrypoints, one per job
```

`import-linter` enforces the layer order `src.train` > `src.eval` > `src.model` > `src.data`.
`eval` sits below `train` because training needs the metrics for its own `compute_metrics`.

## Rules that are not obvious

**Never let `src/model` import from `src`.** Those three modules are copied verbatim into exported model directories by `custom_object_save`.
They may import only torch and transformers, and may reference each other only by relative import.
One `from src...` line breaks every export.

**Boxes disappear silently, so both routes are guarded.** `Trainer._remove_unused_columns` prunes any column not named in the model's forward signature, and `DataCollatorForLanguageModeling` drops any column it does not recognize.
Hence `bbox` is named explicitly in every head's forward, and `LayoutMlmCollator` replaces the stock collator.
A run that loses the boxes still trains and still shows a falling loss.

**`bbox` goes last in a forward signature.** Inserting it earlier shifts the positional order away from ModernBERT's, so a positional `position_ids` would silently rebind into `bbox`.

**Unknown labels and failed validation are fatal, deliberately.** `Document.validate` runs on every document at build time, and adapters raise on an unrecognized label rather than mapping it to the outside class.
That rule caught CORD's dotted `menu.sub.nm` and RVL-CDIP's four misspelled classes, the latter of which had silently discarded a quarter of the corpus.

**Zero-init is load bearing.** The four layout tables start at zero so a fresh graft is bit-identical to its backbone.
`test_real_backbone_zero_init_parity` asserts exactly that against the real checkpoint.
`LayoutEmbedding` exists only as a marker type, because `_init_weights` sees leaf modules and a plain `nn.Embedding` would be indistinguishable from the token table.

## Verifying a change

```sh
uv run pytest              # under a second
ruff check src scripts tests && black --check src scripts tests
pyright                    # standard mode, expected clean
uv run lint-imports        # the layer contract
```

## Running the pipeline

```sh
uv run python scripts/build_datasets.py --group downstream
uv run python scripts/build_datasets.py --group realkie
uv run python scripts/build_datasets.py --group pretrain --workers 3
uv run python scripts/build_datasets.py --dataset idl-fresh --workers 3
uv run python scripts/build_datasets.py --dataset idl-docs --workers 3
uv run python scripts/pretrain.py --pack-only                        # tokenize ahead of time
uv run python scripts/pretrain.py --max-pages 2000 --max-steps 50    # probe first
uv run python scripts/pretrain.py                                    # 512 phase, then 2048
uv run python scripts/finetune.py --dataset funsd --seed 42 43 44
uv run python scripts/evaluate.py
uv run python scripts/export.py --checkpoint data/checkpoints/fp32
uv run python scripts/export.py --push owner/name --dry-run
```

The shipped run did 52 sequences per second at length 512, over 1.91M packed sequences in 10.1 hours.
Compile, flex attention, masked-only MLM scoring, fused AdamW and micro-batch 32 now reach 104, measured in `data/notes/speedup-2026-09-30.md` and shown there to train the same model.
Investigation write-ups from the scratchpad, such as the August findings and the bf16 diagnosis, are kept in `data/notes/`.
Probe before committing to a long run, and monitor it every 30 minutes per the manual rather than watching it.

## Status

Model, data, training, scoring and export are in and tested.
The table covers FUNSD, CORD and SROIE on the 11M-page weights, and `README.md` carries it.
ModernLayoutLM trails LayoutLM v1 by 0.022 on FUNSD and 0.014 on CORD, and leads by 0.011 Hmean on SROIE inside seed noise.
Kleister-Charity was dropped on 2026-10-05 for RealKIE, as its spans were string-matched and its pages stacked onto one grid.
FUNSD fine-tuning is not deterministic at a fixed seed on this GPU: two runs of seed 43 gave 0.7745 and 0.7581.

Three facts the table depends on and neither the code nor the config records.
Fine-tuning must start from `data/checkpoints/fp32`; the earlier `pretrain-base` run kept its master weights in bf16 under flash-attn and transferred about 0.11 F1 worse on FUNSD, and since both write `dtype: float32` on save, only downstream transfer told them apart.
Only runs named `task-variant-sSEED` reach the table, which is how an exploratory run sits beside the matrix without being averaged into it.
And LayoutLM v1 runs at `baseline_learning_rate`, which `finetune` swaps in for any LayoutLM backbone, so `--set learning_rate` never reaches it.

`data/checkpoints` was pruned of weights on 2026-08-19.
The `fp32` root and four fine-tuned examples keep their tensors; every other run keeps its `config.json` and its `result.json` or `pretrain_result.json` and nothing else, so no deleted run can be re-scored without retraining it.
`data/checkpoints/CLEANUP-2026-08-19.md` lists what went.

### Before publishing

The card in `scripts/export.py` and the export in `data/exports/ModernLayoutLM-base-mlm` carry the 11M-page weights and the four-task table, with each model's rate and each task's split, as of 2026-10-04.
Pushing needs `HF_TOKEN` with write access and a repository name.
Per-model validation sweeps were dropped, decided 2026-10-02, as the precision they buy sits below FUNSD's seed noise.

### The 11M-page checkpoint

`data/checkpoints/fp32` has seen about 11M page passes since 2026-10-02.
It continued the 2.19M-page run over 8.8M pages of `data/processed/idl-fresh`, from 21 IDL shards the first corpus never read, in 17.9 hours at 114 seq/s.
`fp32/kept` holds the weights every 2M pages.
On FUNSD it scores 0.7565 at 3e-4, up from 0.7153, and 1e-4 and 5e-4 did no better; 5e-5 has not been tried on these weights.
LayoutLM's own paper puts its FUNSD F1 at 0.6957 after 2M pages and 0.7636 after 11M.
