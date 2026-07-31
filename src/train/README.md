# train

The top layer. Tokenizes canonical parquet into model inputs, then runs the two training stages.

## Files

```
encoding.py   canonical parquet to chunked model inputs
collator.py   the MVLM collator, which keeps the boxes
optimizer.py  parameter groups: layout tables apart from the backbone
pretrain.py   stage 1, layout-aware masked language modeling
finetune.py   stage 2, downstream supervised fine-tuning and the ablations
```

`encoding.py` lives here rather than in `src/data` because applying the processor needs both the
data stage and the model stage, and `src.train` is the only layer allowed to import both.

## Two things that exist only to stop the boxes disappearing

**`LayoutMlmCollator`.** `DataCollatorForLanguageModeling` builds its batch from `input_ids` and
drops every column it does not recognize, `bbox` among them. Wrapping it would not help, so it is
replaced. A run using the stock collator would train a layout model on no layout while reporting a
perfectly healthy falling loss.

**`bbox` named explicitly in every head's forward.** `Trainer._remove_unused_columns` inspects the
signature to decide which dataset columns to keep. The same silent failure, reached by a different
route. `tests/model/test_modeling.py` asserts the signature, and the collator raises if a feature
reaches it without boxes.

## Parameter groups

The four layout tables start at zero and have never seen a gradient. The backbone has seen two
trillion tokens. One learning rate cannot serve both, so `build_param_groups` splits them, and the
layout group is exempted from weight decay because decay would pull the tables back toward the zero
they started at.

`build_param_groups` raises on a model with no `LayoutEmbedding` tables. That is deliberate: it is
the cheapest way to catch a run accidentally configured against a plain ModernBERT.

## Chunking

A page longer than the sequence limit becomes several overlapping chunks rather than being
truncated. Token tasks keep their own slice of the labels per chunk; sequence tasks give every
chunk the page's label, so a long document contributes several examples and is scored per chunk.

## Running

```sh
# measure throughput before committing a night to it
uv run python scripts/train/pretrain.py --max-pages 2000 --max-steps 50
uv run python scripts/train/pretrain.py

uv run python scripts/train/finetune.py --dataset funsd
uv run python scripts/train/finetune.py --dataset funsd --no-bbox   # the no-bbox control
```

On NixOS both need `TRITON_LIBCUDA_PATH` and `TRITON_PTXAS_PATH`, which the flake sets. torch 2.13
routes ModernBERT's RoPE through a triton kernel, and triton otherwise shells out to
`/sbin/ldconfig` and ships a `ptxas` NixOS cannot execute.
