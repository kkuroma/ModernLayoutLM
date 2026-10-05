# train

The top layer: canonical parquet to model batches, then the two training stages.

```
batches.py    tokenize and chunk, pack pages for pretraining, and the MLM collator
pretrain.py   stage 1, one phase of layout-aware MLM, its parameter groups and checkpoint keeping
finetune.py   stage 2, downstream fine-tuning for every variant and the LayoutLM baseline
```

Batching lives here rather than in `src/data` because applying the processor needs both the data and the model stage, and `src.train` is the only layer allowed to import both.

Two things exist only to stop the boxes disappearing.
`LayoutMlmCollator` replaces `DataCollatorForLanguageModeling`, which drops every column it does not recognize, `bbox` among them.
And every head names `bbox` in its forward, because `Trainer._remove_unused_columns` prunes by signature.
Either failure trains a layout model on no layout while the loss falls normally.

The four layout tables start at zero, the backbone has seen two trillion tokens, so `param_groups` gives them separate rates and exempts the tables from weight decay.
It raises on a model without `LayoutEmbedding` tables, the cheapest catch for a run configured against plain ModernBERT.

Pretraining runs in two phases, 512-token windows of single pages and then 2048-token windows of whole documents, each for its own step count.
A packed row longer than the window becomes consecutive windows kept together and in order, rather than losing its tail.
Pretraining keeps fp32 master weights under bf16 autocast, compiles the model, scores only masked tokens and uses fused AdamW.
`KeepCheckpoints` copies the weights of every `keep_every`th step aside, so a long run leaves a transfer curve behind after its rolling checkpoints rotate out.

A page longer than the window becomes overlapping chunks.
Token tasks keep their own label slice per chunk, and scoring stitches the chunks of a page back together before it counts entities.

Both stages need `TRITON_LIBCUDA_PATH` and `TRITON_PTXAS_PATH` on NixOS, which the flake sets.
