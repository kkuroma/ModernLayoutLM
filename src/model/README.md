# model

Stage 2. The ModernLayoutLM architecture: LayoutLM's four 2D position embedding tables grafted onto
a ModernBERT backbone, plus the processor that aligns boxes to subwords.

## Files

```
configuration_modernlayoutlm.py   ModernLayoutLmConfig, extends ModernBertConfig
modeling_modernlayoutlm.py        the graft, the encoder, and three task heads
processing_modernlayoutlm.py      words and boxes to model inputs
registry.py                       Auto* registration and the exported auto_map
```

## The graft

The change is confined to `ModernLayoutLmEmbeddings`. Six lookups are summed onto the token
embeddings before the inherited LayerNorm:

```
tok(id) + x(x0) + y(y0) + x(x1) + y(y1) + w(x1-x0) + h(y1-y0)
```

`x0` and `x1` share table X and `y0` and `y1` share table Y, which is LayoutLM v1 exactly.
Coordinates are clamped into the table range, and a malformed box with `x1 < x0` clamps its width
to zero rather than underflowing.

Attention is untouched. No 2D relative attention bias, because an arbitrary bias is incompatible
with ModernBERT's fused attention kernels and its sliding-window masks.

## Three decisions worth knowing

**Layout tables are zero-initialized.** `LayoutEmbedding` exists purely as a marker type, because
`_init_weights` is called on leaf modules and an ordinary `nn.Embedding` would be
indistinguishable from the token table and would inherit ModernBERT's normal init. Zeroed tables
make a fresh graft numerically identical to its backbone, asserted by
`test_real_backbone_zero_init_parity` against the real checkpoint.

**`bbox=None` skips the layout sum entirely.** That is not the same as passing all-zero boxes,
which mean "every word sits at the origin" and still cost six lookups. Omitting `bbox` reproduces
plain ModernBERT bit for bit, which is what the no-bbox ablation uses.

**`bbox` is the last parameter of every head's forward, and it is named explicitly.** Named,
because `Trainer._remove_unused_columns` inspects the signature and would otherwise drop the
column, training a layout model on no layout while appearing to work. Last, because inserting it
earlier would shift the positional order away from ModernBERT's and silently rebind a positional
`position_ids` into `bbox`.

## Export constraint

These files are copied verbatim into exported model directories by transformers'
`custom_object_save`, so they may import only torch and transformers, and may reference each other
only by relative import. `save_pretrained` follows that relative chain and copies the siblings too.
A single `from src...` line here breaks every export.

`registry.py` is deliberately not part of that chain. It imports the `Auto*` classes, which an
export does not need.

## Usage

```python
from transformers import AutoTokenizer
from src.model import ModernLayoutLmForTokenClassification, ModernLayoutLmProcessor

processor = ModernLayoutLmProcessor(tokenizer=AutoTokenizer.from_pretrained(backbone))
model = ModernLayoutLmForTokenClassification.from_pretrained(backbone, num_labels=7)

# boxes must already be normalized to [0, 1000]; the processor rejects raw pixels
batch = processor(words=words, boxes=boxes, word_labels=labels, return_tensors="pt")
output = model(**batch)
```

Grafting from a ModernBERT checkpoint reports the four tables as MISSING. That is expected: they
are the new parameters, and they load as zeros.
