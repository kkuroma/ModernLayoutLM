# tests

Fast by construction. The default run is under a second, because almost everything uses a tiny
random model or a hand-built record rather than real weights and real downloads.

## Layout

```
conftest.py    tiny_config, a fixed batch, and the backbone path fixture
data/          coordinate primitives, schema validation, one case per adapter
model/         the graft, weight init, and box-to-subword alignment
train/         the MVLM collator and the parameter grouping
```

## Running

```sh
uv run pytest                       # the default, seconds
uv run pytest -m "slow or gpu"      # the heavy ones, currently none
```

Tests needing the pulled ModernBERT backbone skip themselves when it is absent rather than failing,
so a fresh clone still runs green.

## What these tests are actually for

Every failure mode in this project is quiet. Boxes misaligned by one word, a label scheme read with
`O` in the wrong slot, a dataset column pruned before it reaches the model: each of them trains
without complaint, shows a falling loss, and yields a number that means nothing.

So the tests assert correspondences rather than shapes. `test_every_subword_inherits_its_word_box`
checks which box each subword got, not that the tensor is the right size.
`test_zero_init_matches_modernbert` and `test_real_backbone_zero_init_parity` demand bit-identical
logits against plain ModernBERT, which is what proves the surgery did no damage.
`test_boxes_survive_collation` and `test_head_signature_exposes_bbox` guard the two independent
routes by which the boxes can silently vanish.
