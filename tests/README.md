# tests

Fast by construction: the default run takes seconds, because almost everything uses a tiny random model or a hand-built record rather than real weights and downloads.

```
conftest.py     tiny_config, a fixed batch, the backbone path fixture
test_data.py    schema and coordinate primitives, one case per adapter, the contamination filter
test_model.py   the graft, weight init, box-to-subword alignment, the pretrain to fine-tune handoff
test_eval.py    entity F1 over stitched chunks, SROIE field Hmean
test_train.py   packing, the MLM collator, parameter groups, checkpoint keeping
```

```sh
uv run pytest                    # the default, seconds
uv run pytest -m "slow or gpu"   # the heavy ones, currently none
```

Tests needing the pulled ModernBERT backbone skip themselves when it is absent.

Every failure mode here is quiet: boxes shifted by one word, `O` in the wrong label slot, a column pruned before the model sees it.
Each trains without complaint and yields a meaningless number, so the tests assert correspondences rather than shapes.
`test_zero_init_matches_modernbert` and `test_real_backbone_zero_init_parity` demand bit-identical logits against plain ModernBERT, and `test_boxes_survive_collation` and `test_head_signature_exposes_bbox` guard the two routes by which the boxes can vanish.
