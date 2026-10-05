"""
Tests for batching and optimizer setup, the two places the boxes or the graft can silently vanish

A packer that drifts boxes by one position would speed training up and train on the wrong layout
An empty layout parameter group is the cheapest signal that the graft did not happen
"""

import logging
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer, ModernBertConfig, ModernBertModel

from src.config import load_config
from src.model import ModernLayoutLmConfig, ModernLayoutLmForTokenClassification
from src.train.batches import (
    IGNORE_INDEX,
    PAD_BOX,
    SEPARATOR_BOX,
    LayoutMlmCollator,
    chunk_order,
    pack_encodings,
)
from src.train.pretrain import KEPT_FILES, KeepCheckpoints, adamw, param_groups


@pytest.fixture(scope="module")
def tokenizer():
    """The pulled ModernBERT tokenizer."""
    backbone = Path("models/backbones/ModernBERT-base")
    if not (backbone / "tokenizer.json").is_file():
        pytest.skip(f"backbone not pulled to {backbone}")
    return AutoTokenizer.from_pretrained(backbone)


def make_features(count: int = 4, length: int = 16) -> list[dict[str, list]]:
    """Builds encoded rows with distinctive boxes, so a dropped column is obvious."""
    return [
        {
            "input_ids": [100 + index * length + position for position in range(length)],
            "attention_mask": [1] * length,
            "bbox": [[position, index, position + 5, index + 5] for position in range(length)],
        }
        for index in range(count)
    ]


def test_boxes_survive_collation(tokenizer) -> None:
    """
    The boxes must come through unchanged, position for position

    The stock collator drops bbox, and the run then trains on no layout with a healthy loss.
    """
    features = make_features()
    batch = LayoutMlmCollator(tokenizer=tokenizer)(features)

    assert "bbox" in batch
    assert batch["bbox"].shape == (4, 16, 4)
    expected = torch.tensor([feature["bbox"] for feature in features], dtype=torch.long)
    assert torch.equal(batch["bbox"], expected)


def test_missing_boxes_raise(tokenizer) -> None:
    """A feature without bbox means the tokenizer step lost it, which must not be tolerated."""
    features = make_features()
    del features[1]["bbox"]
    with pytest.raises(KeyError, match="no bbox"):
        LayoutMlmCollator(tokenizer=tokenizer)(features)


def test_labels_only_where_masked(tokenizer) -> None:
    """Everything not selected for prediction must be IGNORE_INDEX."""
    torch.manual_seed(0)
    batch = LayoutMlmCollator(tokenizer=tokenizer, mlm_probability=0.3)(make_features())
    selected = batch["labels"] != IGNORE_INDEX
    assert selected.any(), "nothing was masked at all"
    assert not selected.all(), "everything was masked"


def test_labels_recover_the_original_tokens(tokenizer) -> None:
    """
    Where a label is set, it must hold the token that was originally there

    Otherwise the model learns to predict whatever the corruption wrote, which teaches nothing.
    """
    torch.manual_seed(0)
    features = make_features()
    original = torch.tensor([feature["input_ids"] for feature in features], dtype=torch.long)
    batch = LayoutMlmCollator(tokenizer=tokenizer, mlm_probability=0.5)(features)

    selected = batch["labels"] != IGNORE_INDEX
    assert torch.equal(batch["labels"][selected], original[selected])


def test_masking_rate_is_roughly_the_requested_one(tokenizer) -> None:
    """
    Over a large batch the selected fraction must approach the configured probability

    ModernBERT pretrains at 0.3, and masking half as much would slow the stage without an error.
    """
    torch.manual_seed(0)
    batch = LayoutMlmCollator(tokenizer=tokenizer, mlm_probability=0.3)(
        make_features(count=64, length=64)
    )
    rate = (batch["labels"] != IGNORE_INDEX).float().mean().item()
    assert 0.24 < rate < 0.36, rate


def test_padding_is_never_masked(tokenizer) -> None:
    """A masked pad token is a wasted prediction and a spurious gradient."""
    torch.manual_seed(0)
    features = make_features(count=2, length=16)
    for feature in features:
        feature["attention_mask"] = [1] * 8 + [0] * 8
    batch = LayoutMlmCollator(tokenizer=tokenizer, mlm_probability=0.9)(features)
    assert torch.all(batch["labels"][:, 8:] == IGNORE_INDEX)


def test_ragged_rows_raise(tokenizer) -> None:
    """Unpadded rows cannot be stacked, and a clear error beats a torch shape message."""
    features = make_features(count=2, length=16)
    features[1]["input_ids"] = features[1]["input_ids"][:8]
    with pytest.raises(ValueError, match="ragged"):
        LayoutMlmCollator(tokenizer=tokenizer)(features)


def test_tokenizer_without_a_mask_token_raises(tokenizer) -> None:
    """MVLM is impossible without a mask token, so fail at construction not mid-run."""

    class NoMask:
        mask_token_id = None

    with pytest.raises(ValueError, match="mask token"):
        LayoutMlmCollator(tokenizer=NoMask())  # type: ignore[arg-type]


SEP, PAD = 2, 0


def page(start: int, length: int) -> dict:
    """Builds a page whose token ids and box x-coordinates both encode their identity."""
    return {
        "input_ids": list(range(start, start + length)),
        "bbox": [[start + i, 1, start + i + 1, 2] for i in range(length)],
    }


def test_pages_are_packed_into_full_sequences() -> None:
    """Several short pages should share one sequence rather than each padding out to length."""
    packed = list(pack_encodings(iter([page(10, 4), page(50, 4)]), 16, SEP, PAD))
    assert len(packed) == 1
    assert packed[0]["attention_mask"].count(1) == 9  # 4 + separator + 4


def test_every_token_keeps_its_own_box() -> None:
    """
    The box at each position must still belong to the token at that position

    Token ids and box x0 are built equal, so a single misalignment shows up as an inequality.
    """
    packed = list(pack_encodings(iter([page(10, 5), page(80, 6)]), 32, SEP, PAD))
    row = packed[0]
    for token, box in zip(row["input_ids"], row["bbox"], strict=True):
        if token == SEP:
            assert box == SEPARATOR_BOX
        elif token == PAD:
            assert box == PAD_BOX
        else:
            assert box[0] == token, f"token {token} carries box {box}"


def test_sequences_are_exactly_max_length() -> None:
    """Ragged output would fail to stack in the collator."""
    packed = list(pack_encodings(iter([page(i * 10, 7) for i in range(9)]), 24, SEP, PAD))
    assert len(packed) > 1
    for row in packed:
        assert len(row["input_ids"]) == 24
        assert len(row["attention_mask"]) == 24
        assert len(row["bbox"]) == 24


def test_an_over_length_page_gets_its_own_sequence() -> None:
    """
    A page longer than the window must not straddle a pack boundary

    Splitting it would place half a page beside an unrelated one, which is worse than truncating.
    """
    packed = list(pack_encodings(iter([page(10, 3), page(500, 40)]), 16, SEP, PAD))
    assert len(packed) == 2
    assert packed[0]["input_ids"][:3] == [10, 11, 12]
    # The long page starts its own sequence rather than continuing the first
    assert packed[1]["input_ids"][0] == 500


def test_the_trailing_partial_sequence_is_emitted() -> None:
    """A page must never be dropped just because it landed last."""
    packed = list(pack_encodings(iter([page(10, 4)]), 32, SEP, PAD))
    assert len(packed) == 1
    assert packed[0]["input_ids"][:4] == [10, 11, 12, 13]
    assert packed[0]["attention_mask"][4:] == [0] * 28


def test_padding_is_masked_out() -> None:
    """Padding must not be attended to, or the model learns from filler."""
    row = next(iter(pack_encodings(iter([page(10, 5)]), 20, SEP, PAD)))
    for index, mask in enumerate(row["attention_mask"]):
        if mask == 0:
            assert row["bbox"][index] == PAD_BOX


def test_mismatched_boxes_raise() -> None:
    """A page whose boxes and tokens disagree is a bug upstream and must not be packed."""
    broken = {"input_ids": [1, 2, 3], "bbox": [[0, 0, 1, 1]]}
    with pytest.raises(ValueError, match="tokens but"):
        list(pack_encodings(iter([broken]), 16, SEP, PAD))


def test_empty_pages_are_skipped() -> None:
    """An empty page contributes nothing and must not emit a stray separator."""
    packed = list(pack_encodings(iter([{"input_ids": [], "bbox": []}, page(10, 3)]), 16, SEP, PAD))
    assert len(packed) == 1
    assert packed[0]["input_ids"][0] == 10


def test_chunks_of_one_row_stay_together_and_in_order() -> None:
    """
    A long document's windows must follow each other, or a 2048 window reads mid-document text

    The block shuffle orders rows, and a row's overflow chunks ride along in their own order.
    """
    assert chunk_order([0, 0, 1, 2, 2, 2], [2, 0, 1]) == [3, 4, 5, 0, 1, 2]


def test_long_phase_inherits_what_it_does_not_set(tmp_path) -> None:
    """The long phase overrides its window and steps, and keeps every rate of the short one."""
    (tmp_path / "config.toml").write_text(
        '[Paths]\nraw_dir = "r"\nprocessed_dir = "p"\npack_dir = "k"\ncheckpoint_dir = "c"\n'
        'backbone_dir = "b"\nexport_dir = "e"\nreport_dir = "o"\n'
        '[Pretrain]\ndatasets = ["idl"]\nmax_length = 512\nsteps = 10\nmlm_probability = 0.3\n'
        "learning_rate = 5e-5\nlayout_learning_rate = 1e-3\nweight_decay = 0.01\n"
        "warmup_ratio = 0.05\nbatch_size = 32\ngrad_accum = 2\nsave_total_limit = 2\n"
        '[Pretrain.long]\ndatasets = ["idl-docs"]\nmax_length = 2048\nsteps = 4\nbatch_size = 8\n'
        "[Finetune]\nmax_length = 1024\nlearning_rate = 3e-4\nbaseline_learning_rate = 5e-5\n"
        "layout_learning_rate = 1e-4\nweight_decay = 0.01\nwarmup_ratio = 0.1\nepochs = 1\n"
        "batch_size = 8\ngrad_accum = 1\nlabel_all_subwords = false\n"
    )
    config = load_config(tmp_path / "config.toml")
    assert (config.pretrain.max_length, config.pretrain.steps) == (512, 10)
    long = config.pretrain_long
    assert (long.datasets, long.max_length, long.steps) == (["idl-docs"], 2048, 4)
    assert long.batch_size == 8
    assert (long.learning_rate, long.grad_accum) == (5e-5, 2)


def test_layout_parameters_are_found_by_type(tiny_config: ModernLayoutLmConfig) -> None:
    """
    Layout tables are located by module type, not by a name pattern

    A name match would drop a renamed table into the backbone group, at a twentieth of its rate.
    """
    model = ModernLayoutLmForTokenClassification(tiny_config)
    layout = next(
        group for group in param_groups(model, 5e-5, 1e-3, 0.01) if group["name"] == "layout"
    )
    tables = model.model.embeddings.layout_tables()
    assert {id(p) for p in layout["params"]} == {id(table.weight) for table in tables}


def test_groups_split_layout_from_backbone(tiny_config: ModernLayoutLmConfig) -> None:
    """The fresh tables get their own higher rate, and no weight decay."""
    model = ModernLayoutLmForTokenClassification(tiny_config)
    groups = param_groups(model, 5e-5, 1e-3, 0.01)
    by_name = {group["name"]: group for group in groups}
    assert set(by_name) == {"layout", "backbone", "no_decay"}
    assert by_name["layout"]["lr"] == 1e-3
    assert by_name["backbone"]["lr"] == 5e-5
    # Decay would pull the tables back toward the zero they were initialized to
    assert by_name["layout"]["weight_decay"] == 0.0
    assert by_name["backbone"]["weight_decay"] == 0.01
    assert by_name["no_decay"]["weight_decay"] == 0.0


def test_every_trainable_parameter_lands_in_exactly_one_group(
    tiny_config: ModernLayoutLmConfig,
) -> None:
    """A parameter in no group is never optimized, and one in two groups is stepped twice."""
    model = ModernLayoutLmForTokenClassification(tiny_config)
    groups = param_groups(model, 5e-5, 1e-3, 0.01)

    grouped = [parameter for group in groups for parameter in group["params"]]
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(grouped) == len(trainable)
    assert {id(p) for p in grouped} == {id(p) for p in trainable}


def test_layout_group_holds_all_four_tables(tiny_config: ModernLayoutLmConfig) -> None:
    """Four tables of max_2d by hidden_size each, and nothing else."""
    model = ModernLayoutLmForTokenClassification(tiny_config)
    groups = param_groups(model, 5e-5, 1e-3, 0.01)
    layout = next(group for group in groups if group["name"] == "layout")
    expected = 4 * tiny_config.max_2d_position_embeddings * tiny_config.hidden_size
    assert sum(parameter.numel() for parameter in layout["params"]) == expected


def test_a_model_without_layout_tables_raises(tiny_config: ModernLayoutLmConfig) -> None:
    """
    Grouping a plain ModernBERT must fail loudly

    Otherwise a run configured against the wrong class trains happily with no layout at all.
    """
    plain = ModernBertModel(ModernBertConfig(**tiny_config.to_dict()))
    with pytest.raises(ValueError, match="no LayoutEmbedding"):
        param_groups(plain, 5e-5, 1e-3, 0.01)


def test_adamw_logs_each_group(tiny_config: ModernLayoutLmConfig, caplog) -> None:
    """The group log is what a human reads before committing a night of GPU time."""
    model = ModernLayoutLmForTokenClassification(tiny_config)
    with caplog.at_level(logging.INFO, logger="src.train.pretrain"):
        adamw(param_groups(model, 5e-5, 1e-3, 0.01), fused=False)
    assert "layout" in caplog.text
    assert "backbone" in caplog.text
    assert "4 tensors" in caplog.text


def test_keep_checkpoints_copies_weights_only_on_its_interval(tmp_path) -> None:
    """Rotation deletes all but the newest checkpoints, so the kept ones must be copies."""
    for step in (2000, 4000):
        checkpoint = tmp_path / f"checkpoint-{step}"
        checkpoint.mkdir()
        for name in (*KEPT_FILES, "optimizer.pt"):
            (checkpoint / name).write_text(name)

    class Stub:
        output_dir = str(tmp_path)
        global_step = 0

    keep = KeepCheckpoints(every=4000)
    for step in (2000, 4000):
        Stub.global_step = step
        keep.on_save(Stub(), Stub(), Stub())  # type: ignore[arg-type]
    kept = tmp_path / "kept"
    assert [path.name for path in kept.iterdir()] == ["checkpoint-4000"]
    assert sorted(path.name for path in (kept / "checkpoint-4000").iterdir()) == sorted(KEPT_FILES)
