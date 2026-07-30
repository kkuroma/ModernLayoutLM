"""
Tests for the parameter grouping

An empty layout group is the cheapest possible signal that the graft did not happen, so it is worth
asserting rather than eyeballing in a log line.
"""

import pytest
from transformers import ModernBertConfig, ModernBertModel

from src.model import ModernLayoutLmConfig, ModernLayoutLmForTokenClassification
from src.train.optimizer import build_param_groups, describe_param_groups, layout_parameter_names


def test_layout_parameters_are_found_by_type(tiny_config: ModernLayoutLmConfig) -> None:
    """
    Layout tables are located by module type, not by a name pattern

    Matching on names would silently drop a renamed table into the backbone group, where its
    learning rate would be twenty times too low.
    """
    model = ModernLayoutLmForTokenClassification(tiny_config)
    names = layout_parameter_names(model)
    assert len(names) == 4
    assert all(name.endswith(".weight") for name in names)
    assert all("position_embeddings" in name for name in names)


def test_groups_split_layout_from_backbone(tiny_config: ModernLayoutLmConfig) -> None:
    """The fresh tables get their own higher rate, and no weight decay."""
    model = ModernLayoutLmForTokenClassification(tiny_config)
    groups = build_param_groups(
        model=model,
        learning_rate=5e-5,
        layout_learning_rate=1e-3,
        weight_decay=0.01,
    )
    by_name = {group["name"]: group for group in groups}
    assert set(by_name) == {"layout", "backbone", "no_decay"}
    assert by_name["layout"]["lr"] == 1e-3
    assert by_name["backbone"]["lr"] == 5e-5
    # Decay would pull the tables back toward the zero they were initialized to.
    assert by_name["layout"]["weight_decay"] == 0.0
    assert by_name["backbone"]["weight_decay"] == 0.01
    assert by_name["no_decay"]["weight_decay"] == 0.0


def test_every_trainable_parameter_lands_in_exactly_one_group(
    tiny_config: ModernLayoutLmConfig,
) -> None:
    """A parameter in no group is never optimized, and one in two groups is stepped twice."""
    model = ModernLayoutLmForTokenClassification(tiny_config)
    groups = build_param_groups(model, 5e-5, 1e-3, 0.01)

    grouped = [parameter for group in groups for parameter in group["params"]]
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(grouped) == len(trainable)
    assert {id(p) for p in grouped} == {id(p) for p in trainable}


def test_layout_group_holds_all_four_tables(tiny_config: ModernLayoutLmConfig) -> None:
    """Four tables of max_2d by hidden_size each, and nothing else."""
    model = ModernLayoutLmForTokenClassification(tiny_config)
    groups = build_param_groups(model, 5e-5, 1e-3, 0.01)
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
        build_param_groups(plain, 5e-5, 1e-3, 0.01)


def test_describe_reports_each_group(tiny_config: ModernLayoutLmConfig) -> None:
    """The summary is what a human reads before committing a night of GPU time."""
    model = ModernLayoutLmForTokenClassification(tiny_config)
    text = describe_param_groups(build_param_groups(model, 5e-5, 1e-3, 0.01))
    assert "layout" in text
    assert "backbone" in text
    assert "4 tensors" in text.replace("  ", " ")
