"""
Parameter groups for a grafted model

The zero-initialized layout tables and the pretrained backbone cannot share one learning rate
Weight decay is withheld from the layout tables as it pulls them back toward zero
"""

from __future__ import annotations

from typing import Any

from torch import nn

# Modules whose weights conventionally escape weight decay, plus all biases
NO_DECAY_SUFFIXES = ("bias", "norm.weight", "LayerNorm.weight")


def layout_parameter_names(model: nn.Module) -> set[str]:
    """
    Finds the names of every layout embedding parameter in a model

    Matches on the module type rather than on a name pattern, so renaming a table cannot silently
    drop it back into the backbone group.

    Args:
        model: The model to inspect.

    Returns:
        Fully qualified parameter names belonging to layout embedding tables.
    """
    from src.model.modeling_modernlayoutlm import LayoutEmbedding

    names: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, LayoutEmbedding):
            for parameter_name, _ in module.named_parameters(recurse=False):
                names.add(f"{module_name}.{parameter_name}" if module_name else parameter_name)
    return names


def build_param_groups(
    model: nn.Module,
    learning_rate: float,
    layout_learning_rate: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """
    Splits a model's parameters into three groups: layout, decayed backbone, undecayed backbone

    Args:
        model: The model to optimize.
        learning_rate: Rate for the pretrained backbone.
        layout_learning_rate: Rate for the fresh layout tables, normally much higher.
        weight_decay: Decay for the backbone weights that take it.

    Returns:
        Parameter group dicts ready for `AdamW`. Empty groups are omitted.

    Raises:
        ValueError: If the model has no layout tables, which means the wrong model was passed.
    """
    layout_names = layout_parameter_names(model)
    if not layout_names:
        raise ValueError(
            "This model has no LayoutEmbedding tables, so there is nothing to graft. "
            "Check that a ModernLayoutLm class was constructed, not a plain ModernBert one."
        )

    layout: list[nn.Parameter] = []
    decayed: list[nn.Parameter] = []
    undecayed: list[nn.Parameter] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name in layout_names:
            layout.append(parameter)
        elif name.endswith(NO_DECAY_SUFFIXES):
            undecayed.append(parameter)
        else:
            decayed.append(parameter)

    groups = [
        # Decay is withheld here on purpose: it would pull the tables back toward their zero init
        {"params": layout, "lr": layout_learning_rate, "weight_decay": 0.0, "name": "layout"},
        {"params": decayed, "lr": learning_rate, "weight_decay": weight_decay, "name": "backbone"},
        {"params": undecayed, "lr": learning_rate, "weight_decay": 0.0, "name": "no_decay"},
    ]
    return [group for group in groups if group["params"]]


def describe_param_groups(groups: list[dict[str, Any]]) -> str:
    """
    Renders parameter groups as a human summary

    Printed before a run starts, because a layout group holding zero parameters is the single
    cheapest way to notice the graft did not happen.

    Args:
        groups: Parameter groups from `build_param_groups`.

    Returns:
        A printable multi-line string.
    """
    lines = []
    for group in groups:
        count = sum(parameter.numel() for parameter in group["params"])
        lines.append(
            f"  {group['name']:<9} {len(group['params']):>3} tensors  "
            f"{count:>12,} params  lr {group['lr']:g}  wd {group['weight_decay']:g}"
        )
    return "\n".join(lines)
