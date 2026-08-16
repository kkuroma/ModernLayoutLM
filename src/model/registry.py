"""
Registering ModernLayoutLM with the Auto factories

register_modernlayoutlm covers the current process and AUTO_MAP covers an exported directory
This module is copied into exported model directories and may import only from transformers
"""

from __future__ import annotations

from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)

from .configuration_modernlayoutlm import ModernLayoutLmConfig
from .modeling_modernlayoutlm import (
    ModernLayoutLmForMaskedLM,
    ModernLayoutLmForSequenceClassification,
    ModernLayoutLmForTokenClassification,
    ModernLayoutLmModel,
)

# Values are "<module file stem>.<class name>", resolved relative to the exported directory
AUTO_MAP = {
    "AutoConfig": "configuration_modernlayoutlm.ModernLayoutLmConfig",
    "AutoModel": "modeling_modernlayoutlm.ModernLayoutLmModel",
    "AutoModelForMaskedLM": "modeling_modernlayoutlm.ModernLayoutLmForMaskedLM",
    "AutoModelForTokenClassification": (
        "modeling_modernlayoutlm.ModernLayoutLmForTokenClassification"
    ),
    "AutoModelForSequenceClassification": (
        "modeling_modernlayoutlm.ModernLayoutLmForSequenceClassification"
    ),
}

_AUTO_CLASSES = (
    (AutoModel, ModernLayoutLmModel),
    (AutoModelForMaskedLM, ModernLayoutLmForMaskedLM),
    (AutoModelForTokenClassification, ModernLayoutLmForTokenClassification),
    (AutoModelForSequenceClassification, ModernLayoutLmForSequenceClassification),
)


def register_modernlayoutlm() -> None:
    """
    Makes the `Auto*` factories recognize the `modernlayoutlm` model type in this process

    Idempotent, so importing it from several entrypoints is safe.
    """
    AutoConfig.register(ModernLayoutLmConfig.model_type, ModernLayoutLmConfig, exist_ok=True)
    for auto_class, model_class in _AUTO_CLASSES:
        auto_class.register(ModernLayoutLmConfig, model_class, exist_ok=True)


__all__ = ["AUTO_MAP", "register_modernlayoutlm"]
