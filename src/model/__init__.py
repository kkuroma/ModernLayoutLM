"""
The ModernLayoutLM model: config, modeling code and processor

The three modules are copied verbatim into exports, so they import only torch and transformers
Registration covers this process and AUTO_MAP covers an exported directory
"""

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)

from .configuration_modernlayoutlm import ModernLayoutLmConfig
from .modeling_modernlayoutlm import (
    ModernLayoutLmEmbeddings,
    ModernLayoutLmForMaskedLM,
    ModernLayoutLmForSequenceClassification,
    ModernLayoutLmForTokenClassification,
    ModernLayoutLmModel,
    ModernLayoutLmPreTrainedModel,
)
from .processing_modernlayoutlm import ModernLayoutLmProcessor

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


def register_modernlayoutlm() -> None:
    """
    Makes the Auto factories recognize the `modernlayoutlm` model type in this process, idempotently
    """
    AutoConfig.register(ModernLayoutLmConfig.model_type, ModernLayoutLmConfig, exist_ok=True)
    for auto, model in (
        (AutoModel, ModernLayoutLmModel),
        (AutoModelForMaskedLM, ModernLayoutLmForMaskedLM),
        (AutoModelForTokenClassification, ModernLayoutLmForTokenClassification),
        (AutoModelForSequenceClassification, ModernLayoutLmForSequenceClassification),
    ):
        auto.register(ModernLayoutLmConfig, model, exist_ok=True)


__all__ = [
    "AUTO_MAP",
    "ModernLayoutLmConfig",
    "ModernLayoutLmEmbeddings",
    "ModernLayoutLmForMaskedLM",
    "ModernLayoutLmForSequenceClassification",
    "ModernLayoutLmForTokenClassification",
    "ModernLayoutLmModel",
    "ModernLayoutLmPreTrainedModel",
    "ModernLayoutLmProcessor",
    "register_modernlayoutlm",
]
