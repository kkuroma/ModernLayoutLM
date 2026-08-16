"""
The ModernLayoutLM model: config, modelling code and processor

register_modernlayoutlm covers the current process and an exported auto_map covers a fresh one
Every module here is copied verbatim into exported directories, so nothing may import from src
"""

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
from .registry import AUTO_MAP, register_modernlayoutlm

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
