"""
The ModernLayoutLM model: config, modeling code and processor

`register_modernlayoutlm` wires the classes into the `Auto*` factories for the current process.
Exported model directories carry an `auto_map` instead, so a fresh process can load them with
`trust_remote_code=True` without importing this package at all.

Every module in this package is copied verbatim into exported model directories, so nothing here
may import from `src`.
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
