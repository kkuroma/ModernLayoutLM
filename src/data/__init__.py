"""
Stage 1: acquiring datasets and reducing them to one canonical schema

Sources disagree about coordinate space, box granularity and label naming, and this absorbs it
The output is Document rows in parquet with a sidecar carrying the label vocabulary
"""

from .build import BuildReport, build_source
from .schema import COORDINATE_SCALE, DOCUMENT_SCHEMA, DatasetMeta, Document, write_documents
from .sources import GROUPS, SOURCES, get_source, resolve_names

__all__ = [
    "COORDINATE_SCALE",
    "DOCUMENT_SCHEMA",
    "GROUPS",
    "SOURCES",
    "BuildReport",
    "DatasetMeta",
    "Document",
    "build_source",
    "get_source",
    "resolve_names",
    "write_documents",
]
