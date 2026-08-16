"""
Stage 1: acquiring datasets and reducing them to one canonical schema

Sources disagree about coordinate space, box granularity and label naming, and this absorbs it
The output is Document rows in parquet with a sidecar carrying the label vocabulary
"""

from .build import BuildReport, build_source
from .normalize import (
    COORDINATE_SCALE,
    bio_from_runs,
    clamp_box,
    normalize_box,
    quad_to_xyxy,
    split_segment,
)
from .schema import DOCUMENT_SCHEMA, DatasetMeta, Document, write_documents
from .sources import GROUPS, SOURCES, get_source, resolve_names

__all__ = [
    "COORDINATE_SCALE",
    "DOCUMENT_SCHEMA",
    "GROUPS",
    "SOURCES",
    "BuildReport",
    "DatasetMeta",
    "Document",
    "bio_from_runs",
    "build_source",
    "clamp_box",
    "get_source",
    "normalize_box",
    "quad_to_xyxy",
    "resolve_names",
    "split_segment",
    "write_documents",
]
