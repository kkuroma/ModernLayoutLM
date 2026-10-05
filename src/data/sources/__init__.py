"""
One adapter per dataset, behind a common interface

Every adapter exposes describe and load, so build never knows which dataset it is handling
Adding a dataset means writing those two functions and listing the module here
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Protocol

from ..schema import DatasetMeta, Document
from . import cord, doclaynet, funsd, idl, pdfa, pubtables, realkie, sroie


class Source(Protocol):
    """The interface every dataset adapter satisfies."""

    SPLITS: dict[str, str]
    describe: Callable[[], DatasetMeta]
    load: Callable[..., Iterator[Document]]


SOURCES: dict[str, Source] = {
    "funsd": funsd,  # type: ignore[dict-item]
    "cord": cord,  # type: ignore[dict-item]
    "sroie": sroie,  # type: ignore[dict-item]
    "pubtables": pubtables,  # type: ignore[dict-item]
    "doclaynet": doclaynet,  # type: ignore[dict-item]
    "pdfa": pdfa,  # type: ignore[dict-item]
    "idl": idl,  # type: ignore[dict-item]
    "idl-fresh": idl.Fresh,  # type: ignore[dict-item]
    "idl-docs": idl.Docs,  # type: ignore[dict-item]
    "realkie-fcc": realkie.FCC,
    "realkie-nda": realkie.NDA,
    "realkie-charities": realkie.CHARITIES,
}

# Groups the scripts expose as `--group`, so a caller need not spell out every name
GROUPS = {
    "downstream": ["funsd", "cord", "sroie"],
    "pretrain": ["idl", "pubtables", "pdfa", "doclaynet"],
    "realkie": ["realkie-fcc", "realkie-nda", "realkie-charities"],
}


def get_source(name: str) -> Source:
    """
    Looks up a dataset adapter by name

    Args:
        name: The dataset name, a key of `SOURCES`.

    Returns:
        The adapter module.

    Raises:
        KeyError: If no adapter is registered under that name.
    """
    if name not in SOURCES:
        raise KeyError(f"Unknown source {name!r}. Known: {sorted(SOURCES)}")
    return SOURCES[name]


def resolve_names(names: list[str] | None, group: str | None) -> list[str]:
    """
    Turns a name list or a group name into a concrete list of datasets

    Args:
        names: Explicit dataset names, or None.
        group: A key of `GROUPS`, or None.

    Returns:
        The dataset names to process, in a stable order.

    Raises:
        KeyError: If the group is unknown or a name is not registered.
    """
    if names:
        for name in names:
            get_source(name)
        return names
    if group:
        if group not in GROUPS:
            raise KeyError(f"Unknown group {group!r}. Known: {sorted(GROUPS)}")
        return list(GROUPS[group])
    return list(GROUPS["downstream"])


__all__ = ["GROUPS", "SOURCES", "Source", "get_source", "resolve_names"]
