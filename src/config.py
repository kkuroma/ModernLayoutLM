"""
Typed access to `config.toml`

Scripts load it once and pass the dataclasses in, so nothing in `src` reads config off disk
"""

from __future__ import annotations

import ast
import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

CONFIG_FILENAME = "config.toml"


@dataclass(frozen=True)
class Paths:
    """
    Every directory the pipeline reads from or writes to, resolved against the project root
    """

    raw_dir: Path
    processed_dir: Path
    pack_dir: Path
    checkpoint_dir: Path
    backbone_dir: Path
    export_dir: Path
    report_dir: Path


@dataclass(frozen=True)
class PretrainConfig:
    """
    Hyperparameters for one phase of the layout-aware masked language modeling stage
    """

    datasets: list[str]
    max_length: int
    steps: int
    mlm_probability: float
    learning_rate: float
    layout_learning_rate: float
    weight_decay: float
    warmup_ratio: float
    batch_size: int
    grad_accum: int
    save_total_limit: int
    max_pages_per_source: int | None = None
    keep_every: int = 0
    pack_rows: int = 2000
    pack_workers: int | None = None


@dataclass(frozen=True)
class FinetuneConfig:
    """
    Hyperparameters for a single downstream fine-tuning run
    """

    max_length: int
    learning_rate: float
    baseline_learning_rate: float
    layout_learning_rate: float
    weight_decay: float
    warmup_ratio: float
    epochs: int
    batch_size: int
    grad_accum: int
    label_all_subwords: bool


@dataclass(frozen=True)
class Config:
    """
    The whole of `config.toml`, plus the per-task fine-tuning overrides

    `pretrain` is the short-window phase, and `pretrain_long` continues it with its own overrides.
    """

    paths: Paths
    pretrain: PretrainConfig
    pretrain_long: PretrainConfig
    finetune: FinetuneConfig
    finetune_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    def finetune_for(self, task: str) -> FinetuneConfig:
        """
        Returns the fine-tuning config for one task, with its `[Finetune.<task>]` table applied

        Args:
            task: The dataset name.

        Returns:
            A new FinetuneConfig, ignoring override keys it does not recognize.
        """
        known = {f.name for f in fields(FinetuneConfig)}
        overrides = self.finetune_overrides.get(task, {})
        return replace(self.finetune, **{k: v for k, v in overrides.items() if k in known})


def load_config(path: Path | None = None) -> Config:
    """
    Reads `config.toml` into typed dataclasses, with relative paths resolved against its directory

    Args:
        path: Explicit path to a config file, by default the nearest one above this module.

    Returns:
        The parsed Config.

    Raises:
        FileNotFoundError: If no config file can be located.
    """
    here = Path(__file__).resolve().parent
    candidates = (
        [path] if path is not None else [d / CONFIG_FILENAME for d in (here, *here.parents)]
    )
    found = next((c.resolve() for c in candidates if c.is_file()), None)
    if found is None:
        raise FileNotFoundError(f"No {CONFIG_FILENAME} found from {path or here}")
    with found.open("rb") as handle:
        raw = tomllib.load(handle)
    finetune = raw["Finetune"]
    pretrain = PretrainConfig(**{k: v for k, v in raw["Pretrain"].items() if k != "long"})
    return Config(
        paths=Paths(**{name: found.parent / value for name, value in raw["Paths"].items()}),
        pretrain=pretrain,
        pretrain_long=replace(pretrain, **raw["Pretrain"].get("long", {})),
        finetune=FinetuneConfig(**{k: v for k, v in finetune.items() if not isinstance(v, dict)}),
        finetune_overrides={k: v for k, v in finetune.items() if isinstance(v, dict)},
    )


def parse_overrides(items: list[str]) -> dict[str, Any]:
    """
    Turns command-line `KEY=VALUE` strings into config overrides, parsing values as literals

    Args:
        items: Strings such as `learning_rate=5e-5` or `epochs=10`.

    Returns:
        The overrides, with numbers and booleans parsed and anything else kept as a string.
    """
    overrides: dict[str, Any] = {}
    for key, value in (item.split("=", 1) for item in items):
        try:
            overrides[key] = ast.literal_eval(value)
        except ValueError, SyntaxError:
            overrides[key] = value
    return overrides
