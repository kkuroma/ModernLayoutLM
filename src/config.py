"""
Typed access to `config.toml`

Scripts load the file once and hand the resulting dataclasses to `src`, so no module inside
`src` ever reads a path or hyperparameter off disk by itself.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

CONFIG_FILENAME = "config.toml"


@dataclass(frozen=True)
class Paths:
    """
    Every directory the pipeline reads from or writes to, resolved against the project root
    """

    root: Path
    raw_dir: Path
    processed_dir: Path
    checkpoint_dir: Path
    backbone_dir: Path
    export_dir: Path
    report_dir: Path


@dataclass(frozen=True)
class ModelConfig:
    """
    The backbone to graft onto and the geometry of the layout embeddings
    """

    backbone: str
    max_2d_position_embeddings: int
    coordinate_scale: int
    zero_init_2d: bool


@dataclass(frozen=True)
class PretrainConfig:
    """
    Hyperparameters for the layout-aware masked language modelling stage
    """

    datasets: list[str]
    max_length: int
    mlm_probability: float
    learning_rate: float
    layout_learning_rate: float
    weight_decay: float
    warmup_ratio: float
    epochs: int
    batch_size: int
    grad_accum: int
    save_total_limit: int


@dataclass(frozen=True)
class FinetuneConfig:
    """
    Hyperparameters for a single downstream fine-tuning run
    """

    max_length: int
    learning_rate: float
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
    """

    paths: Paths
    model: ModelConfig
    pretrain: PretrainConfig
    finetune: FinetuneConfig
    finetune_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    def finetune_for(self, task: str) -> FinetuneConfig:
        """
        Returns the fine-tuning config for one task, with that task's overrides applied

        Reads the `[Finetune.<task>]` table and replaces only the keys it declares, so a task
        table stays a short diff against the shared defaults.

        Args:
            task: The dataset name, matching a `[Finetune.<task>]` table.

        Returns:
            A new FinetuneConfig. Unrecognised override keys are ignored.
        """
        overrides = self.finetune_overrides.get(task, {})
        known = {f.name for f in fields(FinetuneConfig)}
        merged = {f.name: getattr(self.finetune, f.name) for f in fields(FinetuneConfig)}
        merged.update({k: v for k, v in overrides.items() if k in known})
        return FinetuneConfig(**merged)


def find_project_root(start: Path | None = None) -> Path:
    """
    Walks upward from `start` until it finds the directory holding `config.toml`

    Args:
        start: Directory to begin the search from. Defaults to this file's location.

    Returns:
        The project root directory.

    Raises:
        FileNotFoundError: If no ancestor directory contains `config.toml`.
    """
    current = (start or Path(__file__).resolve().parent).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / CONFIG_FILENAME).is_file():
            return candidate
    raise FileNotFoundError(f"No {CONFIG_FILENAME} found above {current}")


def load_config(path: Path | None = None) -> Config:
    """
    Reads `config.toml` and returns it as typed dataclasses

    Relative paths in the `[Paths]` table are resolved against the project root, so the
    `data` symlink and `models` directory work regardless of the current directory.

    Args:
        path: Explicit path to a config file. Defaults to the one at the project root.

    Returns:
        The parsed Config.

    Raises:
        FileNotFoundError: If no config file can be located.
    """
    config_path = path.resolve() if path is not None else find_project_root() / CONFIG_FILENAME
    if not config_path.is_file():
        raise FileNotFoundError(f"No config file at {config_path}")
    root = config_path.parent

    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    raw_paths = raw["Paths"]
    paths = Paths(
        root=root,
        raw_dir=root / raw_paths["raw_dir"],
        processed_dir=root / raw_paths["processed_dir"],
        checkpoint_dir=root / raw_paths["checkpoint_dir"],
        backbone_dir=root / raw_paths["backbone_dir"],
        export_dir=root / raw_paths["export_dir"],
        report_dir=root / raw_paths["report_dir"],
    )

    raw_finetune = raw["Finetune"]
    overrides = {k: v for k, v in raw_finetune.items() if isinstance(v, dict)}
    defaults = {k: v for k, v in raw_finetune.items() if not isinstance(v, dict)}

    return Config(
        paths=paths,
        model=ModelConfig(**raw["Model"]),
        pretrain=PretrainConfig(**raw["Pretrain"]),
        finetune=FinetuneConfig(**defaults),
        finetune_overrides=overrides,
    )
