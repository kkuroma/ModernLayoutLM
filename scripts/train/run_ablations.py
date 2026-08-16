"""
Runs every task against every variant so each control shares the fine-tuning code path

Variants are layout, nobbox, skipmlm and the LayoutLM baseline
A variant that already wrote result.json is skipped unless --force is given
"""

from __future__ import annotations

import argparse
import logging
import time
import traceback
from dataclasses import replace
from pathlib import Path

from src.config import load_config
from src.train import RunSpec, finetune

LAYOUTLM_BASELINE = "microsoft/layoutlm-base-uncased"

DATASETS = ["funsd", "cord", "sroie"]
VARIANTS = ["layout", "nobbox", "skipmlm", "layoutlm"]


def parse_args() -> argparse.Namespace:
    """
    Parses the command line

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--variant", nargs="+", default=VARIANTS, choices=VARIANTS)
    parser.add_argument("--pretrained", help="the layout-MLM checkpoint, default pretrain-base")
    parser.add_argument("--epochs", type=int, help="override every run's epoch count")
    parser.add_argument("--num-proc", type=int, default=8, help="tokenization workers")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="rerun even if a result exists")
    parser.add_argument("--dry-run", action="store_true", help="list the runs, execute nothing")
    return parser.parse_args()


def build_spec(task: str, variant: str, pretrained: str, backbone: str) -> RunSpec:
    """
    Describes one cell of the ablation matrix

    Args:
        task: The dataset name.
        variant: One of `VARIANTS`.
        pretrained: Path to the layout-MLM checkpoint.
        backbone: Path to the raw ModernBERT backbone.

    Returns:
        The RunSpec for that cell.

    Raises:
        ValueError: If the variant is not recognized.
    """
    name = f"{task}-{variant}"
    if variant == "layout":
        return RunSpec(task=task, backbone=pretrained, run_name=name)
    if variant == "nobbox":
        return RunSpec(task=task, backbone=pretrained, run_name=name, use_bbox=False)
    if variant == "skipmlm":
        return RunSpec(task=task, backbone=backbone, run_name=name)
    if variant == "layoutlm":
        return RunSpec(task=task, backbone=LAYOUTLM_BASELINE, run_name=name)
    raise ValueError(f"Unknown variant {variant!r}")


def main() -> int:
    """
    Runs every requested cell of the matrix, reporting as it goes

    A failed cell is recorded and the matrix continues, because losing fifteen good runs to one
    bad one would be worse than an incomplete table.

    Returns:
        0 when every run succeeded, 1 when any failed.
    """
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config()

    backbone = str(config.paths.backbone_dir / "ModernBERT-base")
    pretrained = args.pretrained or str(config.paths.checkpoint_dir / "pretrain-base")
    output_root = config.paths.checkpoint_dir / "finetune"

    if not Path(pretrained, "config.json").is_file() and "layout" in args.variant:
        print(f"No layout-MLM checkpoint at {pretrained}. Run scripts/train/pretrain.py first.")
        return 1

    planned = [
        build_spec(task, variant, pretrained, backbone)
        for task in args.dataset
        for variant in args.variant
    ]
    pending = [
        spec
        for spec in planned
        if args.force or not (output_root / spec.run_name / "result.json").is_file()
    ]

    print(f"{len(planned)} runs planned, {len(planned) - len(pending)} already done")
    for spec in pending:
        print(f"  {spec.run_name:24s} backbone={spec.backbone} bbox={spec.use_bbox}")
    if args.dry_run:
        return 0

    failures: list[str] = []
    for index, spec in enumerate(pending, start=1):
        finetune_config = config.finetune_for(spec.task)
        if args.epochs is not None:
            finetune_config = replace(finetune_config, epochs=args.epochs)

        print(f"\n{'=' * 70}\n[{index}/{len(pending)}] {spec.run_name}\n{'=' * 70}", flush=True)
        started = time.perf_counter()
        try:
            result = finetune(
                spec=spec,
                config=finetune_config,
                processed_dir=config.paths.processed_dir,
                output_root=output_root,
                seed=args.seed,
                num_proc=args.num_proc,
            )
        except OSError, ValueError, KeyError, RuntimeError:
            print(f"FAILED {spec.run_name}", flush=True)
            traceback.print_exc()
            failures.append(spec.run_name)
            continue
        minutes = (time.perf_counter() - started) / 60
        print(f"{result.summary()}   [{minutes:.1f} min]", flush=True)

    if failures:
        print(f"\n{len(failures)} of {len(pending)} failed: {failures}")
        return 1
    print(f"\nAll {len(pending)} runs finished. Now: uv run python scripts/eval/evaluate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
