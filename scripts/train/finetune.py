"""
Fine-tunes a model on one downstream dataset

Handles the real runs and the ablation controls through the same code path, so a control is never
an approximation of the real recipe.

    uv run python scripts/train/finetune.py --dataset funsd
    uv run python scripts/train/finetune.py --dataset funsd --no-bbox
    uv run python scripts/train/finetune.py --dataset cord --backbone data/checkpoints/pretrain-base
    uv run python scripts/train/finetune.py --dataset funsd --epochs 1 --limit-steps 20
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from src.config import load_config
from src.train import RunSpec, finetune


def parse_args() -> argparse.Namespace:
    """
    Parses the command line

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="a processed dataset name")
    parser.add_argument("--backbone", help="weights to start from, default the configured backbone")
    parser.add_argument("--tokenizer", help="tokenizer to use, default the backbone's own")
    parser.add_argument("--run-name", help="output directory name, default derived from the run")
    parser.add_argument(
        "--no-bbox",
        action="store_true",
        help="withhold boxes entirely, which is the no-bbox control",
    )
    parser.add_argument("--epochs", type=int, help="override the configured epoch count")
    parser.add_argument("--batch-size", type=int, help="override the configured batch size")
    parser.add_argument("--max-length", type=int, help="override the configured sequence length")
    parser.add_argument("--num-proc", type=int, help="worker processes for tokenization")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    return parser.parse_args()


def main() -> int:
    """
    Runs one fine-tuning job and prints its metrics

    Returns:
        0 on success.
    """
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config()

    backbone = args.backbone or str(config.paths.backbone_dir / "ModernBERT-base")
    variant = "nobbox" if args.no_bbox else "layout"
    run_name = args.run_name or f"{args.dataset}-{variant}-{Path(backbone).name}"

    spec = RunSpec(
        task=args.dataset,
        backbone=backbone,
        run_name=run_name,
        use_bbox=not args.no_bbox,
        tokenizer_name=args.tokenizer,
    )
    finetune_config = config.finetune_for(args.dataset)
    overrides = {
        key: value
        for key, value in (
            ("epochs", args.epochs),
            ("batch_size", args.batch_size),
            ("max_length", args.max_length),
        )
        if value is not None
    }
    if overrides:
        finetune_config = replace(finetune_config, **overrides)

    print(f"=== {run_name} ===")
    print(f"backbone:  {backbone}")
    print(f"boxes:     {'withheld (no-bbox control)' if args.no_bbox else 'used'}")
    print(f"config:    {finetune_config}")

    result = finetune(
        spec=spec,
        config=finetune_config,
        processed_dir=config.paths.processed_dir,
        output_root=config.paths.checkpoint_dir / "finetune",
        seed=args.seed,
        num_proc=args.num_proc,
    )
    print(f"\n{result.summary()}")
    print(f"scored on the {result.eval_split} split, {result.train_examples} training chunks")
    print(f"written to {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
