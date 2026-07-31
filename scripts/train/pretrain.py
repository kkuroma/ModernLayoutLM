"""
Runs layout-aware masked language modeling on the pretraining corpora

This is the long run. Use `--max-pages` and `--max-steps` first to measure throughput before
committing a night to it.

    uv run python scripts/train/pretrain.py --max-pages 2000 --max-steps 50   # throughput probe
    uv run python scripts/train/pretrain.py                                    # the real run
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace

from src.config import load_config
from src.train.pretrain import pretrain


def parse_args() -> argparse.Namespace:
    """
    Parses the command line

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", help="weights to graft onto, default the configured backbone")
    parser.add_argument("--dataset", nargs="+", help="corpora to use, default the configured list")
    parser.add_argument("--run-name", default="pretrain-base", help="output directory name")
    parser.add_argument("--epochs", type=float, help="override the configured epoch count")
    parser.add_argument("--batch-size", type=int, help="override the configured batch size")
    parser.add_argument("--max-length", type=int, help="override the configured sequence length")
    parser.add_argument("--max-pages", type=int, help="cap pages per corpus, for a probe")
    parser.add_argument("--max-steps", type=int, help="cap optimizer steps, for a probe")
    parser.add_argument("--num-proc", type=int, help="worker processes for tokenization")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from the newest checkpoint in the run directory",
    )
    return parser.parse_args()


def main() -> int:
    """
    Runs the pretraining job and prints its metrics

    Returns:
        0 on success.
    """
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config()

    backbone = args.backbone or str(config.paths.backbone_dir / "ModernBERT-base")
    datasets = args.dataset or config.pretrain.datasets
    pretrain_config = config.pretrain
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
        pretrain_config = replace(pretrain_config, **overrides)

    output_dir = config.paths.checkpoint_dir / args.run_name
    print(f"=== {args.run_name} ===")
    print(f"backbone: {backbone}")
    print(f"corpora:  {datasets}")
    print(f"config:   {pretrain_config}")

    result = pretrain(
        backbone=backbone,
        datasets=datasets,
        config=pretrain_config,
        processed_dir=config.paths.processed_dir,
        output_dir=output_dir,
        seed=args.seed,
        num_proc=args.num_proc,
        max_pages=args.max_pages,
        max_steps=args.max_steps,
        resume=args.resume,
    )
    print(f"\n{result.summary()}")
    print(f"written to {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
