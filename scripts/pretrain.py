"""
Runs layout-aware masked language modeling in two phases, short windows then long ones

Each phase continues from the last and is skipped once its result exists, so a rerun resumes
Packing is CPU bound, so --pack-only packs ahead and the run starts at its first optimizer step
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace

from transformers import AutoTokenizer

from src.config import load_config, parse_overrides
from src.model import ModernLayoutLmProcessor
from src.train.pretrain import corpus, pretrain

PHASES = ("short", "long")


def parse_args() -> argparse.Namespace:
    """
    Parses the command line

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", help="weights to graft onto or continue, default ModernBERT")
    parser.add_argument("--dataset", nargs="+", help="corpora for every phase run, default config")
    parser.add_argument("--phase", choices=(*PHASES, "both"), default="both", help="phases to run")
    parser.add_argument("--run-name", default="pretrain-base", help="name, suffixed by length")
    parser.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE", help="config override")
    parser.add_argument("--max-pages", type=int, help="cap pages per corpus")
    parser.add_argument("--max-steps", type=int, help="cap optimizer steps, for a probe")
    parser.add_argument("--num-proc", type=int, help="packing processes, default the config")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--pack-only", action="store_true", help="build the pack cache and stop")
    parser.add_argument("--resume", action="store_true", help="continue from the newest checkpoint")
    return parser.parse_args()


def main() -> int:
    """
    Runs the selected phases in order, or only their packing, and prints each outcome

    Returns:
        0 on success.
    """
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config()
    backbone = args.backbone or str(config.paths.backbone_dir / "ModernBERT-base")
    phases = {"short": config.pretrain, "long": config.pretrain_long}
    overrides = parse_overrides(args.set) | ({"datasets": args.dataset} if args.dataset else {})

    for phase in PHASES if args.phase == "both" else (args.phase,):
        settings = replace(phases[phase], **overrides)
        output_dir = config.paths.checkpoint_dir / f"{args.run_name}-{settings.max_length}"
        print(f"=== {output_dir.name}, {phase} phase ===\nbackbone: {backbone}\n{settings}")

        if args.pack_only:
            processor = ModernLayoutLmProcessor(tokenizer=AutoTokenizer.from_pretrained(backbone))
            train, validation = corpus(
                processor,
                settings.datasets,
                settings,
                config.paths.processed_dir,
                config.paths.pack_dir,
                args.num_proc or settings.pack_workers,
                args.max_pages,
                args.seed,
            )
            print(f"packed {len(train):,} train and {len(validation):,} validation sequences")
            continue

        if (output_dir / "pretrain_result.json").is_file():
            print(f"{output_dir} already finished, skipped")
        else:
            result = pretrain(
                backbone=backbone,
                datasets=settings.datasets,
                config=settings,
                processed_dir=config.paths.processed_dir,
                pack_dir=config.paths.pack_dir,
                output_dir=output_dir,
                seed=args.seed,
                num_proc=args.num_proc or settings.pack_workers,
                max_pages=args.max_pages,
                max_steps=args.max_steps,
                resume=args.resume,
            )
            print(f"\n{result.train_sequences} train sequences, {result.metrics}")
            print(f"written to {result.output_dir}")
        backbone = str(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
