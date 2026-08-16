"""
Tokenizes and packs the pretraining corpus ahead of the training run

Packing is CPU bound, so doing it here lets a later run start at its first optimizer step
The cache key covers the split, sequence length, corpus list and page cap from `config.toml`
"""

from __future__ import annotations

import argparse
import logging

from transformers import AutoTokenizer

from src.config import load_config
from src.model import ModernLayoutLmProcessor
from src.train.encoding import build_packed_corpus
from src.train.pretrain import EVAL_PAGES


def parse_args() -> argparse.Namespace:
    """
    Parses the command line

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", help="tokenizer to encode with, default the configured one")
    parser.add_argument("--dataset", nargs="+", help="corpora to pack, default the configured list")
    parser.add_argument(
        "--max-pages", type=int, help="cap pages per corpus, default the configured one"
    )
    parser.add_argument("--num-proc", type=int, help="worker processes for tokenization")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    return parser.parse_args()


def main() -> int:
    """
    Builds and caches the packed train and validation corpora

    Returns:
        0 on success.
    """
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config()

    backbone = args.backbone or str(config.paths.backbone_dir / "ModernBERT-base")
    datasets = args.dataset or config.pretrain.datasets
    max_pages = (
        args.max_pages if args.max_pages is not None else config.pretrain.max_pages_per_source
    )

    tokenizer = AutoTokenizer.from_pretrained(backbone)
    processor = ModernLayoutLmProcessor(tokenizer=tokenizer)
    cache = config.paths.processed_dir.parent / "packed"

    # The same caps the training run uses, or the keys differ and it rebuilds anyway
    for split, cap in (("train", max_pages), ("validation", EVAL_PAGES // len(datasets))):
        packed = build_packed_corpus(
            config.paths.processed_dir,
            datasets,
            split,
            processor,
            config.pretrain.max_length,
            cache,
            args.num_proc,
            cap,
            args.seed,
        )
        print(f"{split}: {len(packed):,} sequences of {config.pretrain.max_length} tokens")

    print(f"cached under {cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
