"""
Builds the downstream datasets into canonical parquet

Downloads on demand through the Hugging Face cache, normalizes, validates, and writes
`data/processed/<name>/<split>/*.parquet` plus a label sidecar.

    uv run python scripts/data/build_downstream.py
    uv run python scripts/data/build_downstream.py --dataset funsd cord
    uv run python scripts/data/build_downstream.py --group token --limit 20
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.config import load_config
from src.data import build_source, resolve_names


def parse_args() -> argparse.Namespace:
    """
    Parses the command line

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", help="explicit dataset names to build")
    parser.add_argument("--group", help="a named group, such as downstream, token or sequence")
    parser.add_argument("--limit", type=int, help="documents per split, for a smoke test")
    parser.add_argument("--verbose", action="store_true", help="log every split as it builds")
    return parser.parse_args()


def main() -> int:
    """
    Builds each requested dataset and prints a report

    Returns:
        0 when every dataset built, 1 when any failed.
    """
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    config = load_config()
    cache_dir = config.paths.raw_dir / "hf-cache"
    names = resolve_names(args.dataset, args.group)

    failures: list[str] = []
    for name in names:
        print(f"\n=== {name} ===", flush=True)
        try:
            report = build_source(
                name=name,
                processed_dir=config.paths.processed_dir,
                cache_dir=cache_dir,
                limit=args.limit,
            )
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            print(f"FAILED {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            failures.append(name)
            continue
        print(report.summary(), flush=True)

    if failures:
        print(f"\n{len(failures)} of {len(names)} failed: {failures}", file=sys.stderr)
        return 1
    print(f"\nBuilt {len(names)} dataset(s) into {config.paths.processed_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
