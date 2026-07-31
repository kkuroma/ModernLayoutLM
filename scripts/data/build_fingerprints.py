"""
Builds the contamination fingerprint set from the benchmark evaluation splits

FUNSD and RVL-CDIP descend from the same UCSF library as our IDL pretraining corpus, so their
evaluation pages can appear in it verbatim. This collects the Bates numbers naming those pages so
the IDL adapter can drop any pretraining page that carries one.

    uv run python scripts/data/build_fingerprints.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from src.config import load_config
from src.data.contamination import Fingerprints, bates_from_filename

# Only the splits we score on. Training pages are fair game for pretraining.
RVL_EVAL_FILES = ["data/Easy_Test_Data.npy", "data/Easy_Valid_Data.npy"]


def parse_args() -> argparse.Namespace:
    """
    Parses the command line

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="directory to write the fingerprint file into")
    return parser.parse_args()


def main() -> int:
    """
    Collects benchmark Bates numbers and writes them beside the raw IDL data

    Returns:
        0 on success.
    """
    args = parse_args()
    config = load_config()
    out = args.out or (config.paths.raw_dir / "idl")
    cache = str(config.paths.raw_dir / "hf-cache")

    bates: set[str] = set()
    for name in RVL_EVAL_FILES:
        path = hf_hub_download(
            "jordyvl/rvl_cdip_easyocr", name, repo_type="dataset", cache_dir=cache
        )
        entries = np.load(path, allow_pickle=True)
        before = len(bates)
        for entry in entries:
            images = entry.get("images") if isinstance(entry, dict) else None
            if images:
                bates |= bates_from_filename(str(images[0].get("file_name", "")))
        print(f"{name}: {len(entries)} pages, +{len(bates) - before} Bates numbers")

    fingerprints = Fingerprints(bates)
    written = fingerprints.write(out)
    print(f"\n{len(fingerprints)} fingerprints -> {written}")
    print("FUNSD derives from RVL-CDIP, so its pages are covered by the same set.")
    print("CORD and SROIE are independent of IIT-CDIP and need no filtering.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
