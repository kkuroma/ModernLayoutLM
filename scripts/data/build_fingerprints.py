"""
Builds the contamination fingerprint set from the benchmark evaluation splits

RVL-CDIP contributes its Bates numbers through file names
FUNSD has no file names in the mirror, so its numbers are read as long digit runs in the test OCR
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from src.config import load_config
from src.data.contamination import Fingerprints, bates_from_filename, page_bates

# Training pages are fair game for pretraining, so only the scored splits count
RVL_EVAL_FILES = ["data/Easy_Test_Data.npy", "data/Easy_Valid_Data.npy"]

# FUNSD ships no separate validation split, so its test split is everything we score on
FUNSD_EVAL_SPLITS = ["test"]


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

    for split in FUNSD_EVAL_SPLITS:
        shards = sorted((config.paths.processed_dir / "funsd" / split).glob("*.parquet"))
        pages = load_dataset("parquet", data_files=[str(p) for p in shards], split="train")
        before = len(bates)
        for words in cast(list[list[str]], pages["words"]):
            bates |= page_bates(words)
        print(f"funsd/{split}: {len(pages)} pages, +{len(bates) - before} digit runs")

    fingerprints = Fingerprints(bates)
    written = fingerprints.write(out)
    print(f"\n{len(fingerprints)} fingerprints -> {written}")
    print("Filtered against by the IDL and RVL-CDIP train adapters.")
    print("CORD and SROIE are independent of IIT-CDIP and need no filtering.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
