"""
Eyeballs a processed dataset, to confirm normalization by looking rather than by assertion

Prints a summary and, optionally, renders the normalized boxes back onto a blank page as an image.
Assertions catch boxes outside the scale, but only a picture catches boxes that are in range and
still in the wrong place, such as a transposed x and y.

    uv run python scripts/data/inspect_dataset.py --dataset funsd
    uv run python scripts/data/inspect_dataset.py --dataset cord --render scratchpad/cord.png
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, ImageDraw

from src.config import load_config
from src.data import COORDINATE_SCALE, DatasetMeta

# Rendered page size in pixels. Boxes are on a 0-1000 grid, so this is a straight scale.
RENDER_SIZE = (700, 900)


def parse_args() -> argparse.Namespace:
    """
    Parses the command line

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="a processed dataset name")
    parser.add_argument("--split", default="train", help="which split to inspect")
    parser.add_argument("--index", type=int, default=0, help="which document to show")
    parser.add_argument("--words", type=int, default=15, help="how many words to print")
    parser.add_argument("--render", type=Path, help="write a box overlay image here")
    return parser.parse_args()


def main() -> int:
    """
    Prints a dataset summary and one document, optionally rendering its boxes

    Returns:
        0 on success, 1 when the dataset or split is missing.
    """
    args = parse_args()
    config = load_config()
    directory = config.paths.processed_dir / args.dataset

    if not directory.is_dir():
        print(f"No processed dataset at {directory}. Build it first.")
        return 1
    meta = DatasetMeta.read(directory)
    shards = sorted((directory / args.split).glob("*.parquet"))
    if not shards:
        print(f"No shards for split {args.split!r} in {directory}. Splits: {list(meta.splits)}")
        return 1

    print(f"=== {meta.source} ===")
    print(f"task:   {meta.task}")
    print(f"splits: {meta.splits}")
    print(f"labels: {len(meta.label_names)}")
    if meta.notes:
        print(f"notes:  {meta.notes}")

    table = pq.read_table(shards[0])
    rows = table.to_pylist()
    row = rows[min(args.index, len(rows) - 1)]

    print(f"\n--- {row['doc_id']} ---")
    print(f"words: {len(row['words'])}  page: {row['page_width']}x{row['page_height']}")
    if row["doc_label"] is not None:
        print(f"doc label: {row['doc_label']} = {meta.label_names[row['doc_label']]}")

    _print_words(row, meta, args.words)
    _print_extremes(rows)

    if args.render:
        _render(row, args.render)
        print(f"\nrendered to {args.render}")
    return 0


def _print_words(row: dict, meta: DatasetMeta, count: int) -> None:
    """
    Prints the first words with their boxes and labels

    Args:
        row: One document as a dict.
        meta: The dataset metadata, for label names.
        count: How many words to print.
    """
    labels = row.get("word_labels")
    print(f"\n{'box':<26} {'label':<22} word")
    for index in range(min(count, len(row["words"]))):
        box = row["bboxes"][index]
        name = ""
        if labels is not None:
            name = meta.label_names[labels[index]] if meta.label_names else str(labels[index])
        print(f"{str(box):<26} {name:<22} {row['words'][index]!r}")


def _print_extremes(rows: list[dict]) -> None:
    """
    Prints coordinate extremes and label frequency across the shard

    Boxes that never approach the edges of the grid, or a label histogram collapsed onto one id,
    both point at a normalization or label-scheme bug.

    Args:
        rows: All documents in the shard.
    """
    max_x = max_y = 0
    min_x = min_y = COORDINATE_SCALE
    labels: Counter[int] = Counter()
    for row in rows:
        for box in row["bboxes"]:
            min_x, min_y = min(min_x, box[0]), min(min_y, box[1])
            max_x, max_y = max(max_x, box[2]), max(max_y, box[3])
        if row.get("word_labels"):
            labels.update(row["word_labels"])
        if row.get("doc_label") is not None:
            labels.update([row["doc_label"]])

    print(f"\nacross {len(rows)} documents in this shard:")
    print(f"  x range {min_x}..{max_x}   y range {min_y}..{max_y}   (scale is {COORDINATE_SCALE})")
    print(f"  most common labels: {labels.most_common(6)}")


def _render(row: dict, path: Path) -> None:
    """
    Draws the document's normalized boxes onto a blank page

    Args:
        row: One document as a dict.
        path: Destination image file.
    """
    width, height = RENDER_SIZE
    image = Image.new("RGB", RENDER_SIZE, "white")
    draw = ImageDraw.Draw(image)
    for box, word in zip(row["bboxes"], row["words"], strict=True):
        pixels = (
            box[0] / COORDINATE_SCALE * width,
            box[1] / COORDINATE_SCALE * height,
            box[2] / COORDINATE_SCALE * width,
            box[3] / COORDINATE_SCALE * height,
        )
        draw.rectangle(pixels, outline="crimson")
        draw.text((pixels[0], pixels[1]), word[:12], fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


if __name__ == "__main__":
    raise SystemExit(main())
