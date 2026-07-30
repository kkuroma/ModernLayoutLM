# data scripts

Acquiring datasets and reducing them to canonical parquet.

## Files

```
build_datasets.py    download, normalize, validate and shard any registered dataset
inspect_dataset.py   print a processed dataset and optionally render its boxes
```

## Running

```sh
uv run python scripts/data/build_datasets.py --group downstream   # funsd cord sroie rvlcdip
uv run python scripts/data/build_datasets.py --group pretrain     # pubtables doclaynet
uv run python scripts/data/build_datasets.py --dataset cord --limit 20
uv run python scripts/data/inspect_dataset.py --dataset cord --render scratchpad/cord.png
```

`--limit` overwrites the full build for that dataset, so rebuild without it afterwards.

`inspect_dataset.py --render` is worth the trouble. Validation catches boxes outside the 0-1000
grid, but only a picture catches boxes that are in range and still wrong, such as a transposed x
and y, or COCO `(x, y, w, h)` read as `(x0, y0, x1, y1)`.
