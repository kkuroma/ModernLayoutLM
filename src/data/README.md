# data

Stage 1. Downloads raw datasets and reduces all of them to one canonical schema, so that no later
stage has to know how any particular source spells things.

## Files

```
schema.py     Document, DatasetMeta, the pyarrow schema, and validation
normalize.py  coordinate primitives every adapter shares
build.py      drives an adapter: stream, validate, shard to parquet, write the sidecar
sources/      one module per dataset, each exposing describe() and load()
```

## The canonical form

`data/raw/<name>/` holds sources exactly as downloaded. `data/processed/<name>/<split>/*.parquet`
holds the canonical rows:

| field | meaning |
|---|---|
| `doc_id`, `source` | provenance |
| `words` | words in the source's reading order |
| `bboxes` | one box per word, `(x0, y0, x1, y1)`, integers in `[0, 1000]`, origin top left |
| `page_width`, `page_height` | original pixel size, for rendering and debugging only |
| `word_labels` | one label per word for token tasks, else null |
| `doc_label` | one label per page for sequence tasks, else null |

`dataset_meta.json` sits beside the splits and carries the label vocabulary, the task, and any
caveat needed to interpret the numbers. The label set is never inferred from whichever rows happen
to be present, or a split missing a rare field would silently renumber every label.

## Validation is fatal on purpose

`Document.validate` runs on every document at build time, and any adapter that hands it a
misaligned document stops the build. This is not defensive politeness. A document whose boxes and
words are off by one still trains, still shows a falling loss, and produces a model that is simply
wrong. Failing at build time costs seconds; failing silently costs a GPU day and a meaningless
number.

The same principle governs unknown label values: adapters raise rather than mapping them to the
outside class. Both times that rule fired, it was right (see below).

## Adding a source

Write `sources/<name>.py` exposing `SPLITS`, `describe() -> DatasetMeta` and
`load(split, cache_dir) -> Iterator[Document]`, then list it in `sources/__init__.py`. `build.py`
needs to know nothing else.

## What each source actually required

**FUNSD** (`nielsr/funsd-layoutlmv3`), 149/50 forms, 7 BIO labels. The clean one. Its boxes are
already on the 0-1000 scale, which is easy to misread as pixels because FUNSD pages happen to be
about 1000 pixels tall. The adapter asserts the scale rather than assuming it.

**CORD** (`naver-clova-ix/cord-v2`), 800/100/100 receipts, 59 BIO labels. Words are buried in a
JSON string, as four-corner quads in pixels, and the field name lives on the line rather than the
word. The published category list is also wrong about the released data: it writes `menu.sub_nm`
where the data says `menu.sub.nm`, and lists a `menu.sub.etc` that does not exist. 29 categories,
enumerated from all three splits.

**SROIE** (`Ssunbell/SROIE_layoutlmv2_sequence`), 594/32/347 receipts, 9 BIO labels. The original
`darentang/sroie` is a loading script, which `datasets` v4 dropped, and its data comes off a Google
Drive link. This mirror works but stores labels as bare integers with no key, and the scheme is not
the conventional one: **`0` is COMPANY and `4` is outside.** Receipt 1 labels
"OLD TOWN KOPITIAM SDN BHD" as 0 and "TAX INVOICE", "GST ID" and the line items as 4. Assuming
`0` meant O would have trained the model to find companies everywhere. The source also omits B and
I, so BIO is reconstructed from contiguous runs, which merges two adjacent entities of the same
field. SROIE numbers here are therefore not strictly comparable to published ones.

**RVL-CDIP** (`jordyvl/rvl_cdip_easyocr`), ~400k pages, 16 classes. The standard release is images
only; this mirror carries precomputed EasyOCR, so no OCR pass is needed. Two wrinkles. It is a
legacy script dataset, so the `.npy` payloads are fetched and unpickled directly. And its
"word-level annotations" are really segment level, with phrases like
"Plasminogen Activator Activity" sharing one box, so segments are split into words with the box
divided along x by character count.

It also spells four classes differently from the dataset paper: `report`, `publication`,
`file_folder` and `news_article`. The first build skipped unknown classes and silently produced 12
of 16 classes, quietly discarding about a quarter of the corpus. That is exactly why unknown labels
now raise.

## OCR

There is no OCR in the pipeline, because no source needs it. PaddleOCR is also not installable
here: `paddlepaddle` publishes no cp314 wheels, and declaring it even as an optional extra makes
the whole lockfile unresolvable. If a future source ships images without text, the OCR pass belongs
in a separate py3.13 virtualenv writing canonical parquet, not in this dependency tree.

## Running

```sh
uv run python scripts/data/build_downstream.py --group token
uv run python scripts/data/build_downstream.py --dataset rvlcdip
uv run python scripts/data/inspect_dataset.py --dataset cord --render scratchpad/cord.png
```

`inspect_dataset.py` earns its place: assertions catch boxes outside the grid, but only the render
catches boxes that are in range and still wrong, such as a transposed x and y.
