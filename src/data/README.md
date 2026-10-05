# data

Stage 1.
Downloads raw datasets and reduces all of them to one canonical schema, so that no later stage has to know how any particular source spells things.

## Files

```
schema.py        Document, DatasetMeta, the pyarrow schema, validation, coordinate primitives
contamination.py Bates fingerprints, so a pretraining page that is a benchmark page is dropped
build.py         drives an adapter: stream, validate, shard to parquet, write the sidecar
sources/         one module per dataset, each exposing describe() and load()
```

Acquisition lives here too, not in the scripts.
`scripts/` holds argument parsing and printing and nothing else, so every step that touches a document is importable and testable.

## The canonical form

`data/raw/<name>/` holds sources exactly as downloaded.
`data/processed/<name>/<split>/*.parquet` holds the canonical rows:

| field | meaning |
|---|---|
| `doc_id`, `source` | provenance |
| `words` | words in the source's reading order |
| `bboxes` | one box per word, `(x0, y0, x1, y1)`, integers in `[0, 1000]`, origin top left |
| `page_width`, `page_height` | original pixel size, for rendering and debugging only |
| `word_labels` | one label per word for token tasks, else null |
| `doc_label` | one label per page for sequence tasks, else null |

`dataset_meta.json` sits beside the splits and carries the label vocabulary, the task, and any caveat needed to interpret the numbers.
The label set is never inferred from whichever rows happen to be present, or a split missing a rare field would silently renumber every label.

## Validation is fatal on purpose

`Document.validate` runs on every document at build time, and any adapter that hands it a misaligned document stops the build.
This is not defensive politeness.
A document whose boxes and words are off by one still trains, still shows a falling loss, and produces a model that is simply wrong.
Failing at build time costs seconds; failing silently costs a GPU day and a meaningless number.

The same principle governs unknown label values: adapters raise rather than mapping them to the outside class.
Both times that rule fired it was right: on CORD's dotted `menu.sub.nm`, and on four RVL-CDIP classes a mirror spelled differently from the paper, which had silently discarded a quarter of that corpus.

## Adding a source

Write `sources/<name>.py` exposing `SPLITS`, `describe() -> DatasetMeta` and `load(split, cache_dir) -> Iterator[Document]`, then list it in `sources/__init__.py`.
`build.py` needs to know nothing else.
A source that also exposes `parts(split)` and `load_part(part)` builds its parts in parallel processes under `--workers`, which is how IDL unpacks one shard per process.

## What each source actually required

**FUNSD** (`nielsr/funsd-layoutlmv3`), 149/50 forms, 7 BIO labels.
The clean one.
Its boxes are already on the 0-1000 scale, which is easy to misread as pixels because FUNSD pages happen to be about 1000 pixels tall.
The adapter asserts the scale rather than assuming it.

**CORD** (`naver-clova-ix/cord-v2`), 800/100/100 receipts, 59 BIO labels.
Words are buried in a JSON string, as four-corner quads in pixels, and the field name lives on the line rather than the word.
The published category list is also wrong about the released data: it writes `menu.sub_nm` where the data says `menu.sub.nm`, and lists a `menu.sub.etc` that does not exist.
29 categories, enumerated from all three splits.

**SROIE** (`Ssunbell/SROIE_layoutlmv2_sequence`), 594/32/347 receipts, 9 BIO labels.
The original `darentang/sroie` is a loading script, which `datasets` v4 dropped, and its data comes off a Google Drive link.
This mirror works but stores labels as bare integers with no key, and the scheme is not the conventional one: **`0` is COMPANY and `4` is outside.** Receipt 1 labels "OLD TOWN KOPITIAM SDN BHD" as 0 and "TAX INVOICE", "GST ID" and the line items as 4.
Assuming `0` meant O would have trained the model to find companies everywhere.
The source also omits B and I, so BIO is reconstructed from contiguous runs, which merges two adjacent entities of the same field.
SROIE numbers here are therefore not strictly comparable to published ones.

**RealKIE** (`zenodo.org/records/13327077`, CC BY-NC 4.0), three sources sharing `realkie.py`: `realkie-fcc` with 221/75/74 invoices and 11 fields, `realkie-nda` with 248/93/98 agreements and 3 fields, `realkie-charities` with 322/108/108 reports and 28 fields.
Only the CSVs and `ocr/` are unzipped into `data/raw/realkie`, as the page images are most of the 7.1GB and nothing reads them.
Labels are character spans over the release's OCR text, so a span tags exactly the OCR words it overlaps and two adjacent entities of one field stay two.
Each page keeps its own 0-1000 grid.

**IDL** (the IMDB distillation of the Industry Documents Library), 54 shards of about 480k pages.
The first corpus read every ninth shard, and `idl-fresh` reads 21 of the other 48, so continued pretraining never repeats a page.
`idl-docs` reads 7 of the 27 shards neither touched, one row per document rather than per page, for the 2048-token phase.
Its pages keep their order and each keeps its own 0-1000 grid, as the release stores a document's pages together and in order.
Both drop pages whose Bates numbers match a benchmark page.

## OCR

There is no OCR in the pipeline, because no source needs it.
PaddleOCR is also not installable here: `paddlepaddle` publishes no cp314 wheels, and declaring it even as an optional extra makes the whole lockfile unresolvable.
If a future source ships images without text, the OCR pass belongs in a separate py3.13 virtualenv writing canonical parquet, not in this dependency tree.

## Running

```sh
uv run python scripts/build_datasets.py --group downstream
uv run python scripts/build_datasets.py --dataset idl-fresh --workers 3
```
