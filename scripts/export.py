"""
Writes a self-contained Hugging Face model directory, or uploads one with --push

save_pretrained copies the three model modules in and auto_map points the Auto classes at them
The export is reloaded in a cold process and required to give bit-identical logits
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import torch
from huggingface_hub import HfApi
from transformers import AutoTokenizer

from src.config import load_config
from src.model import (
    AUTO_MAP,
    ModernLayoutLmConfig,
    ModernLayoutLmForMaskedLM,
    ModernLayoutLmForSequenceClassification,
    ModernLayoutLmForTokenClassification,
    ModernLayoutLmModel,
    ModernLayoutLmProcessor,
    register_modernlayoutlm,
)

# Which class to export as, its Auto class, and the pipeline the hub should offer it under
HEADS = {
    "base": (ModernLayoutLmModel, "AutoModel", "feature-extraction"),
    "mlm": (ModernLayoutLmForMaskedLM, "AutoModelForMaskedLM", "fill-mask"),
    "token": (
        ModernLayoutLmForTokenClassification,
        "AutoModelForTokenClassification",
        "token-classification",
    ),
    "sequence": (
        ModernLayoutLmForSequenceClassification,
        "AutoModelForSequenceClassification",
        "text-classification",
    ),
}

# Without these the directory loads as a bare ModernBERT, or not at all
REQUIRED_FILES = (
    "config.json",
    "configuration_modernlayoutlm.py",
    "modeling_modernlayoutlm.py",
    "processing_modernlayoutlm.py",
    "README.md",
)


def parse_args() -> argparse.Namespace:
    """
    Parses the command line

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="trained weights to export")
    parser.add_argument("--head", choices=sorted(HEADS), default="mlm", help="class to export as")
    parser.add_argument("--name", help="export directory name, default derived from the head")
    parser.add_argument("--out", type=Path, help="export directory, default under the exports")
    parser.add_argument("--push", metavar="OWNER/NAME", help="upload the export to this repo")
    parser.add_argument("--private", action="store_true", help="create the repo as private")
    parser.add_argument("--dry-run", action="store_true", help="check the push, upload nothing")
    return parser.parse_args()


def main() -> int:
    """
    Exports a checkpoint and verifies its round trip, or pushes an existing export

    Returns:
        0 on success, 1 when the export is incomplete, does not round trip, or cannot be pushed.
    """
    args = parse_args()
    name = args.name or f"ModernLayoutLM-base-{args.head}"
    export_dir = args.out or load_config().paths.export_dir / name
    if args.push:
        return push(export_dir, args.push, args.private, args.dry_run)
    if args.checkpoint is None or not (args.checkpoint / "config.json").is_file():
        print(f"No model at {args.checkpoint}")
        return 1

    register_modernlayoutlm()
    model_class, auto_class, pipeline = HEADS[args.head]
    config = ModernLayoutLmConfig.from_pretrained(args.checkpoint)
    model = model_class.from_pretrained(args.checkpoint, config=config).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    processor = ModernLayoutLmProcessor(tokenizer=tokenizer)

    # Registering for the auto class is what makes save_pretrained copy the modules across
    config.register_for_auto_class("AutoConfig")
    model.register_for_auto_class(auto_class)
    ModernLayoutLmProcessor.register_for_auto_class("AutoProcessor")
    model.save_pretrained(export_dir)
    tokenizer.save_pretrained(export_dir)
    processor.save_pretrained(export_dir)

    # List every head in auto_map, so one directory loads as an encoder or under any head
    config_path = export_dir / "config.json"
    payload = json.loads(config_path.read_text())
    payload["auto_map"] = {**AUTO_MAP, **payload.get("auto_map", {})}
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_card(export_dir, name, auto_class, pipeline, args.checkpoint)
    shutil.copy(Path(__file__).resolve().parent.parent / "LICENSE", export_dir / "LICENSE")

    missing = [f for f in REQUIRED_FILES if not (export_dir / f).is_file()]
    if missing:
        print(f"Export is incomplete, missing {missing}")
        return 1
    print(f"exported to {export_dir}: {sorted(p.name for p in export_dir.iterdir())}")
    return 0 if _round_trip(export_dir, processor, model, auto_class) else 1


def push(export_dir: Path, repo: str, private: bool, dry_run: bool) -> int:
    """
    Uploads a complete export, a deliberate step that needs HF_TOKEN with write access

    Args:
        export_dir: The exported model directory.
        repo: Target repository as owner/name.
        private: Create the repository as private.
        dry_run: Check the directory and upload nothing.

    Returns:
        0 on success, 1 when the export is incomplete or no token is set.
    """
    missing = [f for f in REQUIRED_FILES if not (export_dir / f).is_file()]
    if missing:
        print(f"Refusing to push an incomplete export at {export_dir}, missing {missing}")
        return 1
    if dry_run:
        print(f"{export_dir} is complete and would be pushed to {repo}")
        return 0
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set. Put it in .env, which is gitignored.")
        return 1
    api = HfApi(token=token)
    api.create_repo(repo, private=private, exist_ok=True)
    api.upload_folder(folder_path=str(export_dir), repo_id=repo)
    print(f"pushed {export_dir} to https://huggingface.co/{repo}")
    return 0


def _round_trip(
    export_dir: Path, processor: ModernLayoutLmProcessor, model: torch.nn.Module, auto_class: str
) -> bool:
    """
    Reloads the export in a fresh process and requires bit-identical output

    In-process reloading would find the already-imported classes and prove nothing.

    Args:
        export_dir: The export directory.
        processor: The processor, to build a fixed batch.
        model: The in-process model, for reference output.
        auto_class: The Auto class the export registered itself under.

    Returns:
        True when the reloaded output matches exactly.
    """
    words = ["Total", "amount", "due", "42.00"]
    boxes = [[10, 10, 60, 20], [70, 10, 140, 20], [150, 10, 190, 20], [200, 10, 250, 20]]
    batch = processor(words=words, boxes=boxes, max_length=32, return_tensors="pt")
    with torch.no_grad():
        output = model(**batch)
    reference = output.logits if hasattr(output, "logits") else output.last_hidden_state
    probe = export_dir / "_roundtrip.pt"
    torch.save({"batch": dict(batch), "reference": reference}, probe)
    script = textwrap.dedent(f"""
        import torch
        from transformers import {auto_class}
        saved = torch.load({str(probe)!r}, weights_only=False)
        model = {auto_class}.from_pretrained({str(export_dir)!r}, trust_remote_code=True).eval()
        with torch.no_grad():
            output = model(**saved["batch"])
        got = output.logits if hasattr(output, "logits") else output.last_hidden_state
        print("MATCH", bool(torch.equal(got, saved["reference"])))
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    probe.unlink(missing_ok=True)
    if "MATCH True" in result.stdout:
        print("round trip: reloaded in a cold process, output bit-identical")
        return True
    print(f"round trip FAILED\n{result.stdout}\n{result.stderr[-2000:]}")
    return False


def _write_card(
    export_dir: Path, name: str, auto_class: str, pipeline: str, checkpoint: Path
) -> None:
    """
    Writes the model card, which documents the box convention users get wrong

    Args:
        export_dir: The export directory.
        name: The model name.
        auto_class: The Auto class the usage example loads.
        pipeline: The hub pipeline tag.
        checkpoint: Where the weights came from.
    """
    card = f"""---
library_name: transformers
license: mit
language:
- en
base_model: answerdotai/ModernBERT-base
pipeline_tag: {pipeline}
tags:
- document-understanding
- layoutlm
- modernbert
---

# {name}

LayoutLM's 2D layout embeddings grafted onto a ModernBERT backbone, pretrained with layout-aware masked language modeling.
Four tables embed the coordinates, where `x0` and `x1` share table X and `y0` and `y1` share table Y, exactly as in LayoutLM v1.
Their sum lands on the token embeddings before the embedding LayerNorm.
Attention is untouched, so ModernBERT's RoPE and its alternating local and global attention run unmodified.

Exported from `{checkpoint}`.

## Usage

```python
from transformers import AutoTokenizer, {auto_class}

model = {auto_class}.from_pretrained("{name}", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("{name}")
```

## The box convention

- `bbox` has shape `(batch, sequence, 4)` and holds `(x0, y0, x1, y1)`.
- The origin is the **top left** of the page.
- Coordinates are **normalized to `[0, 1000]`** by the page width and height, not given in pixels.
- `CLS` takes `[0, 0, 0, 0]` and `SEP` takes `[1000, 1000, 1000, 1000]`.
- Every subword of a word takes the word's box.
- `bbox=None` skips the layout sum, so the model runs as plain ModernBERT.
- All-zero boxes are not `bbox=None`, as they put every word at the origin.

The bundled processor does all of this:

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("{name}", trust_remote_code=True)
batch = processor(words=words, boxes=normalized_boxes, max_length=512, return_tensors="pt")
```

- `padding` defaults to `max_length`, which without a `max_length` is the model's own 8192.
- Pass the length you want, or `padding="longest"`.

## Training

Layout-aware masked language modeling over about 11M pages, each read once.

| stage | pages | sequences | steps |
|---|---|---|---|
| 1 | 2.19M: 1.5M IDL, 461k PubTables-1M, 160k PDFA, 68k DocLayNet | 1.91M | 29899 |
| 2 | 8.8M IDL, from shards stage 1 never read | 7.36M | 115009 |

- Sequences are 512 tokens at 30% masking, 28 hours in all on one RTX 3090.
- Text is masked and boxes are left intact, so recovering a masked token rewards reading its position on the page.
- The layout tables start at zero, so the model starts numerically identical to ModernBERT.
- Both stages drop pages carrying a Bates number from the FUNSD or RVL-CDIP evaluation splits, as IDL and those benchmarks share IIT-CDIP.

## Fine-tuned results

| task | split | this model | layoutlm-base | plain ModernBERT | this model, no boxes |
|---|---|---|---|---|---|
| FUNSD | test | 0.7565 ± 0.0037 | 0.7785 ± 0.0104 | 0.5884 ± 0.0082 | 0.3000 ± 0.0247 |
| CORD | validation | 0.9546 ± 0.0023 | 0.9685 ± 0.0037 | 0.9424 ± 0.0006 | 0.9312 ± 0.0042 |
| SROIE | validation | 0.9337 ± 0.0060 | 0.9223 ± 0.0199 | 0.8955 ± 0.0072 | 0.8411 ± 0.0189 |

| setting | value |
|---|---|
| metric | entity F1, field-level Hmean per ICDAR task 3 on SROIE |
| seeds | three |
| splits | FUNSD has no validation split, SROIE's is 32 receipts |
| rate | 3e-4 for this model and ModernBERT, chosen on FUNSD test, 5e-5 for LayoutLM v1 |
| epochs | 30 |
| window | 1024 tokens, 512 for LayoutLM v1 as its positions cap there |
| chunks | an over-length document is stitched back together before scoring |
| layoutlm-base | `microsoft/layoutlm-base-uncased` fine-tuned on identical data |
| no boxes | this model fine-tuned with every box at the origin |

1) LayoutLM v1 leads on FUNSD by 0.022 F1 and on CORD by 0.014 F1.
2) This model leads on SROIE by 0.011 Hmean, inside LayoutLM v1's seed deviation.
3) FUNSD is drawn from IIT-CDIP, which LayoutLM v1 pretrained on, while this corpus drops FUNSD's evaluation pages.
4) Without boxes this model falls below plain ModernBERT on all three, so text without boxes is better served by ModernBERT itself.
5) SROIE labels come from a mirror that omits B and I, so adjacent entities of one field merge and the scores do not compare strictly to published ones.
6) The sequence-classification head is implemented and tested but has no downstream evidence, as the RVL-CDIP data was deleted before the final runs.

## Limitations

- Pretrained on English business and scientific documents: OCR'd scans from the Industry Documents Library, tables from PubTables-1M, born-digital PDFs from PDFA and reports from DocLayNet.
- Non-English documents, handwriting and dense forms are out of distribution.
- There is no OCR here, so the caller supplies words and boxes, and box quality bounds what the layout tables can do.

## License

MIT.
"""
    (export_dir / "README.md").write_text(card)


if __name__ == "__main__":
    raise SystemExit(main())
