"""
Writes a self-contained Hugging Face model directory

save_pretrained copies the three model modules in and auto_map points the Auto classes at them
The result is reloaded in a fresh subprocess and required to give bit-identical logits
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import torch
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

# Which class to export, and which Auto class the checkpoint should register itself under
HEADS = {
    "base": (ModernLayoutLmModel, "AutoModel"),
    "mlm": (ModernLayoutLmForMaskedLM, "AutoModelForMaskedLM"),
    "token": (ModernLayoutLmForTokenClassification, "AutoModelForTokenClassification"),
    "sequence": (ModernLayoutLmForSequenceClassification, "AutoModelForSequenceClassification"),
}

# Modules that must land in the export for trust_remote_code to work
REQUIRED_FILES = (
    "config.json",
    "configuration_modernlayoutlm.py",
    "modeling_modernlayoutlm.py",
    "processing_modernlayoutlm.py",
)


def parse_args() -> argparse.Namespace:
    """
    Parses the command line

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="trained weights to export")
    parser.add_argument("--name", help="export directory name, default derived from the head")
    parser.add_argument(
        "--head",
        choices=sorted(HEADS),
        default="mlm",
        help="which class to export the weights as",
    )
    parser.add_argument("--tokenizer", help="tokenizer to bundle, default the checkpoint's own")
    return parser.parse_args()


def main() -> int:
    """
    Exports a checkpoint and verifies the round trip

    Returns:
        0 when the reloaded model reproduces the original logits exactly, 1 otherwise.
    """
    args = parse_args()
    config = load_config()
    register_modernlayoutlm()

    if not (args.checkpoint / "config.json").is_file():
        print(f"No model at {args.checkpoint}")
        return 1

    model_class, auto_class = HEADS[args.head]
    name = args.name or f"ModernLayoutLM-base-{args.head}"
    export_dir = config.paths.export_dir / name

    model_config = ModernLayoutLmConfig.from_pretrained(args.checkpoint)
    model = model_class.from_pretrained(args.checkpoint, config=model_config)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.checkpoint)
    processor = ModernLayoutLmProcessor(tokenizer=tokenizer)

    # Registering for the auto class is what makes save_pretrained copy the modules across
    model_config.register_for_auto_class("AutoConfig")
    model.register_for_auto_class(auto_class)
    ModernLayoutLmProcessor.register_for_auto_class("AutoProcessor")

    model.save_pretrained(export_dir)
    tokenizer.save_pretrained(export_dir)
    processor.save_pretrained(export_dir)
    _widen_auto_map(export_dir)
    _write_model_card(export_dir, name, args.head, args.checkpoint)

    missing = [f for f in REQUIRED_FILES if not (export_dir / f).is_file()]
    if missing:
        print(f"Export is incomplete, missing {missing}")
        return 1

    print(f"exported to {export_dir}")
    print(f"files: {sorted(p.name for p in export_dir.iterdir())}")

    return 0 if _verify(export_dir, processor, model, auto_class) else 1


def _widen_auto_map(export_dir: Path) -> None:
    """
    Records every head in `auto_map`, not only the one that was exported

    `register_for_auto_class` writes a single entry, but the modeling module defines all of them.
    Listing all four lets a user load the same directory as a bare encoder or with another head.

    Args:
        export_dir: The export directory.
    """
    path = export_dir / "config.json"
    payload = json.loads(path.read_text())
    payload["auto_map"] = {**AUTO_MAP, **payload.get("auto_map", {})}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _verify(
    export_dir: Path,
    processor: ModernLayoutLmProcessor,
    model: torch.nn.Module,
    auto_class: str,
) -> bool:
    """
    Reloads the export in a fresh process and compares logits

    A subprocess is used deliberately. Reloading in this process would find the already-imported
    classes and prove nothing about whether the copied modules are self-contained.

    Args:
        export_dir: The export directory.
        processor: The processor, to build a fixed batch.
        model: The in-process model, for reference logits.
        auto_class: The Auto class the export registered itself under.

    Returns:
        True when the reloaded logits match exactly.
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
        model = {auto_class}.from_pretrained({str(export_dir)!r}, trust_remote_code=True)
        model.eval()
        with torch.no_grad():
            output = model(**saved["batch"])
        got = output.logits if hasattr(output, "logits") else output.last_hidden_state
        print("RELOADED", type(model).__name__)
        print("MATCH", bool(torch.equal(got, saved["reference"])))
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    probe.unlink(missing_ok=True)

    if "MATCH True" in result.stdout:
        print("\nround trip: reloaded in a cold process, logits bit-identical")
        return True
    print("\nround trip FAILED")
    print(result.stdout)
    print(result.stderr[-2000:])
    return False


def _write_model_card(export_dir: Path, name: str, head: str, checkpoint: Path) -> None:
    """
    Writes a README documenting the box convention, which is the part users get wrong

    Args:
        export_dir: The export directory.
        name: The model name.
        head: Which head was exported.
        checkpoint: Where the weights came from.
    """
    auto_class = HEADS[head][1]
    card = f"""---
library_name: transformers
tags:
- document-understanding
- layoutlm
- modernbert
---

# {name}

LayoutLM's 2D layout embeddings on a ModernBERT backbone. Four coordinate embedding tables are
summed into the token embeddings before the embedding LayerNorm, exactly as in LayoutLM v1, where
`x0` and `x1` share table X and `y0` and `y1` share table Y. Attention is untouched, so
ModernBERT's RoPE and its alternating local and global attention work unmodified.

Exported from `{checkpoint}`.

## Usage

```python
from transformers import AutoTokenizer, {auto_class}

model = {auto_class}.from_pretrained("{name}", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("{name}")
```

## The box convention

This is the part that is easy to get wrong.

- `bbox` has shape `(batch, sequence, 4)` and holds `(x0, y0, x1, y1)`.
- The origin is the **top left** of the page.
- Coordinates are **normalized to `[0, 1000]`**, not pixels. Divide by the page width and height
  and multiply by 1000.
- `CLS` conventionally takes `[0, 0, 0, 0]` and `SEP` takes `[1000, 1000, 1000, 1000]`.
- Every token needs a box. A word split into several subwords gives each of them the word's box.

The bundled processor does all of this:

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("{name}", trust_remote_code=True)
batch = processor(words=words, boxes=normalized_boxes, return_tensors="pt")
```

Passing `bbox=None` skips the layout sum entirely and the model behaves as plain ModernBERT. Note
that this is **not** the same as passing all-zero boxes, which mean every word sits at the origin.

## Training

Layout-aware masked language modeling on PubTables-1M and DocLayNet, roughly 528k pages. Text is
masked and boxes are left intact, so recovering a masked token rewards reading its position on the
page. The layout tables are zero-initialized, so the model begins numerically identical to
ModernBERT and grows layout sensitivity from there.
"""
    (export_dir / "README.md").write_text(card)


if __name__ == "__main__":
    raise SystemExit(main())
