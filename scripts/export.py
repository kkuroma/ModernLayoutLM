"""
Writes a self-contained Hugging Face model directory, or uploads one with --push

save_pretrained copies the three model modules in and auto_map points the Auto classes at them
The export is reloaded in a cold process and required to give bit-identical logits
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import torch
from huggingface_hub import HfApi, get_token
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

ROOT = Path(__file__).resolve().parent.parent

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
    _write_card(export_dir, pipeline)
    for name in ("LICENSE", "performance.png"):
        shutil.copy(ROOT / name, export_dir / name)

    missing = [f for f in REQUIRED_FILES if not (export_dir / f).is_file()]
    if missing:
        print(f"Export is incomplete, missing {missing}")
        return 1
    print(f"exported to {export_dir}: {sorted(p.name for p in export_dir.iterdir())}")
    return 0 if _round_trip(export_dir, processor, model, auto_class) else 1


def push(export_dir: Path, repo: str, private: bool, dry_run: bool) -> int:
    """
    Uploads a complete export, a deliberate step that needs a token with write access

    Args:
        export_dir: The exported model directory.
        repo: Target repository as owner/name.
        private: Create the repository as private.
        dry_run: Check the directory and upload nothing.

    Returns:
        0 on success, 1 when the export is incomplete or no token is found.
    """
    missing = [f for f in REQUIRED_FILES if not (export_dir / f).is_file()]
    if missing:
        print(f"Refusing to push an incomplete export at {export_dir}, missing {missing}")
        return 1
    if dry_run:
        print(f"{export_dir} is complete and would be pushed to {repo}")
        return 0
    if get_token() is None:
        print("No Hugging Face token, run `hf auth login` with write access")
        return 1
    api = HfApi()
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


def _write_card(export_dir: Path, pipeline: str) -> None:
    """
    Writes the model card from the repository README with the hub metadata on top

    Args:
        export_dir: The export directory.
        pipeline: The hub pipeline tag.
    """
    readme = (ROOT / "README.md").read_text()
    # The setup and run steps only make sense inside the repository
    body = readme.partition("\n## Reproducing")[0].rstrip() + "\n"
    metadata = f"""---
library_name: transformers
license: apache-2.0
language:
- en
base_model: answerdotai/ModernBERT-base
pipeline_tag: {pipeline}
tags:
- document-understanding
- layoutlm
- modernbert
---

"""
    (export_dir / "README.md").write_text(metadata + body)


if __name__ == "__main__":
    raise SystemExit(main())
