"""
Uploads an exported model directory to the Hugging Face Hub

Publishing is a decision, so this stays a deliberate step outside the pipeline
Requires HF_TOKEN with write access to the target repository
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

# Without these the repository loads as a bare ModernBERT, or not at all
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
    parser.add_argument("--export", type=Path, required=True, help="an exported model directory")
    parser.add_argument("--repo", required=True, help="target repo id, as owner/name")
    parser.add_argument("--private", action="store_true", help="create the repo as private")
    parser.add_argument(
        "--dry-run", action="store_true", help="check the directory, upload nothing"
    )
    return parser.parse_args()


def main() -> int:
    """
    Validates an export and uploads it

    Returns:
        0 on success, 1 when the export is incomplete or no token is available.
    """
    args = parse_args()

    if not args.export.is_dir():
        print(f"No export at {args.export}")
        return 1
    missing = [name for name in REQUIRED_FILES if not (args.export / name).is_file()]
    if missing:
        print(f"Refusing to push an incomplete export, missing {missing}")
        return 1

    if args.dry_run:
        print(f"{args.export} is complete and would be pushed to {args.repo}")
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set. Put it in .env, which is gitignored.")
        return 1

    api = HfApi(token=token)
    api.create_repo(args.repo, private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(args.export), repo_id=args.repo)
    print(f"pushed {args.export} to https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
