"""
Collects finished runs into one results table

Reads the result.json each fine-tuning run writes and renders it as markdown and JSON per task
The document is tables only, so a reader draws the comparison rather than being told it
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.config import load_config

# Column order in the rendered table, when the metric is present
COLUMNS = [
    "f1",
    "precision",
    "recall",
    "hmean",
    "field_precision",
    "field_recall",
    "accuracy",
    "macro_f1",
    "token_accuracy",
    "loss",
]

# Row order, ours first, then the baseline, then the two controls
VARIANT_ORDER = ["layout", "layoutlm", "skipmlm", "nobbox", "plain"]

WHAT_EACH_VARIANT_IS = {
    "layout": "ModernLayoutLM: layout-MLM then fine-tune",
    "layoutlm": "microsoft/layoutlm-base-uncased, identical data",
    "skipmlm": "ours, fine-tuned straight off ModernBERT",
    "nobbox": "ours, boxes withheld at fine-tune time",
    "plain": "plain ModernBERT: no layout-MLM, no boxes",
}


def parse_args() -> argparse.Namespace:
    """
    Parses the command line

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, help="directory holding run subdirectories")
    parser.add_argument("--out", type=Path, help="directory to write the table into")
    return parser.parse_args()


def main() -> int:
    """
    Renders the results table from every run found on disk

    Returns:
        0 on success, 1 when no runs were found.
    """
    args = parse_args()
    config = load_config()
    runs_dir = args.runs or config.paths.checkpoint_dir / "finetune"
    out_dir = args.out or config.paths.report_dir

    results = _collect(runs_dir)
    if not results:
        print(f"No result.json found under {runs_dir}. Run a fine-tuning job first.")
        return 1

    table = _render(results)
    print(table)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.md").write_text(table + "\n")
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwritten to {out_dir / 'results.md'} and {out_dir / 'results.json'}")
    return 0


def _collect(runs_dir: Path) -> list[dict[str, Any]]:
    """
    Reads every run result under a directory

    Args:
        runs_dir: Directory holding one subdirectory per run.

    Returns:
        Run records, each holding the task, variant, metrics and training size.
    """
    newest: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
    for path in sorted(runs_dir.glob("*/result.json")):
        payload = json.loads(path.read_text())
        spec = payload.get("spec", {})
        record = {
            "task": spec.get("task", "unknown"),
            "variant": _variant(spec),
            "run_name": spec.get("run_name", path.parent.name),
            "backbone": spec.get("backbone", ""),
            "eval_split": payload.get("eval_split", ""),
            "train_examples": payload.get("train_examples", 0),
            "metrics": payload.get("metrics", {}),
        }
        key = (record["task"], record["variant"])
        stamp = path.stat().st_mtime
        if key not in newest or stamp > newest[key][0]:
            newest[key] = (stamp, record)
    return [record for _, record in newest.values()]


def _variant(spec: dict[str, Any]) -> str:
    """
    Names the ablation variant a run represents

    The LayoutLM check matches the checkpoint name, not the whole path. This project's directory
    is called ModernLayoutLM, so a substring test against the full path calls every run a
    LayoutLM baseline.

    Args:
        spec: The run's RunSpec as saved.

    Returns:
        One of "layout", "nobbox", "skipmlm", "plain" or "layoutlm".
    """
    backbone = str(spec.get("backbone", ""))
    name = Path(backbone).name.lower()
    if name.startswith("layoutlm"):
        return "layoutlm"

    # Key on the backbone, not the run name, as "wide" read as skipmlm
    pretrained = not name.startswith("modernbert")
    if spec.get("use_bbox", True):
        return "layout" if pretrained else "skipmlm"
    return "nobbox" if pretrained else "plain"


def _render(results: list[dict[str, Any]]) -> str:
    """
    Renders the results as markdown, one table per task

    Args:
        results: Run records from `_collect`.

    Returns:
        A markdown document.
    """
    lines = ["# Downstream results", "", *_variant_table(), ""]
    tasks = sorted({record["task"] for record in results})

    for task in tasks:
        rows = [record for record in results if record["task"] == task]
        present = [
            column for column in COLUMNS if any(column in record["metrics"] for record in rows)
        ]
        lines.append(f"## {task}, by variant")
        lines.append("")
        lines.append("| variant | " + " | ".join(present) + " | train chunks |")
        lines.append("|---" * (len(present) + 2) + "|")
        ordered = sorted(
            rows,
            key=lambda r: (
                VARIANT_ORDER.index(r["variant"]) if r["variant"] in VARIANT_ORDER else 99
            ),
        )
        for record in ordered:
            cells = [
                f"{record['metrics'][column]:.4f}" if column in record["metrics"] else "-"
                for column in present
            ]
            train = record["train_examples"]
            lines.append(f"| {record['variant']} | " + " | ".join(cells) + f" | {train} |")
        lines.append("")

    return "\n".join(lines)


def _variant_table() -> list[str]:
    """
    Renders the legend naming what each variant is

    Returns:
        Markdown lines for the variant table.
    """
    lines = ["## Variants", "", "| variant | backbone and boxes |", "|---|---|"]
    for name in VARIANT_ORDER:
        if name in WHAT_EACH_VARIANT_IS:
            lines.append(f"| {name} | {WHAT_EACH_VARIANT_IS[name]} |")
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
