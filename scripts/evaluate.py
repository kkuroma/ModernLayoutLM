"""
Collects finished runs into one results table

Reads the result.json each run writes, pooling a cell's seed repeats into a mean and a deviation
The document is tables only, so a reader draws the comparison rather than being told it
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
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
    "token_accuracy",
    "loss",
]

# Row order, ours first, then the baseline, then the two controls
VARIANT_ORDER = ["layout", "layoutlm", "skipmlm", "nobbox", "plain"]

# Runs the table is built from, the trailing seed being what marks one
SEEDED = re.compile(r"-s(\d+)$")

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
    Reads every seeded run under a directory, pooling the repeats of each cell

    Only directories named task-variant-sSEED are read.
    Exploratory runs sit beside them under other names, and averaging them in would mix setups.

    Args:
        runs_dir: Directory holding one subdirectory per run.

    Returns:
        One record per (task, variant), each metric carrying its mean, deviation and seed count.
    """
    pooled: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for path in sorted(runs_dir.glob("*/result.json")):
        match = SEEDED.search(path.parent.name)
        if match is None:
            continue
        payload = json.loads(path.read_text())
        payload["seed"] = int(match.group(1))
        spec = payload.get("spec", {})
        pooled.setdefault((spec.get("task", "unknown"), _variant(spec)), []).append(payload)

    records = []
    for (task, variant), payloads in pooled.items():
        payloads.sort(key=lambda payload: payload["seed"])
        first = payloads[0]
        records.append(
            {
                "task": task,
                "variant": variant,
                "seeds": [payload["seed"] for payload in payloads],
                "backbone": first.get("spec", {}).get("backbone", ""),
                "eval_split": first.get("eval_split", ""),
                "train_examples": first.get("train_examples", 0),
                "metrics": _pool([payload.get("metrics", {}) for payload in payloads]),
            }
        )
    return records


def _pool(metrics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Averages one cell's seed repeats metric by metric

    Args:
        metrics: One metric mapping per seed.

    Returns:
        Metric name to mean, standard deviation and the values pooled, deviation zero at one seed.
    """
    names = [name for name in metrics[0] if all(name in entry for entry in metrics)]
    pooled = {}
    for name in names:
        values = [float(entry[name]) for entry in metrics]
        pooled[name] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    return pooled


def _variant(spec: dict[str, Any]) -> str:
    """
    Names the ablation variant a run represents

    The LayoutLM check matches the checkpoint name, not the whole path.
    This project's directory is ModernLayoutLM, so a full-path test calls every run LayoutLM.

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
        lines.append("| variant | " + " | ".join(present) + " | seeds | train chunks |")
        lines.append("|---" * (len(present) + 3) + "|")
        ordered = sorted(
            rows,
            key=lambda r: (
                VARIANT_ORDER.index(r["variant"]) if r["variant"] in VARIANT_ORDER else 99
            ),
        )
        for record in ordered:
            cells = [_cell(record["metrics"].get(column)) for column in present]
            seeds = len(record["seeds"])
            train = record["train_examples"]
            lines.append(
                f"| {record['variant']} | " + " | ".join(cells) + f" | {seeds} | {train} |"
            )
        lines.append("")

    return "\n".join(lines)


def _cell(pooled: dict[str, Any] | None) -> str:
    """
    Renders one pooled metric

    Args:
        pooled: The metric's mean, deviation and values, or None when the cell lacks it.

    Returns:
        The mean alone at one seed, mean and deviation beyond that.
    """
    if pooled is None:
        return "-"
    if len(pooled["values"]) == 1:
        return f"{pooled['mean']:.4f}"
    return f"{pooled['mean']:.4f} ± {pooled['std']:.4f}"


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
