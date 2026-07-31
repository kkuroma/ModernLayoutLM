"""
Collects finished runs into one results table

Reads the `result.json` each fine-tuning run writes and renders a markdown and JSON table grouped
by task. Four variants per task, because a single number proves nothing on its own:

- layout: ModernLayoutLM after layout-MLM, the real thing
- nobbox: identical recipe with the boxes withheld, isolating the 2D embeddings
- skipmlm: fine-tuned straight off ModernBERT, isolating the pretraining stage
- layoutlm: microsoft/layoutlm-base-uncased on the identical processed data

    uv run python scripts/eval/evaluate.py
    uv run python scripts/eval/evaluate.py --runs data/checkpoints/finetune --out notebooks/results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.config import load_config

# The metric each task is judged on, and whether more is better.
HEADLINE = {
    "token": "f1",
    "sequence": "accuracy",
}

# Column order in the rendered table, when the metric is present.
COLUMNS = ["f1", "precision", "recall", "accuracy", "macro_f1", "token_accuracy", "loss"]

# Row order, strongest claim first, so the table reads as an argument rather than a dump.
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

    # A run starting from the raw backbone never saw the layout-MLM stage.
    pretrained = "pretrain" in name
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
    lines = ["# Results", ""]
    tasks = sorted({record["task"] for record in results})

    for task in tasks:
        rows = [record for record in results if record["task"] == task]
        present = [
            column for column in COLUMNS if any(column in record["metrics"] for record in rows)
        ]
        lines.append(f"## {task}")
        lines.append("")
        lines.append("| variant | " + " | ".join(present) + " | what it is |")
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
            note = WHAT_EACH_VARIANT_IS.get(record["variant"], "")
            lines.append(f"| {record['variant']} | " + " | ".join(cells) + f" | {note} |")
        lines.append("")
        lines.extend(_verdict(task, rows))
        lines.append("")

    return "\n".join(lines)


def _verdict(task: str, rows: list[dict[str, Any]]) -> list[str]:
    """
    States plainly whether layout beat the controls on this task

    The point of the table is this comparison, so it is spelled out rather than left to the reader.
    A null or negative result is reported as such.

    Args:
        task: The task name.
        rows: Run records for this task.

    Returns:
        Markdown lines describing the outcome.
    """
    by_variant = {record["variant"]: record for record in rows}
    layout = by_variant.get("layout") or by_variant.get("skipmlm")
    if layout is None:
        return ["No layout run for this task yet."]

    metric = HEADLINE.get(_task_kind(layout), "f1")
    if metric not in layout["metrics"]:
        return [f"No {metric} recorded for the layout run."]
    score = layout["metrics"][metric]

    lines = []
    for name, label in (
        ("plain", "plain ModernBERT"),
        ("layoutlm", "layoutlm-base-uncased"),
        ("skipmlm", "the same model without the layout-MLM stage"),
    ):
        other = by_variant.get(name)
        if other is None or metric not in other["metrics"]:
            lines.append(f"- {label}: not run")
            continue
        delta = score - other["metrics"][metric]
        verdict = "beats" if delta > 0 else "does not beat"
        lines.append(f"- {verdict} {label} by {delta:+.4f} {metric}")
    return lines


def _task_kind(record: dict[str, Any]) -> str:
    """
    Guesses whether a run was a token or sequence task from its metrics

    Args:
        record: A run record.

    Returns:
        "token" or "sequence".
    """
    return "token" if "f1" in record["metrics"] else "sequence"


if __name__ == "__main__":
    raise SystemExit(main())
