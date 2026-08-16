"""
Scores a mid-run pretraining checkpoint so a stalled run is caught within a day

Only CORD and SROIE are scored as they are independent of the IDL pretraining corpus
Results are keyed by checkpoint step so the series plots against pretraining loss
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import replace
from pathlib import Path

from src.config import load_config
from src.train import RunSpec, finetune

# Independent of IIT-CDIP, so a score here cannot be inflated by pretraining contamination
CLEAN_TASKS = ["cord", "sroie"]

# Use fewer epochs as this only tracks the trend between checkpoints
PROBE_EPOCHS = 12


def parse_args() -> argparse.Namespace:
    """
    Parses the command line

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="pretraining checkpoint")
    parser.add_argument("--dataset", nargs="+", default=CLEAN_TASKS, help="tasks to score")
    parser.add_argument("--epochs", type=int, default=PROBE_EPOCHS, help="fine-tuning epochs")
    parser.add_argument("--num-proc", type=int, default=8, help="tokenization workers")
    parser.add_argument("--out", type=Path, help="where to append the results series")
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    """
    Extracts the training step from a checkpoint directory name

    Args:
        path: A path like `.../checkpoint-40000`.

    Returns:
        The step number, or 0 when the name carries none.
    """
    match = re.search(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else 0


def main() -> int:
    """
    Fine-tunes the checkpoint briefly on each clean task and records the scores

    Returns:
        0 when every task scored, 1 when any failed.
    """
    args = parse_args()
    config = load_config()

    if not (args.checkpoint / "config.json").is_file():
        print(f"No checkpoint at {args.checkpoint}")
        return 1

    step = checkpoint_step(args.checkpoint)
    out = args.out or (config.paths.report_dir / "checkpoint_series.json")
    series = json.loads(out.read_text()) if out.is_file() else []

    entry: dict[str, object] = {
        "step": step,
        "checkpoint": str(args.checkpoint),
        "epochs": args.epochs,
        "scored_at": time.strftime("%Y-%m-%d %H:%M"),
        "tasks": {},
    }
    tasks: dict[str, float] = {}
    failures: list[str] = []

    for task in args.dataset:
        spec = RunSpec(
            task=task,
            backbone=str(args.checkpoint),
            run_name=f"probe-{task}-{step}",
        )
        finetune_config = replace(config.finetune_for(task), epochs=args.epochs)
        print(f"\n=== {task} @ step {step} ===", flush=True)
        try:
            result = finetune(
                spec=spec,
                config=finetune_config,
                processed_dir=config.paths.processed_dir,
                output_root=config.paths.checkpoint_dir / "probes",
                num_proc=args.num_proc,
            )
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            print(f"FAILED {task}: {type(exc).__name__}: {exc}")
            failures.append(task)
            continue
        headline = result.metrics.get("f1", result.metrics.get("accuracy", 0.0))
        tasks[task] = headline
        print(f"{task}: {headline:.4f}", flush=True)

    entry["tasks"] = tasks
    if failures:
        entry["failed"] = failures
    if tasks:
        entry["mean"] = sum(tasks.values()) / len(tasks)
    # Skip empty probes as all-zero entries plot as a collapse
    if tasks:
        series.append(entry)
    else:
        print("no task scored; not appending to the series")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(series, indent=2) + "\n")

    print(f"\n{'step':>8}  {'mean':>7}  " + "  ".join(f"{t:>7}" for t in args.dataset))
    for row in series:
        scores = row.get("tasks", {})
        mean = row.get("mean", 0.0)
        cells = "  ".join(
            f"{scores[t]:>7.4f}" if t in scores else f"{'-':>7}" for t in args.dataset
        )
        print(f"{row['step']:>8}  {mean:>7.4f}  {cells}")
    print(f"\nseries -> {out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
