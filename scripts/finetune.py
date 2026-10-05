"""
Fine-tunes the downstream tasks: any cells of the variant matrix, or quick probes of checkpoints

Runs are named task-variant-sSEED, plus an optional tag, and skipped once their result.json exists
--probe fine-tunes each checkpoint briefly and appends its scores to a series keyed by step
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
import traceback
from dataclasses import replace
from pathlib import Path

from src.config import load_config, parse_overrides
from src.train import RunSpec, finetune

# Every reported baseline cell came from revision 30e3cdd39c11f09757b0fcf7598533d05052acef
LAYOUTLM_BASELINE = "microsoft/layoutlm-base-uncased"

DATASETS = ["funsd", "cord", "sroie", "realkie-fcc", "realkie-nda", "realkie-charities"]
VARIANTS = ["layout", "nobbox", "skipmlm", "plain", "layoutlm"]

# The fp32 pretrain, as the bf16-weight one transfers ~0.11 F1 worse on FUNSD
PRETRAIN_CHECKPOINT = "fp32"

# Probes track a trend between checkpoints, so they train briefly on tasks outside IIT-CDIP
PROBE_TASKS = ["cord", "sroie"]
PROBE_EPOCHS = 12


def parse_args() -> argparse.Namespace:
    """
    Parses the command line

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", choices=DATASETS, help="default all, or probe set")
    parser.add_argument("--variant", nargs="+", default=VARIANTS, choices=VARIANTS)
    parser.add_argument("--seed", type=int, nargs="+", default=[42], help="one run per seed")
    parser.add_argument("--pretrained", help="the layout-MLM checkpoint, default the fp32 one")
    parser.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE", help="config override")
    parser.add_argument("--tag", help="suffix for run names, to keep an override apart")
    parser.add_argument("--probe", type=Path, nargs="+", help="checkpoints to probe instead")
    parser.add_argument("--series", type=Path, help="probe series file, default in the reports")
    parser.add_argument("--num-proc", type=int, default=8, help="tokenization workers")
    parser.add_argument("--force", action="store_true", help="rerun even if a result exists")
    parser.add_argument("--dry-run", action="store_true", help="list the runs, execute nothing")
    return parser.parse_args()


def main() -> int:
    """
    Plans the runs, executes those not yet done, and reports each as it finishes

    A failed run is recorded and the rest continue, as one bad cell should not cost the others.

    Returns:
        0 when every run succeeded, 1 when any failed.
    """
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config()
    overrides = parse_overrides(args.set)
    suffix = f"-{args.tag}" if args.tag else ""

    if args.probe:
        overrides.setdefault("epochs", PROBE_EPOCHS)
        output_root = config.paths.checkpoint_dir / "probes"
        planned = [
            RunSpec(
                task, str(path), f"probe-{task}-{_label(path, config.paths.checkpoint_dir)}{suffix}"
            )
            for path in args.probe
            for task in args.dataset or PROBE_TASKS
        ]
    else:
        output_root = config.paths.checkpoint_dir / "finetune"
        pretrained = args.pretrained or str(config.paths.checkpoint_dir / PRETRAIN_CHECKPOINT)
        raw = str(config.paths.backbone_dir / "ModernBERT-base")
        backbones = {
            "layout": (pretrained, True),
            "nobbox": (pretrained, False),
            "skipmlm": (raw, True),
            "plain": (raw, False),
            "layoutlm": (LAYOUTLM_BASELINE, True),
        }
        planned = [
            RunSpec(
                task,
                backbones[variant][0],
                f"{task}-{variant}-s{seed}{suffix}",
                backbones[variant][1],
            )
            for seed in args.seed
            for task in args.dataset or DATASETS
            for variant in args.variant
        ]

    done = {
        spec.run_name for spec in planned if (output_root / spec.run_name / "result.json").is_file()
    }
    pending = [spec for spec in planned if args.force or spec.run_name not in done]
    print(f"{len(planned)} runs planned, {len(planned) - len(pending)} already done")
    for spec in pending:
        print(f"  {spec.run_name:32s} backbone={spec.backbone} bbox={spec.use_bbox}")
    if args.dry_run:
        return 0

    failures, series = [], []
    for index, spec in enumerate(pending, start=1):
        print(f"\n{'=' * 70}\n[{index}/{len(pending)}] {spec.run_name}\n{'=' * 70}", flush=True)
        started = time.perf_counter()
        seed = int(match.group(1)) if (match := re.search(r"-s(\d+)", spec.run_name)) else 42
        try:
            result = finetune(
                spec=spec,
                config=replace(config.finetune_for(spec.task), **overrides),
                processed_dir=config.paths.processed_dir,
                output_root=output_root,
                seed=seed,
                num_proc=args.num_proc,
            )
        except OSError, ValueError, KeyError, RuntimeError:
            traceback.print_exc()
            failures.append(spec.run_name)
            continue
        print(f"{result.summary()}   [{(time.perf_counter() - started) / 60:.1f} min]", flush=True)
        headline = result.metrics.get("f1", result.metrics.get("accuracy", 0.0))
        series.append({"checkpoint": spec.backbone, "task": spec.task, "score": headline})

    if args.probe:
        epochs = overrides["epochs"]
        _append_series(
            args.series or config.paths.report_dir / "checkpoint_series.json", series, epochs
        )
    if failures:
        print(f"\n{len(failures)} of {len(pending)} failed: {failures}")
    return 1 if failures else 0


def _append_series(path: Path, scores: list[dict], epochs: object) -> None:
    """
    Appends one entry per probed checkpoint to the series file and prints the series

    Args:
        path: The series file, created when absent.
        scores: One score per probed checkpoint and task.
        epochs: The fine-tuning epochs every probe ran.
    """
    series = json.loads(path.read_text()) if path.is_file() else []
    for checkpoint in dict.fromkeys(score["checkpoint"] for score in scores):
        tasks = {s["task"]: s["score"] for s in scores if s["checkpoint"] == checkpoint}
        series.append(
            {
                "step": _step(Path(checkpoint)),
                "checkpoint": checkpoint,
                "epochs": epochs,
                "scored_at": time.strftime("%Y-%m-%d %H:%M"),
                "tasks": tasks,
                "mean": sum(tasks.values()) / len(tasks),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(series, indent=2) + "\n")
    for row in series:
        cells = "  ".join(f"{task}={score:.4f}" for task, score in row["tasks"].items())
        print(f"{row['step']:>8}  mean={row['mean']:.4f}  {cells}")
    print(f"series -> {path}")


def _label(path: Path, root: Path) -> str:
    """
    Names a checkpoint by its path under the checkpoint root, so every probe run name is unique

    Args:
        path: The checkpoint directory.
        root: The configured checkpoint root.

    Returns:
        The relative path joined by dashes, such as `fp32-kept-checkpoint-26000`.
    """
    resolved = path.resolve()
    relative = (
        resolved.relative_to(root.resolve())
        if resolved.is_relative_to(root.resolve())
        else Path(resolved.name)
    )
    return "-".join(relative.parts)


def _step(path: Path) -> int:
    """
    Reads the training step off a checkpoint directory name

    Args:
        path: A path like `.../checkpoint-40000`.

    Returns:
        The step, or 0 when the name carries none.
    """
    match = re.search(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else 0


if __name__ == "__main__":
    raise SystemExit(main())
