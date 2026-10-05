"""
Downstream token-classification fine-tuning, one function for every task and variant via a RunSpec

That makes every control the same code path rather than an approximation of it
The LayoutLM v1 baseline goes through it too, on the same data and the same processor
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    LayoutLMForTokenClassification,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import EvalPrediction

from src.config import FinetuneConfig
from src.data import DatasetMeta
from src.eval import (
    FIELD_LEVEL_SOURCES,
    decode_token_predictions,
    score_fields,
    stitch_chunks,
    token_metrics,
)
from src.model import (
    ModernLayoutLmConfig,
    ModernLayoutLmForTokenClassification,
    ModernLayoutLmProcessor,
)
from src.train.batches import SAMPLE_INDEX, load_split, prepare_task_splits
from src.train.pretrain import adamw, param_groups

logger = logging.getLogger(__name__)

# Learned absolute positions cap LayoutLM v1 at 512 tokens
LAYOUTLM_MAX_LENGTH = 512


@dataclass
class RunSpec:
    """
    One fine-tuning run, including the ablation variants

    Args:
        task: The dataset name, such as "funsd".
        backbone: Path or hub id of the weights to start from.
        run_name: Directory-safe name for outputs.
        use_bbox: When False every box is zeroed, the no-bbox control.
    """

    task: str
    backbone: str
    run_name: str
    use_bbox: bool = True


@dataclass
class RunResult:
    """
    What a fine-tuning run produced, written beside it as `result.json`

    Args:
        spec: The run that was executed.
        metrics: Final metrics on the evaluation split.
        eval_split: Which split the metrics came from.
        output_dir: Where the weights and metrics were written.
        train_examples: Chunks in the training split after tokenization.
        history: Per-evaluation metric snapshots.
    """

    spec: RunSpec
    metrics: dict[str, float]
    eval_split: str
    output_dir: str
    train_examples: int
    history: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        """
        Renders the run's metrics on one line

        Returns:
            A printable string.
        """
        scores = "  ".join(f"{key}={value:.4f}" for key, value in sorted(self.metrics.items()))
        return f"{self.spec.run_name}: {scores}"


def is_layoutlm(backbone: str) -> bool:
    """
    Reports whether a backbone is the original LayoutLM rather than our graft

    Args:
        backbone: Path or hub id of the weights.

    Returns:
        True for a `layoutlm` checkpoint.
    """
    return "layoutlm-" in backbone.lower() or backbone.lower().endswith("layoutlm")


def finetune(
    spec: RunSpec,
    config: FinetuneConfig,
    processed_dir: Path,
    output_root: Path,
    seed: int = 42,
    num_proc: int | None = None,
) -> RunResult:
    """
    Fine-tunes one model on one downstream dataset and scores it on validation, else test

    Chunks of a long page are stitched back together before scoring, and SROIE adds field metrics.

    Args:
        spec: What to run.
        config: Hyperparameters for this task.
        processed_dir: Root of the processed dataset tree.
        output_root: Directory the run's own directory goes under.
        seed: Random seed.
        num_proc: Worker processes for tokenization.

    Returns:
        The run's result.

    Raises:
        ValueError: If the dataset has no word labels.
    """
    if is_layoutlm(spec.backbone):
        config = replace(config, learning_rate=config.baseline_learning_rate)
    if is_layoutlm(spec.backbone) and config.max_length > LAYOUTLM_MAX_LENGTH:
        logger.warning("clamping max_length to %d for LayoutLM", LAYOUTLM_MAX_LENGTH)
        config = replace(config, max_length=LAYOUTLM_MAX_LENGTH)
    tokenizer = AutoTokenizer.from_pretrained(spec.backbone)
    processor = ModernLayoutLmProcessor(
        tokenizer=tokenizer, label_all_subwords=config.label_all_subwords
    )
    splits, meta = prepare_task_splits(
        processed_dir, spec.task, processor, config.max_length, num_proc=num_proc
    )
    if meta.task != "token":
        raise ValueError(f"{spec.task} has task {meta.task!r}, not token classification")
    if not spec.use_bbox:
        splits = {name: _zero_boxes(dataset) for name, dataset in splits.items()}

    eval_split = "validation" if "validation" in splits else "test"
    scored = splits[eval_split]
    pages = list(scored[SAMPLE_INDEX])
    # A stray column would become a forward argument
    scored = scored.remove_columns([SAMPLE_INDEX])
    words = None
    if meta.source in FIELD_LEVEL_SOURCES:
        words = [list(page) for page in load_split(processed_dir, spec.task, eval_split)["words"]]

    model = _build_model(spec, meta)
    output_dir = output_root / spec.run_name
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size * 2,
            gradient_accumulation_steps=config.grad_accum,
            num_train_epochs=config.epochs,
            warmup_ratio=config.warmup_ratio,
            bf16=True,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="no",
            seed=seed,
            report_to=[],
            # Pruning must spare bbox, which every head names in its forward signature
            remove_unused_columns=True,
        ),
        train_dataset=splits["train"],
        eval_dataset=scored,
        compute_metrics=_metric_fn(meta, pages, words),
        # Foreach rather than fused, which the reported results were trained with
        optimizers=(adamw(_param_groups(model, spec, config), fused=False), None),
    )
    trainer.train()
    metrics = trainer.evaluate(cast(Any, scored))
    # Ship the processor too or the box convention is a guess
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    result = RunResult(
        spec=spec,
        metrics={k.removeprefix("eval_"): v for k, v in metrics.items() if isinstance(v, float)},
        eval_split=eval_split,
        output_dir=str(output_dir),
        train_examples=len(splits["train"]),
        history=[entry for entry in trainer.state.log_history if "eval_loss" in entry],
    )
    (output_dir / "result.json").write_text(json.dumps(asdict(result), indent=2) + "\n")
    return result


def _build_model(spec: RunSpec, meta: DatasetMeta) -> Any:
    """
    Builds a token-classification head, as LayoutLM's own class for the baseline

    Args:
        spec: The run being set up.
        meta: The dataset metadata, supplying the label vocabulary.

    Returns:
        A model with a fresh classification head.
    """
    labels = {
        "num_labels": len(meta.label_names),
        "id2label": dict(enumerate(meta.label_names)),
        "label2id": {name: index for index, name in enumerate(meta.label_names)},
    }
    if is_layoutlm(spec.backbone):
        return LayoutLMForTokenClassification.from_pretrained(spec.backbone, **labels)
    config = ModernLayoutLmConfig.from_pretrained(spec.backbone, **labels, reference_compile=False)
    return ModernLayoutLmForTokenClassification.from_pretrained(spec.backbone, config=config)


def _param_groups(model: Any, spec: RunSpec, config: FinetuneConfig) -> list[dict[str, Any]]:
    """
    Builds parameter groups, with LayoutLM's pretrained 2D tables on the backbone rate

    Args:
        model: The model to optimize.
        spec: The run, which says whether this is the LayoutLM baseline.
        config: Hyperparameters for this task.

    Returns:
        Parameter groups ready for `adamw`.
    """
    if not is_layoutlm(spec.backbone):
        return param_groups(
            model, config.learning_rate, config.layout_learning_rate, config.weight_decay
        )
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    return [
        {
            "params": [p for n, p in named if "bias" not in n],
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
            "name": "backbone",
        },
        {
            "params": [p for n, p in named if "bias" in n],
            "lr": config.learning_rate,
            "weight_decay": 0.0,
            "name": "no_decay",
        },
    ]


def _metric_fn(meta: DatasetMeta, pages: list[int], words: list[list[str]] | None) -> Any:
    """
    Builds the compute_metrics callback, entity F1 plus field Hmean for a field-level dataset

    Args:
        meta: The dataset metadata, supplying the label names.
        pages: The source page of each evaluation row, in the order Trainer evaluates them.
        words: The evaluation split's words per page, for a field-level dataset only.

    Returns:
        A callable for `Trainer(compute_metrics=...)`.
    """

    def compute(prediction: EvalPrediction) -> dict[str, float]:
        """Scores one evaluation pass."""
        logits = prediction.predictions
        logits = np.asarray(logits[0] if isinstance(logits, tuple) else logits)
        labels = np.asarray(prediction.label_ids)
        scores = token_metrics(logits, labels, meta.label_names, pages)
        if words is None:
            return scores
        predicted, truth = decode_token_predictions(logits, labels, meta.label_names)
        return scores | score_fields(words, *stitch_chunks(predicted, truth, pages))

    return compute


def _zero_boxes(dataset: Dataset) -> Dataset:
    """
    Moves every box to the origin, for the no-bbox control

    Args:
        dataset: A tokenized split.

    Returns:
        The same split with all coordinates zeroed.
    """

    def zero(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """Zeroes one batch of boxes."""
        batch["bbox"] = [[[0, 0, 0, 0] for _ in row] for row in batch["bbox"]]
        return batch

    return dataset.map(zero, batched=True, desc="zeroing boxes")
