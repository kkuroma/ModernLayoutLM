"""
Downstream supervised fine-tuning

One function drives every downstream task. The variant to run is expressed as a `RunSpec`, which is
what makes the ablations cheap: the no-bbox control is the same code path with `use_bbox=False`, and
the LayoutLM baseline is the same code path with a different backbone and tokenizer.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction

from src.config import FinetuneConfig
from src.data import DatasetMeta
from src.eval.metrics import sequence_metrics, token_metrics
from src.model import (
    ModernLayoutLmConfig,
    ModernLayoutLmForSequenceClassification,
    ModernLayoutLmForTokenClassification,
    ModernLayoutLmProcessor,
)
from src.train.encoding import prepare_task_splits
from src.train.optimizer import build_param_groups, describe_param_groups

logger = logging.getLogger(__name__)

# Which split to score when a dataset has no dedicated validation split.
FALLBACK_EVAL_SPLIT = "test"


@dataclass
class RunSpec:
    """
    One fine-tuning run, including the ablation variants

    Args:
        task: The dataset name, such as "funsd".
        backbone: Path or hub id of the weights to start from.
        run_name: Directory-safe name for outputs.
        use_bbox: When False, boxes are withheld from the model entirely. This is the no-bbox
            control, and it reduces ModernLayoutLM to plain ModernBERT.
        tokenizer_name: Tokenizer to use, defaulting to the backbone's own.
    """

    task: str
    backbone: str
    run_name: str
    use_bbox: bool = True
    tokenizer_name: str | None = None


@dataclass
class RunResult:
    """
    What a fine-tuning run produced

    Args:
        spec: The run that was executed.
        metrics: Final metrics on the evaluation split.
        eval_split: Which split the metrics came from.
        output_dir: Where the checkpoint and metrics were written.
        train_examples: Chunks in the training split after tokenization.
        history: Per-evaluation metric snapshots, for plotting.
    """

    spec: RunSpec
    metrics: dict[str, float]
    eval_split: str
    output_dir: str
    train_examples: int
    history: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        """
        Renders the headline metrics on one line

        Returns:
            A printable string.
        """
        scores = "  ".join(f"{key}={value:.4f}" for key, value in sorted(self.metrics.items()))
        return f"{self.spec.run_name}: {scores}"


def finetune(
    spec: RunSpec,
    config: FinetuneConfig,
    processed_dir: Path,
    output_root: Path,
    seed: int = 42,
    num_proc: int | None = None,
) -> RunResult:
    """
    Fine-tunes one model on one downstream dataset and scores it

    Args:
        spec: What to run.
        config: Hyperparameters for this task.
        processed_dir: Root of the processed dataset tree.
        output_root: Directory to write the run's checkpoint and metrics under.
        seed: Random seed.
        num_proc: Worker processes for tokenization.

    Returns:
        The run's result.

    Raises:
        ValueError: If the dataset's task is not supported for fine-tuning.
    """
    config = _clamp_length(spec, config)
    tokenizer = AutoTokenizer.from_pretrained(spec.tokenizer_name or spec.backbone)
    processor = ModernLayoutLmProcessor(
        tokenizer=tokenizer,
        label_all_subwords=config.label_all_subwords,
    )
    splits, meta = prepare_task_splits(
        processed_dir=processed_dir,
        name=spec.task,
        processor=processor,
        max_length=config.max_length,
        num_proc=num_proc,
        max_train_pages=config.max_train_pages,
    )
    if meta.task not in {"token", "sequence"}:
        raise ValueError(f"{spec.task} has task {meta.task!r}, which is not fine-tunable")

    if not spec.use_bbox:
        splits = {name: _zero_boxes(dataset) for name, dataset in splits.items()}

    model = _build_model(spec, meta)
    groups = _param_groups(model, config)
    logger.info("parameter groups:\n%s", describe_param_groups(groups))

    eval_split = "validation" if "validation" in splits else FALLBACK_EVAL_SPLIT
    output_dir = output_root / spec.run_name

    trainer = Trainer(
        model=model,
        args=_training_args(config, output_dir, seed),
        train_dataset=splits["train"],
        eval_dataset=splits[eval_split],
        compute_metrics=_metric_fn(meta),
        optimizers=(_adamw(groups), None),
    )
    trainer.train()
    metrics = trainer.evaluate(cast(Any, splits[eval_split]))
    # Intermediate checkpoints are off, so the final weights are saved explicitly. The processor
    # goes with them, because a model whose box convention has to be guessed is not reusable.
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    result = RunResult(
        spec=spec,
        metrics={_strip_prefix(k): v for k, v in metrics.items() if isinstance(v, float)},
        eval_split=eval_split,
        output_dir=str(output_dir),
        train_examples=len(splits["train"]),
        history=[entry for entry in trainer.state.log_history if "eval_loss" in entry],
    )
    _write_result(result, output_dir)
    return result


# The original LayoutLM has learned absolute position embeddings capped at 512. ModernLayoutLM
# uses RoPE and has no such limit, which is one of the things the comparison is measuring.
LAYOUTLM_MAX_LENGTH = 512


def _clamp_length(spec: RunSpec, config: FinetuneConfig) -> FinetuneConfig:
    """
    Caps the sequence length for a baseline that cannot go as long as ours

    LayoutLM v1 has learned absolute position embeddings and hard-fails past 512 tokens. Rather
    than let the run crash deep inside the embedding lookup, the length is clamped here and the
    difference is logged, since it is part of what the comparison is measuring.

    Args:
        spec: The run being set up.
        config: The requested hyperparameters.

    Returns:
        The config, with max_length reduced if the backbone requires it.
    """
    if not is_layoutlm_baseline(spec.backbone) or config.max_length <= LAYOUTLM_MAX_LENGTH:
        return config
    logger.warning(
        "clamping max_length %d to %d for the LayoutLM baseline, which cannot go longer",
        config.max_length,
        LAYOUTLM_MAX_LENGTH,
    )
    return replace(config, max_length=LAYOUTLM_MAX_LENGTH)


def is_layoutlm_baseline(backbone: str) -> bool:
    """
    Reports whether a backbone is the original LayoutLM rather than our graft

    Args:
        backbone: Path or hub id of the weights.

    Returns:
        True when the weights are a `layoutlm` checkpoint.
    """
    return "layoutlm-" in backbone.lower() or backbone.lower().endswith("layoutlm")


def _param_groups(model: Any, config: FinetuneConfig) -> list[dict[str, Any]]:
    """
    Builds parameter groups, tolerating a baseline model that has no layout tables

    `build_param_groups` refuses a model without `LayoutEmbedding` tables on purpose, since that
    is how a run misconfigured against a plain ModernBERT gets caught. The LayoutLM baseline is
    the one legitimate exception: its 2D embeddings are pretrained, not freshly grafted, so they
    belong on the backbone rate anyway.

    Args:
        model: The model to optimize.
        config: Hyperparameters for this task.

    Returns:
        Parameter groups ready for AdamW.
    """
    try:
        return build_param_groups(
            model=model,
            learning_rate=config.learning_rate,
            layout_learning_rate=config.layout_learning_rate,
            weight_decay=config.weight_decay,
        )
    except ValueError:
        if not is_layoutlm_baseline(str(getattr(model.config, "_name_or_path", ""))):
            raise
        decayed = [p for n, p in model.named_parameters() if p.requires_grad and "bias" not in n]
        undecayed = [p for n, p in model.named_parameters() if p.requires_grad and "bias" in n]
        return [
            {
                "params": decayed,
                "lr": config.learning_rate,
                "weight_decay": config.weight_decay,
                "name": "backbone",
            },
            {
                "params": undecayed,
                "lr": config.learning_rate,
                "weight_decay": 0.0,
                "name": "no_decay",
            },
        ]


def _build_model(spec: RunSpec, meta: DatasetMeta) -> Any:
    """
    Constructs the head appropriate to the dataset's task

    A LayoutLM backbone builds transformers' own LayoutLM classes instead of ours. Its forward
    already accepts `bbox` on the same 0-1000 convention, so the processed data and the processor
    are shared with our runs, which is what makes the comparison ours rather than cross-paper.

    Args:
        spec: The run being set up.
        meta: The dataset metadata, supplying the label count and task.

    Returns:
        An initialized model with a fresh classification head.
    """
    label_kwargs = {
        "num_labels": len(meta.label_names),
        "id2label": dict(enumerate(meta.label_names)),
        "label2id": {name: index for index, name in enumerate(meta.label_names)},
    }
    if is_layoutlm_baseline(spec.backbone):
        return _build_layoutlm(spec, meta, label_kwargs)

    model_config = ModernLayoutLmConfig.from_pretrained(
        spec.backbone,
        **label_kwargs,
        # ModernBERT enables its own torch.compile path on CUDA by default, which makes triton
        # shell out to /sbin/ldconfig. That does not exist on NixOS. Disabled here rather than in
        # the config class, because it is a property of this machine and not of the architecture.
        reference_compile=False,
    )
    head = (
        ModernLayoutLmForTokenClassification
        if meta.task == "token"
        else ModernLayoutLmForSequenceClassification
    )
    return head.from_pretrained(spec.backbone, config=model_config)


def _build_layoutlm(spec: RunSpec, meta: DatasetMeta, label_kwargs: dict[str, Any]) -> Any:
    """
    Builds the original LayoutLM with the head this task needs

    Args:
        spec: The run being set up.
        meta: The dataset metadata.
        label_kwargs: num_labels and the label maps.

    Returns:
        An initialized LayoutLM model.
    """
    from transformers import (
        LayoutLMForSequenceClassification,
        LayoutLMForTokenClassification,
    )

    head = (
        LayoutLMForTokenClassification
        if meta.task == "token"
        else LayoutLMForSequenceClassification
    )
    return head.from_pretrained(spec.backbone, **label_kwargs)


def _training_args(config: FinetuneConfig, output_dir: Path, seed: int) -> TrainingArguments:
    """
    Builds the TrainingArguments for a downstream run

    Args:
        config: Hyperparameters for this task.
        output_dir: Where to write checkpoints and logs.
        seed: Random seed.

    Returns:
        Configured TrainingArguments.
    """
    return TrainingArguments(
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
        # bbox must survive column pruning. The heads name it explicitly so Trainer keeps it, and
        # this assertion is what would catch a regression in that.
        remove_unused_columns=True,
    )


def _adamw(groups: list[dict[str, Any]]) -> Any:
    """
    Builds an AdamW optimizer over prepared parameter groups

    Args:
        groups: Parameter groups carrying their own lr and weight decay.

    Returns:
        The optimizer.
    """
    from torch.optim import AdamW

    # `name` is ours, for logging, and torch rejects unknown keys.
    clean = [{k: v for k, v in group.items() if k != "name"} for group in groups]
    return AdamW(clean)


def _metric_fn(meta: DatasetMeta) -> Any:
    """
    Builds the compute_metrics callback for a dataset's task

    Args:
        meta: The dataset metadata, supplying the task and label names.

    Returns:
        A callable suitable for `Trainer(compute_metrics=...)`.
    """

    def compute(prediction: EvalPrediction) -> dict[str, float]:
        """Scores one evaluation pass."""
        logits = prediction.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        labels = np.asarray(prediction.label_ids)
        if meta.task == "token":
            return token_metrics(np.asarray(logits), labels, meta.label_names)
        return sequence_metrics(np.asarray(logits), labels)

    return compute


def _zero_boxes(dataset: Dataset) -> Dataset:
    """
    Replaces every box with the origin, for the no-bbox control

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


def _strip_prefix(key: str) -> str:
    """
    Removes Trainer's `eval_` prefix from a metric name

    Args:
        key: The metric name as Trainer reported it.

    Returns:
        The bare metric name.
    """
    return key.removeprefix("eval_")


def _write_result(result: RunResult, output_dir: Path) -> None:
    """
    Writes a run's metrics beside its checkpoint

    Args:
        result: The finished run.
        output_dir: The run's output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(asdict(result), indent=2) + "\n")
