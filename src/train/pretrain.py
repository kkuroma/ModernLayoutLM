"""
Layout-aware masked language modelling

The stage that makes the graft worth having. Fine-tuning alone leaves the zero-initialized layout
tables with only a few hundred documents to learn from, and the FUNSD skip-MLM control confirms it:
0.510 entity F1 with boxes against 0.508 without, which is no difference at all. Here the tables see
hundreds of thousands of pages.

The objective is LayoutLM's MVLM. Text is masked, boxes are left untouched, so recovering a masked
token rewards reading its position on the page.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer, Trainer, TrainingArguments

from src.config import PretrainConfig
from src.model import ModernLayoutLmConfig, ModernLayoutLmForMaskedLM, ModernLayoutLmProcessor
from src.train.collator import LayoutMlmCollator
from src.train.encoding import load_split, tokenize_split
from src.train.optimizer import build_param_groups, describe_param_groups

logger = logging.getLogger(__name__)

# Pages held back for a validation loss. The corpora ship their own validation splits, but they are
# tens of thousands of pages and scoring all of them every epoch is wasted time.
EVAL_PAGES = 2000


@dataclass
class PretrainResult:
    """
    What a pretraining run produced

    Args:
        output_dir: Where the final weights were written.
        train_sequences: Chunks in the training set after tokenization.
        eval_sequences: Chunks held back for validation.
        metrics: Final validation metrics.
        history: Per-log-step snapshots, for plotting the loss curve.
    """

    output_dir: str
    train_sequences: int
    eval_sequences: int
    metrics: dict[str, float]
    history: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        """
        Renders the headline numbers on one line

        Returns:
            A printable string.
        """
        scores = "  ".join(f"{key}={value:.4f}" for key, value in sorted(self.metrics.items()))
        return f"pretrain: {self.train_sequences} train chunks  {scores}"


def pretrain(
    backbone: str,
    datasets: list[str],
    config: PretrainConfig,
    processed_dir: Path,
    output_dir: Path,
    seed: int = 42,
    num_proc: int | None = None,
    max_pages: int | None = None,
    max_steps: int | None = None,
) -> PretrainResult:
    """
    Runs layout-aware MLM over one or more pretraining corpora

    Args:
        backbone: Path or hub id of the ModernBERT weights to graft onto.
        datasets: Processed corpus names to concatenate, such as pubtables and doclaynet.
        config: Pretraining hyperparameters.
        processed_dir: Root of the processed dataset tree.
        output_dir: Where to write checkpoints and the final weights.
        seed: Random seed.
        num_proc: Worker processes for tokenization.
        max_pages: Cap on pages per corpus, for a throughput probe.
        max_steps: Cap on optimizer steps, for a throughput probe.

    Returns:
        The run's result.

    Raises:
        ValueError: If `datasets` is empty.
    """
    if not datasets:
        raise ValueError("Pretraining needs at least one corpus")

    tokenizer = AutoTokenizer.from_pretrained(backbone)
    processor = ModernLayoutLmProcessor(tokenizer=tokenizer)

    train_parts: list[Dataset] = []
    eval_parts: list[Dataset] = []
    for name in datasets:
        logger.info("tokenizing %s", name)
        train_parts.append(
            _encode(processed_dir, name, "train", processor, config, num_proc, max_pages)
        )
        eval_parts.append(
            _encode(
                processed_dir,
                name,
                "validation",
                processor,
                config,
                num_proc,
                EVAL_PAGES // len(datasets),
            )
        )

    train_dataset = _merge(train_parts).shuffle(seed=seed)
    eval_dataset = _merge(eval_parts)
    logger.info("train chunks %d, eval chunks %d", len(train_dataset), len(eval_dataset))

    model = ModernLayoutLmForMaskedLM.from_pretrained(
        backbone,
        config=ModernLayoutLmConfig.from_pretrained(backbone, reference_compile=False),
    )
    groups = build_param_groups(
        model=model,
        learning_rate=config.learning_rate,
        layout_learning_rate=config.layout_learning_rate,
        weight_decay=config.weight_decay,
    )
    logger.info("parameter groups:\n%s", describe_param_groups(groups))

    trainer = Trainer(
        model=model,
        args=_training_args(config, output_dir, seed, max_steps),
        train_dataset=cast(Any, train_dataset),
        eval_dataset=cast(Any, eval_dataset),
        data_collator=LayoutMlmCollator(
            tokenizer=tokenizer,
            mlm_probability=config.mlm_probability,
        ),
        optimizers=(_adamw(groups), None),
    )
    trainer.train()
    metrics = trainer.evaluate(cast(Any, eval_dataset))

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))

    result = PretrainResult(
        output_dir=str(output_dir),
        train_sequences=len(train_dataset),
        eval_sequences=len(eval_dataset),
        metrics={k.removeprefix("eval_"): v for k, v in metrics.items() if isinstance(v, float)},
        history=list(trainer.state.log_history),
    )
    (output_dir / "pretrain_result.json").write_text(json.dumps(asdict(result), indent=2) + "\n")
    return result


def _encode(
    processed_dir: Path,
    name: str,
    split: str,
    processor: ModernLayoutLmProcessor,
    config: PretrainConfig,
    num_proc: int | None,
    max_pages: int | None,
) -> Dataset:
    """
    Loads and tokenizes one corpus split into fixed-length chunks

    Args:
        processed_dir: Root of the processed tree.
        name: The corpus name.
        split: The split name.
        processor: The processor to encode with.
        config: Pretraining hyperparameters, supplying the sequence length.
        num_proc: Worker processes for tokenization.
        max_pages: Cap on pages to read, or None for all.

    Returns:
        A tokenized dataset of `input_ids`, `attention_mask` and `bbox`.
    """
    raw = load_split(processed_dir, name, split)
    if max_pages is not None and max_pages < len(raw):
        raw = raw.select(range(max_pages))
    return tokenize_split(
        dataset=raw,
        processor=processor,
        task="pretrain",
        max_length=config.max_length,
        num_proc=num_proc,
    )


def _merge(parts: list[Dataset]) -> Dataset:
    """
    Concatenates tokenized corpora into one dataset

    Args:
        parts: Tokenized datasets sharing the same columns.

    Returns:
        The concatenation, or the single part when there is only one.
    """
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


def _training_args(
    config: PretrainConfig,
    output_dir: Path,
    seed: int,
    max_steps: int | None,
) -> TrainingArguments:
    """
    Builds the TrainingArguments for the pretraining run

    Args:
        config: Pretraining hyperparameters.
        output_dir: Where to write checkpoints and logs.
        seed: Random seed.
        max_steps: Cap on optimizer steps, for a throughput probe.

    Returns:
        Configured TrainingArguments.
    """
    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,
        gradient_accumulation_steps=config.grad_accum,
        num_train_epochs=config.epochs,
        max_steps=max_steps if max_steps is not None else -1,
        warmup_ratio=config.warmup_ratio,
        bf16=True,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=2000,
        save_strategy="steps",
        save_steps=2000,
        save_total_limit=config.save_total_limit,
        dataloader_num_workers=4,
        seed=seed,
        report_to=[],
        # The collator builds the batch, so Trainer must not prune bbox out of the dataset first.
        remove_unused_columns=False,
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

    return AdamW([{k: v for k, v in group.items() if k != "name"} for group in groups])
