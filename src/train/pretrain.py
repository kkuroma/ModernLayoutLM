"""
Layout-aware masked language modeling, the stage that trains the zero-initialized layout tables

Text is masked and boxes are left intact, so recovering a token rewards reading its position
The layout tables train at their own rate and without decay, which would pull them back to zero
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import torch.multiprocessing
from datasets import Dataset
from torch import nn
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from src.config import PretrainConfig
from src.model import ModernLayoutLmConfig, ModernLayoutLmForMaskedLM, ModernLayoutLmProcessor
from src.model.modeling_modernlayoutlm import LayoutEmbedding
from src.train.batches import LayoutMlmCollator, build_packed_corpus

logger = logging.getLogger(__name__)

# Hold out a small slice as the shipped validation splits are tens of thousands of pages
EVAL_PAGES = 2000

# Biases and norm weights conventionally escape weight decay
NO_DECAY_SUFFIXES = ("bias", "norm.weight", "LayerNorm.weight")

# What a kept checkpoint needs to be fine-tuned from, leaving the optimizer state behind
KEPT_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")


@dataclass
class PretrainResult:
    """
    What a pretraining run produced

    Args:
        output_dir: Where the final weights were written.
        train_sequences: Sequences in the training set.
        eval_sequences: Sequences held back for validation.
        metrics: Final validation metrics.
        history: Per-log-step snapshots, for plotting the loss curve.
    """

    output_dir: str
    train_sequences: int
    eval_sequences: int
    metrics: dict[str, float]
    history: list[dict[str, Any]] = field(default_factory=list)


class KeepCheckpoints(TrainerCallback):
    """
    Copies the weights of every `every`th step aside, beyond the reach of checkpoint rotation

    Args:
        every: Step interval, a multiple of the save interval, or 0 to keep nothing.
    """

    def __init__(self, every: int) -> None:
        self.every = every

    def on_save(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **_: Any
    ) -> None:
        if not self.every or state.global_step % self.every:
            return
        root = Path(cast(str, args.output_dir))
        source = root / f"checkpoint-{state.global_step}"
        target = root / "kept" / source.name
        target.mkdir(parents=True, exist_ok=True)
        for name in KEPT_FILES:
            shutil.copy2(source / name, target / name)
        logger.info("kept %s", target)


def param_groups(
    model: nn.Module, learning_rate: float, layout_learning_rate: float, weight_decay: float
) -> list[dict[str, Any]]:
    """
    Splits parameters into layout, decayed backbone and undecayed backbone groups for AdamW

    Layout tables are found by module type, so renaming one cannot drop it into the backbone.

    Args:
        model: The model to optimize.
        learning_rate: Rate for the pretrained backbone.
        layout_learning_rate: Rate for the layout tables.
        weight_decay: Decay for the backbone weights that take it.

    Returns:
        Non-empty parameter groups, each carrying a `name` for logging.

    Raises:
        ValueError: If the model has no layout tables, which means the wrong class was built.
    """
    layout = {
        f"{prefix}.{name}" if prefix else name
        for prefix, module in model.named_modules()
        if isinstance(module, LayoutEmbedding)
        for name, _ in module.named_parameters(recurse=False)
    }
    if not layout:
        raise ValueError("This model has no LayoutEmbedding tables, so there is nothing to graft")
    buckets: dict[str, list[nn.Parameter]] = {"layout": [], "backbone": [], "no_decay": []}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            bucket = "no_decay" if name.endswith(NO_DECAY_SUFFIXES) else "backbone"
            buckets["layout" if name in layout else bucket].append(parameter)
    rates = {
        "layout": (layout_learning_rate, 0.0),
        "backbone": (learning_rate, weight_decay),
        "no_decay": (learning_rate, 0.0),
    }
    return [
        {"params": params, "lr": rates[name][0], "weight_decay": rates[name][1], "name": name}
        for name, params in buckets.items()
        if params
    ]


def adamw(groups: list[dict[str, Any]], fused: bool = True) -> AdamW:
    """
    Builds AdamW and logs its groups, the cheapest check that the graft happened

    Args:
        groups: Parameter groups carrying their own rate, decay and name.
        fused: Use the fused kernel, which matches the foreach one only to rounding.

    Returns:
        The optimizer.
    """
    for group in groups:
        count = sum(parameter.numel() for parameter in group["params"])
        logger.info(
            "  %-9s %3d tensors %12s params  lr %g  wd %g",
            group["name"],
            len(group["params"]),
            f"{count:,}",
            group["lr"],
            group["weight_decay"],
        )
    return AdamW([{k: v for k, v in g.items() if k != "name"} for g in groups], fused=fused)


def corpus(
    processor: ModernLayoutLmProcessor,
    datasets: list[str],
    config: PretrainConfig,
    processed_dir: Path,
    pack_dir: Path,
    num_proc: int | None = None,
    max_pages: int | None = None,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """
    Packs the pretraining corpora into train and validation sets, cached in `pack_dir`

    Args:
        processor: The processor to encode with.
        datasets: Processed corpus names, in mixing order.
        config: Pretraining hyperparameters, supplying the sequence length and page cap.
        processed_dir: Root of the processed tree.
        pack_dir: Where the packed sets are cached.
        num_proc: Worker processes for tokenization.
        max_pages: Cap on training pages per corpus, overriding `config.max_pages_per_source`.
        seed: Seed for the page subsample and the training shuffle.

    Returns:
        The shuffled training set and the validation set.

    Raises:
        ValueError: If `datasets` is empty.
    """
    if not datasets:
        raise ValueError("Pretraining needs at least one corpus")
    caps = {
        "train": max_pages if max_pages is not None else config.max_pages_per_source,
        "validation": EVAL_PAGES // len(datasets),
    }
    built = {
        split: build_packed_corpus(
            processed_dir,
            datasets,
            split,
            processor,
            config.max_length,
            pack_dir,
            num_proc,
            cap,
            seed,
            config.pack_rows,
        )
        for split, cap in caps.items()
    }
    return built["train"].shuffle(seed=seed), built["validation"]


def pretrain(
    backbone: str,
    datasets: list[str],
    config: PretrainConfig,
    processed_dir: Path,
    pack_dir: Path,
    output_dir: Path,
    seed: int = 42,
    num_proc: int | None = None,
    max_pages: int | None = None,
    max_steps: int | None = None,
    resume: bool = False,
    attn_implementation: str = "flex_attention",
) -> PretrainResult:
    """
    Runs one phase of layout-aware MLM with fp32 master weights under bf16 autocast

    bf16 master weights cost 18 F1 on FUNSD while the loss looked fine, so the weights stay fp32.

    Args:
        backbone: ModernBERT weights to graft onto, or a ModernLayoutLM checkpoint to continue.
        datasets: Processed corpus names to concatenate.
        config: Pretraining hyperparameters.
        processed_dir: Root of the processed dataset tree.
        pack_dir: Where the packed sets are cached.
        output_dir: Where checkpoints and the final weights go.
        seed: Random seed.
        num_proc: Worker processes for tokenization.
        max_pages: Cap on pages per corpus, overriding `config.max_pages_per_source`.
        max_steps: Optimizer steps in place of `config.steps`, for a probe.
        resume: Continue from the newest complete checkpoint in `output_dir`, if any.
        attn_implementation: "flex_attention" compiles the sliding window into its kernel.

    Returns:
        The run's result, also written to `output_dir / "pretrain_result.json"`.
    """
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    processor = ModernLayoutLmProcessor(tokenizer=tokenizer)
    train, validation = corpus(
        processor, datasets, config, processed_dir, pack_dir, num_proc, max_pages, seed
    )
    logger.info("train sequences %d, eval sequences %d", len(train), len(validation))

    model_config = ModernLayoutLmConfig.from_pretrained(
        backbone, reference_compile=False, attn_implementation=attn_implementation
    )
    model = ModernLayoutLmForMaskedLM.from_pretrained(backbone, config=model_config)
    # Score masked tokens only, set on the instance so every saved config keeps dense logits
    model.sparse_prediction = True
    groups = param_groups(
        model, config.learning_rate, config.layout_learning_rate, config.weight_decay
    )

    # Python 3.14 starts workers by forkserver, and each would re-read every pack shard from disk
    torch.multiprocessing.set_start_method("fork", force=True)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size * 2,
            gradient_accumulation_steps=config.grad_accum,
            max_steps=max_steps if max_steps is not None else config.steps,
            warmup_ratio=config.warmup_ratio,
            bf16=True,
            torch_compile=True,
            logging_steps=100,
            eval_strategy="steps",
            eval_steps=2000,
            save_strategy="steps",
            save_steps=2000,
            save_total_limit=config.save_total_limit,
            dataloader_num_workers=4,
            # Spawned once, so an edit to src mid-run never reaches a worker
            dataloader_persistent_workers=True,
            seed=seed,
            report_to=[],
            # The collator builds the batch, so Trainer must not prune bbox out of the dataset
            remove_unused_columns=False,
        ),
        train_dataset=cast(Any, train),
        eval_dataset=cast(Any, validation),
        data_collator=LayoutMlmCollator(tokenizer, config.mlm_probability),
        optimizers=(adamw(groups), None),
        callbacks=[KeepCheckpoints(config.keep_every)],
    )
    checkpoint = _latest_checkpoint(output_dir) if resume else None
    logger.info("starting from %s", checkpoint or backbone)
    trainer.train(resume_from_checkpoint=checkpoint)
    metrics = trainer.evaluate(cast(Any, validation))

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    result = PretrainResult(
        output_dir=str(output_dir),
        train_sequences=len(train),
        eval_sequences=len(validation),
        metrics={k.removeprefix("eval_"): v for k, v in metrics.items() if isinstance(v, float)},
        history=list(trainer.state.log_history),
    )
    (output_dir / "pretrain_result.json").write_text(json.dumps(asdict(result), indent=2) + "\n")
    return result


def _latest_checkpoint(output_dir: Path) -> str | None:
    """
    Finds the newest checkpoint that finished writing its optimizer state

    Args:
        output_dir: The run's output directory.

    Returns:
        The checkpoint path, or None when there is nothing to resume from.
    """
    usable = [
        path
        for path in output_dir.glob("checkpoint-*")
        if (path / "trainer_state.json").is_file() and (path / "optimizer.pt").is_file()
    ]
    return str(max(usable, key=lambda path: int(path.name.split("-")[-1]))) if usable else None
