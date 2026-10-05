"""
Pretraining and fine-tuning, the top layer, importing from data, model and eval

The stock collator and optimizer setup lose the boxes or the graft, hence batches and param_groups
"""

from .finetune import RunResult, RunSpec, finetune
from .pretrain import pretrain

__all__ = ["RunResult", "RunSpec", "finetune", "pretrain"]
