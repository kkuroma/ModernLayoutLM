"""
Pretraining and fine-tuning

The top layer. Imports from `src.data`, `src.model` and `src.eval`, which is why the processor is
applied here rather than in the data stage.

Two things in here exist only because the stock transformers equivalents would silently discard
the boxes:

- `LayoutMlmCollator`, because `DataCollatorForLanguageModeling` drops any column it does not know.
- `build_param_groups`, because the zero-initialized layout tables and a 2T-token backbone cannot
  share one learning rate.
"""

from .collator import LayoutMlmCollator
from .encoding import prepare_task_splits, tokenize_split
from .finetune import RunResult, RunSpec, finetune
from .optimizer import build_param_groups, describe_param_groups, layout_parameter_names

__all__ = [
    "LayoutMlmCollator",
    "RunResult",
    "RunSpec",
    "build_param_groups",
    "describe_param_groups",
    "finetune",
    "layout_parameter_names",
    "prepare_task_splits",
    "tokenize_split",
]
