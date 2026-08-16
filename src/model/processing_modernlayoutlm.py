"""
Turning words and boxes into model inputs

There is no image tower, so this only tokenizes words and broadcasts each box across its subwords
This module is copied into exported model directories and may import only torch and transformers
"""

from __future__ import annotations

from typing import Any

from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import BatchEncoding

# Follow LayoutLM: CLS at the origin, SEP at the far corner, padding masked out
CLS_BOX = (0, 0, 0, 0)
SEP_BOX = (1000, 1000, 1000, 1000)
PAD_BOX = (0, 0, 0, 0)

# Label id meaning "do not compute loss here", the torch cross-entropy default
IGNORE_INDEX = -100


class ModernLayoutLmProcessor(ProcessorMixin):
    """
    Pairs a fast tokenizer with box and label expansion over subwords

    Args:
        tokenizer: Any fast tokenizer. Injected rather than hardcoded so the LayoutLM baseline can
            reuse this class with `LayoutLMTokenizerFast` over identical data.
        coordinate_scale: The scale boxes are expected to already be normalized to.
        label_all_subwords: When True every subword of a word carries the word's label. When
            False only the first subword does and the rest get `IGNORE_INDEX`, which is the
            convention the published FUNSD and CORD numbers use.
    """

    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        tokenizer: Any,
        coordinate_scale: int = 1000,
        label_all_subwords: bool = False,
    ) -> None:
        self.coordinate_scale = coordinate_scale
        self.label_all_subwords = label_all_subwords
        super().__init__(tokenizer)

    # Diverge as ProcessorMixin.__call__ is typed for image and audio processors
    def __call__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        words: list[str] | list[list[str]],
        boxes: list[list[int]] | list[list[list[int]]],
        word_labels: list[int] | list[list[int]] | None = None,
        max_length: int | None = None,
        padding: bool | str = "max_length",
        truncation: bool = True,
        return_overflowing_tokens: bool = False,
        stride: int = 0,
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> BatchEncoding:
        """
        Tokenizes words and aligns boxes, and optionally labels, to the resulting subwords

        Accepts a single document or a batch of them, detected by whether `words` holds strings
        or lists. Special tokens receive the sentinel boxes defined in this module. When
        `return_overflowing_tokens` is set, a long page becomes several chunks and the boxes
        follow each chunk, so no realignment is needed downstream.

        Args:
            words: One document's words, or a batch of documents' words.
            boxes: One box per word as (x0, y0, x1, y1), already normalized to
                `coordinate_scale`, batched to match `words`.
            word_labels: One class index per word, batched to match `words`. Omit for
                pretraining and for document classification.
            max_length: Truncation length. Defaults to the tokenizer's own.
            padding: Passed to the tokenizer.
            truncation: Passed to the tokenizer.
            return_overflowing_tokens: Whether to emit every chunk of an over-length document
                rather than only the first.
            stride: Overlap in tokens between consecutive chunks.
            return_tensors: "pt" for torch tensors, None for lists.
            **kwargs: Forwarded to the tokenizer.

        Returns:
            A BatchEncoding with `input_ids`, `attention_mask`, `bbox`, and `labels` when
            `word_labels` was given.

        Raises:
            ValueError: If `words` and `boxes` disagree in length, or a coordinate falls outside
                `[0, coordinate_scale]`.
        """
        batched = bool(words) and isinstance(words[0], list)
        batch_words: list[list[str]] = words if batched else [words]  # type: ignore[assignment]
        batch_boxes: list[list[list[int]]] = boxes if batched else [boxes]  # type: ignore[assignment]
        batch_labels: list[list[int]] | None = None
        if word_labels is not None:
            batch_labels = word_labels if batched else [word_labels]  # type: ignore[assignment]

        self._validate(batch_words, batch_boxes, batch_labels)

        encoding = self.tokenizer(
            text=batch_words,
            is_split_into_words=True,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_overflowing_tokens=return_overflowing_tokens,
            stride=stride,
            return_tensors=None,
            **kwargs,
        )

        sample_map = encoding.get("overflow_to_sample_mapping")
        if sample_map is None:
            sample_map = list(range(len(encoding["input_ids"])))

        aligned_boxes: list[list[list[int]]] = []
        aligned_labels: list[list[int]] = []
        for chunk_index, sample_index in enumerate(sample_map):
            word_ids = encoding.word_ids(chunk_index)
            source_boxes = batch_boxes[sample_index]
            aligned_boxes.append(self._expand_boxes(word_ids, source_boxes))
            if batch_labels is not None:
                aligned_labels.append(self._expand_labels(word_ids, batch_labels[sample_index]))

        encoding["bbox"] = aligned_boxes
        if batch_labels is not None:
            encoding["labels"] = aligned_labels

        if return_tensors is not None:
            encoding.convert_to_tensors(tensor_type=return_tensors)
        return encoding

    def _validate(
        self,
        batch_words: list[list[str]],
        batch_boxes: list[list[list[int]]],
        batch_labels: list[list[int]] | None,
    ) -> None:
        """
        Checks that the parallel inputs line up and the coordinates are in range

        Catching this here turns a silent misalignment, which would train a model on wrong boxes
        without ever erroring, into an immediate failure.

        Args:
            batch_words: Words per document.
            batch_boxes: Boxes per document.
            batch_labels: Labels per document, or None.

        Raises:
            ValueError: On a length mismatch or an out of range coordinate.
        """
        if len(batch_words) != len(batch_boxes):
            raise ValueError(f"Got {len(batch_words)} word lists but {len(batch_boxes)} box lists")
        if batch_labels is not None and len(batch_labels) != len(batch_words):
            raise ValueError(
                f"Got {len(batch_words)} word lists but {len(batch_labels)} label lists"
            )

        for index, (doc_words, doc_boxes) in enumerate(zip(batch_words, batch_boxes, strict=True)):
            if len(doc_words) != len(doc_boxes):
                raise ValueError(
                    f"Document {index} has {len(doc_words)} words but {len(doc_boxes)} boxes"
                )
            if batch_labels is not None and len(batch_labels[index]) != len(doc_words):
                raise ValueError(
                    f"Document {index} has {len(doc_words)} words "
                    f"but {len(batch_labels[index])} labels"
                )
            for box in doc_boxes:
                if len(box) != 4:
                    raise ValueError(f"Document {index} has a box with {len(box)} coordinates")
                if any(not 0 <= value <= self.coordinate_scale for value in box):
                    raise ValueError(
                        f"Document {index} has box {box} outside "
                        f"[0, {self.coordinate_scale}]. Normalize before calling the processor."
                    )

    def _expand_boxes(
        self,
        word_ids: list[int | None],
        source_boxes: list[list[int]],
    ) -> list[list[int]]:
        """
        Assigns a box to every token position in one chunk

        Args:
            word_ids: Per-token index into `source_boxes`, None for special and padding tokens.
            source_boxes: One box per word of the source document.

        Returns:
            One box per token position.
        """
        special_boxes = self._special_boxes(word_ids)
        return [
            list(source_boxes[word_id]) if word_id is not None else special_boxes[position]
            for position, word_id in enumerate(word_ids)
        ]

    def _special_boxes(self, word_ids: list[int | None]) -> dict[int, list[int]]:
        """
        Chooses a sentinel box for each token that came from no word

        Leading specials anchor the origin as CLS does, the first trailing special takes the far
        corner as SEP does, and everything past it is padding at the origin.

        Args:
            word_ids: Per-token index into the source words, None for non-word tokens.

        Returns:
            A mapping from token position to sentinel box.
        """
        word_positions = [i for i, word_id in enumerate(word_ids) if word_id is not None]
        last_word = word_positions[-1] if word_positions else -1

        boxes: dict[int, list[int]] = {}
        seen_separator = False
        for position, word_id in enumerate(word_ids):
            if word_id is not None:
                continue
            if position < last_word:
                boxes[position] = list(CLS_BOX)
            elif not seen_separator:
                boxes[position] = list(SEP_BOX)
                seen_separator = True
            else:
                boxes[position] = list(PAD_BOX)
        return boxes

    def _expand_labels(
        self,
        word_ids: list[int | None],
        source_labels: list[int],
    ) -> list[int]:
        """
        Assigns a label to every token position in one chunk

        Special and padding tokens always get `IGNORE_INDEX`. Whether the non-first subwords of a
        word inherit the word's label follows `label_all_subwords`.

        Args:
            word_ids: Per-token index into `source_labels`, None for non-word tokens.
            source_labels: One class index per word of the source document.

        Returns:
            One label per token position.
        """
        labels: list[int] = []
        previous = None
        for word_id in word_ids:
            if word_id is None:
                labels.append(IGNORE_INDEX)
            elif word_id != previous or self.label_all_subwords:
                labels.append(source_labels[word_id])
            else:
                labels.append(IGNORE_INDEX)
            previous = word_id
        return labels


__all__ = ["ModernLayoutLmProcessor"]
