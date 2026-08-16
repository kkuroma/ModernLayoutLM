"""
Primitives for turning source coordinates into our canonical form

Sources arrive as pixels, four-corner quads, line-level boxes or coordinates on an unstated scale
The conversion rules live here so they are written once and tested once
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

# Boxes are integers on this canonical scale
COORDINATE_SCALE = 1000

# A four-corner quad as CORD stores it: corners 1 to 4 clockwise from the top left
QUAD_KEYS = ("x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4")


def clamp_box(box: Sequence[float], scale: int = COORDINATE_SCALE) -> list[int]:
    """
    Clamps a box into `[0, scale]` and orders its corners

    Sources routinely emit coordinates a pixel or two outside the page, and occasionally emit
    `x1 < x0` for right-to-left or rotated text. Both are repaired rather than rejected, since
    dropping the word would lose its text too.

    Args:
        box: Four coordinates as (x0, y0, x1, y1).
        scale: Upper bound for every coordinate.

    Returns:
        A box with `0 <= x0 <= x1 <= scale` and `0 <= y0 <= y1 <= scale`.

    Raises:
        ValueError: If `box` does not hold exactly four coordinates.
    """
    if len(box) != 4:
        raise ValueError(f"Expected 4 coordinates, got {len(box)}")
    x0, y0, x1, y1 = (max(0, min(int(value), scale)) for value in box)
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def normalize_box(
    box: Sequence[float],
    page_width: float,
    page_height: float,
    scale: int = COORDINATE_SCALE,
) -> list[int]:
    """
    Scales a pixel box onto the canonical `[0, scale]` grid

    Args:
        box: Four pixel coordinates as (x0, y0, x1, y1).
        page_width: Page width in the same units as the box.
        page_height: Page height in the same units as the box.
        scale: Target upper bound.

    Returns:
        The clamped, normalized box as integers.

    Raises:
        ValueError: If the page has a non-positive dimension, or the box is malformed.
    """
    if page_width <= 0 or page_height <= 0:
        raise ValueError(f"Page dimensions must be positive, got {page_width}x{page_height}")
    if len(box) != 4:
        raise ValueError(f"Expected 4 coordinates, got {len(box)}")
    scaled = [
        box[0] / page_width * scale,
        box[1] / page_height * scale,
        box[2] / page_width * scale,
        box[3] / page_height * scale,
    ]
    return clamp_box([int(round(value)) for value in scaled], scale)


def quad_to_xyxy(quad: Mapping[str, float]) -> list[float]:
    """
    Collapses a four-corner quad into an axis-aligned box

    CORD stores each word as four corners, which supports rotated text. We take the enclosing
    axis-aligned rectangle, because the model's embedding tables are axis-aligned anyway.

    Args:
        quad: Corner coordinates keyed `x1`..`x4` and `y1`..`y4`.

    Returns:
        The enclosing box as (x0, y0, x1, y1), in the quad's own units.

    Raises:
        KeyError: If any of the eight corner keys is missing.
    """
    missing = [key for key in QUAD_KEYS if key not in quad]
    if missing:
        raise KeyError(f"Quad is missing {missing}")
    xs = [quad["x1"], quad["x2"], quad["x3"], quad["x4"]]
    ys = [quad["y1"], quad["y2"], quad["y3"], quad["y4"]]
    return [min(xs), min(ys), max(xs), max(ys)]


def split_segment(text: str, box: Sequence[float]) -> list[tuple[str, list[float]]]:
    """
    Splits a multi-word segment into words, dividing its box along the x axis

    Several sources only box a whole line or phrase. Word extents are estimated by character
    count, including the separating spaces, which assumes roughly monospaced horizontal text. It
    is an approximation, but a far better one than giving every word of a line the same wide box.

    Args:
        text: The segment's text, possibly several whitespace-separated words.
        box: The segment's box as (x0, y0, x1, y1), in any units.

    Returns:
        One (word, box) pair per word, left to right. Empty when `text` has no words.
    """
    words = text.split()
    if not words:
        return []
    if len(words) == 1:
        return [(words[0], list(box))]

    x0, y0, x1, y1 = box
    span = x1 - x0
    # Character budget includes one space between each pair of words, matching the source text
    total = sum(len(word) for word in words) + (len(words) - 1)
    if span <= 0 or total <= 0:
        return [(word, list(box)) for word in words]

    pairs: list[tuple[str, list[float]]] = []
    cursor = 0
    for index, word in enumerate(words):
        start = x0 + span * cursor / total
        end = x0 + span * (cursor + len(word)) / total
        pairs.append((word, [start, y0, end, y1]))
        cursor += len(word) + 1 if index < len(words) - 1 else len(word)
    return pairs


def bio_from_runs(
    labels: Sequence[int], names: Sequence[str], outside: int
) -> tuple[list[int], list[str]]:
    """
    Converts flat per-word field labels into BIO tags

    Some sources tag each word with its field but never mark where one entity ends and the next
    begins. Contiguous runs of the same field are treated as one entity, so the first word gets
    `B-` and the rest `I-`. Two adjacent entities of the same field will merge, which is the
    standard limitation of recovering BIO after the fact.

    Args:
        labels: One field index per word, indexing into `names`.
        names: Field names, including the outside class.
        outside: The index in `names` meaning "not part of any field".

    Returns:
        The BIO label indices per word, and the BIO label names they index into.

    Raises:
        ValueError: If a label falls outside `names`, or `outside` is not a valid index.
    """
    if not 0 <= outside < len(names):
        raise ValueError(f"outside index {outside} is not a valid index into {len(names)} names")
    fields = [name for index, name in enumerate(names) if index != outside]
    bio_names = ["O"] + [f"{prefix}-{field}" for field in fields for prefix in ("B", "I")]
    bio_index = {name: index for index, name in enumerate(bio_names)}

    tags: list[int] = []
    previous = outside
    for label in labels:
        if not 0 <= label < len(names):
            raise ValueError(f"Label {label} is outside the {len(names)} known names")
        if label == outside:
            tags.append(0)
        else:
            prefix = "I" if label == previous else "B"
            tags.append(bio_index[f"{prefix}-{names[label]}"])
        previous = label
    return tags, bio_names
