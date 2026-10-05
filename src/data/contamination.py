"""
Excludes benchmark documents out of the pre-training corpus using the document's Bates number

FUNSD and RVL-CDIP descended from IIT-CDIP which is drawn from UCSF Industry Document Library
Some pages from FUNSD and RVL-CDIP may be included in IDL
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Require 7 digits as shorter runs collide with years and amounts
MIN_BATES_DIGITS = 7

# Match digit runs as stripping non-digits glued "1011-1019." into one number
_DIGIT_RUN = re.compile(r"\d+")

FINGERPRINT_FILE = "eval_fingerprints.json"


def bates_from_filename(name: str) -> set[str]:
    """
    Extracts Bates numbers from a benchmark file name

    Args:
        name: A file name such as "10345348_10345354.jpg".

    Returns:
        Every digit run of at least `MIN_BATES_DIGITS` digits found in the stem.
    """
    stem = name.rsplit(".", 1)[0]
    parts = re.split(r"[_\-]", stem)
    return {p for p in parts if p.isdigit() and len(p) >= MIN_BATES_DIGITS}


def page_bates(tokens: list[str]) -> set[str]:
    """
    Extracts candidate Bates numbers from a page's OCR tokens

    Args:
        tokens: The page's words.

    Returns:
        Every long enough digit run appearing in any token.
    """
    found: set[str] = set()
    for token in tokens:
        text = str(token)
        # Cheap guard first: most tokens are too short to hold one, and this runs over billions
        if len(text) < MIN_BATES_DIGITS:
            continue
        for run in _DIGIT_RUN.findall(text):
            if len(run) >= MIN_BATES_DIGITS:
                found.add(run)
    return found


class Fingerprints:
    """
    The set of benchmark identifiers that must not appear in pretraining

    Args:
        bates: Bates numbers drawn from benchmark evaluation splits.
    """

    def __init__(self, bates: set[str] | None = None) -> None:
        self.bates = bates or set()

    def __len__(self) -> int:
        return len(self.bates)

    def contaminated(self, tokens: list[str]) -> str | None:
        """
        Reports whether a page carries a benchmark Bates number

        Args:
            tokens: The page's OCR tokens.

        Returns:
            The offending Bates number, or None when the page is clean.
        """
        if not self.bates:
            return None
        for candidate in page_bates(tokens):
            if candidate in self.bates:
                return candidate
        return None

    def write(self, directory: Path) -> Path:
        """
        Saves the fingerprint set beside the processed data

        Args:
            directory: Where to write `FINGERPRINT_FILE`.

        Returns:
            The path written.
        """
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / FINGERPRINT_FILE
        path.write_text(json.dumps({"bates": sorted(self.bates)}, indent=0))
        return path

    @classmethod
    def read(cls, directory: Path) -> Fingerprints:
        """
        Loads the fingerprint set, returning an empty one when absent

        An absent file means no benchmark overlaps this corpus, as for CORD and SROIE.
        Callers that require filtering should check `len`.

        Args:
            directory: Where `FINGERPRINT_FILE` lives.

        Returns:
            The fingerprints, empty when the file does not exist.
        """
        path = directory / FINGERPRINT_FILE
        if not path.is_file():
            return cls()
        payload = json.loads(path.read_text())
        return cls(set(payload.get("bates", [])))
