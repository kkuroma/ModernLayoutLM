"""
Keeping benchmark documents out of the pretraining corpus

FUNSD and RVL-CDIP are both descended from IIT-CDIP, which is itself drawn from the UCSF Industry
Documents Library. Pretraining on IDL therefore risks training on the very pages we later evaluate
on, which is the flaw that makes some published numbers in this area hard to interpret.

Matching by identifier does not work: IDL keys pages by UCSF tid (`ffbb0000`) while RVL-CDIP names
them by Bates number (`10345348_10345354`). Matching by text does not work either, because the two
releases were OCR'd by different engines, so the same page yields different tokens.

What does survive both problems is the Bates number itself. These documents carry it printed on the
page, so it appears in the OCR of any release. Extracting the Bates numbers from the benchmark file
names gives a fingerprint set, and any pretraining page whose OCR contains one is dropped.

The filter over-drops slightly, since a Bates number can appear on a neighboring page of the same
document, which is the right direction to err. It also under-detects when OCR mangles the number,
so this reduces contamination rather than proving its absence. CORD and SROIE are independent of
IIT-CDIP entirely and need no filtering.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Bates numbers are long digit runs. Seven is short enough to catch the real ones and long enough
# that ordinary numbers on a page, years, amounts, phone fragments, do not collide.
MIN_BATES_DIGITS = 7

# Non-digits are stripped before comparison, so "2031316091." and "2031316091" both match.
_DIGITS = re.compile(r"\D")

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
        Every token that reduces to a long enough digit run.
    """
    found: set[str] = set()
    for token in tokens:
        text = str(token)
        # Cheap guard first: most tokens are not mostly digits, and this runs over billions.
        if len(text) < MIN_BATES_DIGITS:
            continue
        digits = _DIGITS.sub("", text)
        if len(digits) >= MIN_BATES_DIGITS:
            found.add(digits)
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

        An absent file is not an error: it means no benchmark overlaps this corpus, which is the
        case for CORD and SROIE. Callers that require filtering should check `len`.

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
