"""
Tests for the benchmark contamination filter

A filter that silently stopped matching would look plausible in every count it reports
These pin the matching behavior rather than the plumbing around it
"""

from pathlib import Path

from src.data.contamination import (
    MIN_BATES_DIGITS,
    Fingerprints,
    bates_from_filename,
    page_bates,
)


def test_bates_extracted_from_a_paired_filename() -> None:
    """RVL-CDIP names pages by a Bates range, and both ends must be captured."""
    assert bates_from_filename("10345348_10345354.jpg") == {"10345348", "10345354"}


def test_bates_extracted_from_a_single_filename() -> None:
    """Some pages carry one Bates number rather than a range."""
    assert bates_from_filename("2501634600.jpg") == {"2501634600"}


def test_short_numbers_are_not_bates() -> None:
    """
    A year or a page number must not become a fingerprint

    A short digit run would match thousands of unrelated pages and gut the corpus.
    """
    assert bates_from_filename("2020_15.jpg") == set()
    assert all(len(b) >= MIN_BATES_DIGITS for b in bates_from_filename("10345348_12.jpg"))


def test_non_numeric_filenames_yield_nothing() -> None:
    """Some RVL-CDIP names are not Bates at all, and must not produce junk fingerprints."""
    assert bates_from_filename("PUBLICATIONS034845-4.jpg") == set()


def test_page_bates_survives_ocr_punctuation() -> None:
    """
    OCR appends stray punctuation, so a Bates number is read as the digit run inside a token

    Without this the same page read by two OCR engines would fail to match.
    """
    assert "2031316091" in page_bates(["2031316091."])
    assert "2031316091" in page_bates(["(2031316091)"])


def test_page_bates_does_not_join_across_a_separator() -> None:
    """
    A separator ends a number, so a range is two short numbers rather than one long one

    Deleting every non-digit instead turned the citation "1011-1019." into "10111019" and the
    confidence interval "(0.092-0.262)" into "00920262". Both are long enough to look like Bates
    numbers, both collided with real ones, and both dropped clean PubTables pages.
    """
    assert page_bates(["1011-1019."]) == set()
    assert page_bates(["(0.092-0.262)"]) == set()
    assert page_bates(["R10,001-R15,000"]) == set()
    # A genuine Bates number beside a range is still found
    assert page_bates(["2031316091/1011-1019"]) == {"2031316091"}


def test_page_bates_ignores_short_tokens() -> None:
    """Ordinary numbers on a page must not be treated as identifiers."""
    assert page_bates(["2020", "15", "3.14", "hello"]) == set()


def test_contaminated_page_is_detected() -> None:
    """A page carrying a benchmark Bates number must be reported."""
    prints = Fingerprints({"2031316091"})
    assert prints.contaminated(["memo", "dated", "2031316091"]) == "2031316091"


def test_clean_page_passes() -> None:
    """A page with no benchmark identifier must not be dropped."""
    prints = Fingerprints({"2031316091"})
    assert prints.contaminated(["memo", "dated", "1999", "9999999"]) is None


def test_empty_fingerprints_never_filter() -> None:
    """
    With no fingerprints the filter is a no-op

    CORD and SROIE are independent of IIT-CDIP, so a corpus with no overlap must pass everything
    through rather than silently dropping pages.
    """
    assert Fingerprints().contaminated(["2031316091"]) is None


def test_fingerprints_roundtrip(tmp_path: Path) -> None:
    """The set must survive being written and reloaded, or a rebuild loses the filter."""
    prints = Fingerprints({"10345348", "2501634600"})
    prints.write(tmp_path)
    assert Fingerprints.read(tmp_path).bates == prints.bates


def test_missing_file_reads_as_empty(tmp_path: Path) -> None:
    """An absent file means no overlap, not an error."""
    assert len(Fingerprints.read(tmp_path)) == 0
