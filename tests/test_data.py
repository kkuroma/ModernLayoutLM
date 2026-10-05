"""
Tests for the data stage: the schema and its coordinate primitives, the adapters and the filter

Each adapter is parsed against a hand-built record, so nothing here needs the network
A document whose boxes and words disagree still trains and still shows a falling loss
"""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from src.data.contamination import MIN_BATES_DIGITS, Fingerprints, bates_from_filename, page_bates
from src.data.schema import (
    COORDINATE_SCALE,
    DatasetMeta,
    Document,
    bio_from_runs,
    clamp_box,
    clamp_boxes,
    normalize_box,
    quad_to_xyxy,
    split_segment,
    write_documents,
)
from src.data.sources import GROUPS, SOURCES, cord, idl, realkie, resolve_names, sroie


def make_document(**overrides) -> Document:
    """Builds a valid two-word document, with fields optionally replaced."""
    fields = {
        "doc_id": "test-0",
        "source": "test",
        "words": ["Total", "42.00"],
        "bboxes": [[10, 20, 60, 30], [70, 20, 120, 30]],
        "page_width": 800,
        "page_height": 1000,
        "word_labels": [1, 2],
    }
    fields.update(overrides)
    return Document(**fields)


def test_valid_document_passes() -> None:
    """The happy path must not raise."""
    make_document().validate()


def test_word_and_box_mismatch_raises() -> None:
    """One box per word is the core invariant."""
    with pytest.raises(ValueError, match="words but"):
        make_document(bboxes=[[10, 20, 60, 30]]).validate()


def test_label_mismatch_raises() -> None:
    """Labels must line up with words too, or the entities shift."""
    with pytest.raises(ValueError, match="labels"):
        make_document(word_labels=[1]).validate()


def test_empty_document_raises() -> None:
    """A page with no words cannot be trained on and must not reach parquet."""
    with pytest.raises(ValueError, match="no words"):
        make_document(words=[], bboxes=[], word_labels=[]).validate()


def test_out_of_scale_box_raises() -> None:
    """Raw pixels left unnormalized are the most likely adapter bug."""
    with pytest.raises(ValueError, match=r"outside \[0, 1000\]"):
        make_document(bboxes=[[0, 0, 2480, 3500], [1, 1, 2, 2]]).validate()


def test_inverted_box_raises() -> None:
    """Validation runs after clamping, so an inverted box here means clamping was skipped."""
    with pytest.raises(ValueError, match="inverted"):
        make_document(bboxes=[[60, 20, 10, 30], [1, 1, 2, 2]]).validate()


def test_malformed_box_raises() -> None:
    """A box that is not four numbers must be caught before pyarrow sees it."""
    with pytest.raises(ValueError, match="4 coordinates"):
        make_document(bboxes=[[10, 20, 60], [1, 1, 2, 2]]).validate()


def test_roundtrip_through_parquet(tmp_path: Path) -> None:
    """Documents must survive the schema unchanged, boxes included."""
    documents = [make_document(), make_document(doc_id="test-1", doc_label=3)]
    path = tmp_path / "shard.parquet"
    assert write_documents(documents, path) == 2

    table = pq.read_table(path)
    rows = table.to_pylist()
    assert [row["doc_id"] for row in rows] == ["test-0", "test-1"]
    assert rows[0]["bboxes"] == [[10, 20, 60, 30], [70, 20, 120, 30]]
    assert rows[0]["doc_label"] is None
    assert rows[1]["doc_label"] == 3


def test_writing_nothing_raises(tmp_path: Path) -> None:
    """
    An empty parquet file is worse than no file

    It reads back as a valid but empty split, so a failed build would look like a finished one.
    """
    with pytest.raises(ValueError, match="empty parquet"):
        write_documents([], tmp_path / "shard.parquet")


def test_meta_roundtrip(tmp_path: Path) -> None:
    """The label sidecar must survive a write and read unchanged."""
    meta = DatasetMeta(
        source="test",
        task="token",
        label_names=["O", "B-X", "I-X"],
        splits={"train": 10},
        notes="a note",
    )
    meta.write(tmp_path)
    assert DatasetMeta.read(tmp_path) == meta


def test_missing_meta_raises(tmp_path: Path) -> None:
    """A dataset without its sidecar has an unknown label vocabulary and must not load."""
    with pytest.raises(FileNotFoundError, match="rebuild"):
        DatasetMeta.read(tmp_path)


def test_clamp_box_bounds_coordinates() -> None:
    """Coordinates outside the scale are pulled to the edge, not rejected."""
    assert clamp_box([-20, -5, 2000, 1500]) == [0, 0, COORDINATE_SCALE, COORDINATE_SCALE]


def test_clamp_box_reorders_inverted_corners() -> None:
    """
    A box with x1 < x0 is repaired rather than dropped

    Rotated and right-to-left text produce these, and dropping the box would lose the word too.
    """
    assert clamp_box([900, 800, 100, 200]) == [100, 200, 900, 800]


def test_clamp_box_rejects_wrong_arity() -> None:
    """A box that is not four numbers is a bug in the adapter, not bad data."""
    with pytest.raises(ValueError, match="4 coordinates"):
        clamp_box([1, 2, 3])


def test_clamp_boxes_matches_clamp_box_row_by_row() -> None:
    """The vectorized path must agree with the scalar one on negatives, overshoot and inversions."""
    boxes = np.random.default_rng(0).uniform(-50, 1100, size=(500, 4))
    boxes[:5] = [
        [-0.5, 0.9, 999.9, 1000.4],
        [10, 20, 5, 1],
        [0, 0, 0, 0],
        [3.7, -3.7, 1e9, 2],
        [1, 1, 1, 1],
    ]
    assert clamp_boxes(boxes) == [clamp_box(box.tolist()) for box in boxes]


def test_clamp_boxes_rejects_non_finite() -> None:
    """A NaN would otherwise cast to an arbitrary integer and pass as a coordinate."""
    with pytest.raises(ValueError, match="finite"):
        clamp_boxes(np.array([[0.0, np.nan, 1.0, 1.0]]))


def test_normalize_box_scales_pixels() -> None:
    """A box spanning half the page lands at half the scale."""
    assert normalize_box([0, 0, 100, 50], 200, 100) == [0, 0, 500, 500]


def test_normalize_box_covers_full_page() -> None:
    """A full-page box maps exactly onto the scale, with no off-by-one."""
    assert normalize_box([0, 0, 762, 1000], 762, 1000) == [0, 0, 1000, 1000]


def test_normalize_box_rejects_degenerate_page() -> None:
    """A zero-width page would divide by zero, so it must be caught."""
    with pytest.raises(ValueError, match="positive"):
        normalize_box([0, 0, 10, 10], 0, 100)


def test_quad_to_xyxy_takes_the_enclosing_box() -> None:
    """A rotated quad collapses to the axis-aligned rectangle that contains it."""
    quad = {"x1": 10, "y1": 20, "x2": 50, "y2": 15, "x3": 55, "y3": 40, "x4": 12, "y4": 45}
    assert quad_to_xyxy(quad) == [10, 15, 55, 45]


def test_quad_to_xyxy_reports_missing_corners() -> None:
    """A truncated quad names what is missing rather than raising a bare KeyError."""
    with pytest.raises(KeyError, match="missing"):
        quad_to_xyxy({"x1": 0, "y1": 0})


def test_split_segment_passes_single_words_through() -> None:
    """One word keeps its box exactly."""
    assert split_segment("Total", [10, 20, 60, 30]) == [("Total", [10, 20, 60, 30])]


def test_split_segment_divides_by_character_count() -> None:
    """
    A multi-word segment is divided along x in proportion to word length

    Word boxes must tile the segment left to right without overlap, inside the original box.
    The vertical extent is shared, since the segment is one line.
    """
    pairs = split_segment("ab cd", [0, 10, 100, 20])
    assert [word for word, _ in pairs] == ["ab", "cd"]
    boxes = [box for _, box in pairs]
    assert boxes[0][0] == 0
    assert boxes[-1][2] == 100
    assert boxes[0][2] <= boxes[1][0]
    assert all(box[1] == 10 and box[3] == 20 for box in boxes)


def test_split_segment_handles_no_words() -> None:
    """Whitespace-only OCR output contributes nothing."""
    assert split_segment("   ", [0, 0, 10, 10]) == []


def test_split_segment_survives_zero_width_box() -> None:
    """A degenerate box still yields every word, just stacked at the same place."""
    pairs = split_segment("a b c", [50, 0, 50, 10])
    assert len(pairs) == 3


def test_bio_from_runs_marks_entity_starts() -> None:
    """A contiguous run of one field becomes one entity: B then I."""
    names = ["COMPANY", "DATE", "O"]
    tags, bio_names = bio_from_runs([0, 0, 2, 1, 1, 1], names, outside=2)
    readable = [bio_names[tag] for tag in tags]
    assert readable == ["B-COMPANY", "I-COMPANY", "O", "B-DATE", "I-DATE", "I-DATE"]


def test_bio_from_runs_restarts_after_outside() -> None:
    """A field interrupted by O starts a new entity rather than continuing the old one."""
    names = ["TOTAL", "O"]
    tags, bio_names = bio_from_runs([0, 1, 0], names, outside=1)
    assert [bio_names[tag] for tag in tags] == ["B-TOTAL", "O", "B-TOTAL"]


def test_bio_from_runs_places_outside_first() -> None:
    """O must be label 0 in the output vocabulary regardless of where it sat in the source."""
    _, bio_names = bio_from_runs([], ["COMPANY", "DATE", "ADDRESS", "TOTAL", "O"], outside=4)
    assert bio_names[0] == "O"
    assert bio_names == [
        "O",
        "B-COMPANY",
        "I-COMPANY",
        "B-DATE",
        "I-DATE",
        "B-ADDRESS",
        "I-ADDRESS",
        "B-TOTAL",
        "I-TOTAL",
    ]


def test_bio_from_runs_rejects_unknown_label() -> None:
    """A label id with no name means the source scheme was misread."""
    with pytest.raises(ValueError, match="outside the"):
        bio_from_runs([5], ["A", "O"], outside=1)


def test_bio_from_runs_rejects_bad_outside_index() -> None:
    """An outside index off the end of the names is a caller bug."""
    with pytest.raises(ValueError, match="not a valid index"):
        bio_from_runs([0], ["A", "O"], outside=9)


def test_every_source_exposes_the_interface() -> None:
    """`build_source` calls exactly these two functions, so every adapter must have them."""
    for name, source in SOURCES.items():
        assert callable(source.describe), name
        assert callable(source.load), name
        assert source.SPLITS, name


def test_every_source_declares_labels_matching_its_task() -> None:
    """
    A scorable task needs a label vocabulary, and a pretraining corpus must not claim one

    Pretraining corpora are exempt, as MLM predicts tokens rather than labels.
    Asserting both directions catches a corpus mislabeled as a task before the Trainer does.
    """
    for name, source in SOURCES.items():
        meta = source.describe()
        assert meta.source == name
        assert meta.task in {"token", "pretrain"}
        if meta.task == "pretrain":
            assert not meta.label_names, name
        else:
            assert meta.label_names, name


def test_groups_only_name_registered_sources() -> None:
    """A typo in a group would fail at build time, so catch it here instead."""
    for group, names in GROUPS.items():
        for name in names:
            assert name in SOURCES, f"group {group} names unknown source {name}"


def test_resolve_names_prefers_explicit_names() -> None:
    """Explicit names win over the group, and an unknown name raises."""
    assert resolve_names(["funsd"], "downstream") == ["funsd"]
    assert resolve_names(None, "pretrain") == GROUPS["pretrain"]
    assert resolve_names(None, None) == GROUPS["downstream"]
    with pytest.raises(KeyError, match="Unknown source"):
        resolve_names(["nope"], None)
    with pytest.raises(KeyError, match="Unknown group"):
        resolve_names(None, "nope")


def test_cord_labels_bio_per_line() -> None:
    """
    The first word of a CORD line starts an entity and the rest continue it

    Field names come from the line, so this is the only place CORD's B and I get decided.
    """
    lines = [
        {
            "category": "menu.nm",
            "words": [
                {"text": "Nasi", "quad": _quad(10, 20, 50, 30)},
                {"text": "Campur", "quad": _quad(60, 20, 120, 30)},
            ],
        }
    ]
    document = cord._build("cord-test-0", lines, width=200, height=100)
    assert document is not None
    names = cord.LABEL_NAMES
    assert document.word_labels is not None
    assert [names[label] for label in document.word_labels] == ["B-menu.nm", "I-menu.nm"]
    assert document.words == ["Nasi", "Campur"]
    # x spans 10..50 of a 200px page, so 50..250 on the 1000 grid
    assert document.bboxes[0] == [50, 200, 250, 300]


def test_cord_rejects_unknown_category() -> None:
    """
    An unlisted category must raise rather than silently becoming O

    The released data writes `menu.sub.nm` dotted where the published list uses an underscore.
    Falling through to O would erase that field.
    """
    lines = [{"category": "menu.invented", "words": [{"text": "x", "quad": _quad(0, 0, 1, 1)}]}]
    with pytest.raises(ValueError, match="unknown CORD category"):
        cord._build("cord-test-1", lines, width=10, height=10)


def test_cord_skips_blank_words() -> None:
    """Whitespace-only OCR text contributes no word, and no box either."""
    lines = [
        {
            "category": "menu.nm",
            "words": [
                {"text": "  ", "quad": _quad(0, 0, 5, 5)},
                {"text": "Nasi", "quad": _quad(10, 10, 20, 20)},
            ],
        }
    ]
    document = cord._build("cord-test-2", lines, width=100, height=100)
    assert document is not None
    assert document.words == ["Nasi"]


def test_cord_returns_none_for_a_wordless_receipt() -> None:
    """A receipt whose lines hold nothing usable is dropped, not written as an empty row."""
    assert cord._build("cord-test-3", [], width=100, height=100) is None


def test_sroie_label_scheme_is_the_recovered_one() -> None:
    """
    SROIE's source integers are pinned to the mapping recovered by inspection

    0 is COMPANY and 4 is outside, the reverse of the usual convention.
    A silent flip would make every SROIE number meaningless.
    """
    assert sroie.SOURCE_FIELDS == ["COMPANY", "DATE", "ADDRESS", "TOTAL", "O"]
    assert sroie.OUTSIDE_INDEX == 4
    assert sroie.LABEL_NAMES[0] == "O"
    assert sroie.LABEL_NAMES[1] == "B-COMPANY"
    assert len(sroie.LABEL_NAMES) == 9


def _quad(x0: int, y0: int, x1: int, y1: int) -> dict[str, int]:
    """Builds an axis-aligned CORD quad from a rectangle."""
    return {
        "x1": x0,
        "y1": y0,
        "x2": x1,
        "y2": y0,
        "x3": x1,
        "y3": y1,
        "x4": x0,
        "y4": y1,
    }


def test_idl_fresh_never_reads_a_shard_the_first_corpus_used(tmp_path) -> None:
    """
    Continued pretraining is only on unseen pages if the two shard sets are disjoint

    Names sort as strings, p10 before p2, so the test builds them the way the release names them.
    """
    for index in range(54):
        (tmp_path / f"imdb_pretrain_p{index}.npy").touch()
    everything = sorted(tmp_path.glob("imdb_*.npy"))

    fresh = idl.Fresh.shard_files(tmp_path)
    assert not set(fresh) & set(everything[:: idl.SHARD_STRIDE])
    assert len(fresh) == idl.Fresh.TRAIN_FILES + 2


def test_idl_docs_never_reads_a_shard_an_earlier_corpus_used(tmp_path) -> None:
    """The long phase is only on unseen pages if its shards miss both earlier corpora."""
    for index in range(54):
        (tmp_path / f"imdb_pretrain_p{index}.npy").touch()
    everything = sorted(tmp_path.glob("imdb_*.npy"))

    docs = idl.Docs.shard_files(tmp_path)
    assert not set(docs) & set(everything[:: idl.SHARD_STRIDE])
    assert not set(docs) & set(idl.Fresh.shard_files(tmp_path))
    assert len(docs) == idl.Docs.TRAIN_FILES + 2


def test_idl_documents_join_pages_in_order_each_on_its_own_grid(tmp_path, monkeypatch) -> None:
    """A document's pages are concatenated in page order and their boxes are not offset."""
    monkeypatch.setattr(idl, "_FINGERPRINTS", Fingerprints())

    def page(image_id: str, number: int, word: str) -> dict:
        boxes = np.tile([0.1, 0.2, 0.3, 0.25], (idl.MIN_WORDS, 1))
        return {
            "image_id": image_id,
            "ucsf_document_page": number,
            "ocr_tokens": [f"{word}{i}" for i in range(idl.MIN_WORDS)],
            "ocr_normalized_boxes": boxes,
        }

    entries = [{"version": 1}, page("a", 0, "first"), page("a", 1, "second"), page("b", 0, "only")]
    shard = tmp_path / "imdb_pretrain_p1.npy"
    np.save(shard, np.array(entries, dtype=object), allow_pickle=True)

    documents = list(idl.load_documents(shard))
    assert [document.doc_id for document in documents] == ["idl-a", "idl-b"]
    words = documents[0].words
    assert words[: idl.MIN_WORDS] == [f"first{i}" for i in range(idl.MIN_WORDS)]
    assert words[idl.MIN_WORDS :] == [f"second{i}" for i in range(idl.MIN_WORDS)]
    assert {tuple(box) for box in documents[0].bboxes} == {(100, 200, 300, 250)}


def write_realkie(root: Path, labels: list[dict]) -> None:
    """Writes a one-document RealKIE split: two pages, three words each, one CSV row."""
    import csv
    import gzip
    import json

    words = ["Pay", "Acme", "Corp", "Acme", "Ltd", "today"]
    offsets, cursor = [], 0
    for word in words:
        offsets.append((cursor, cursor + len(word)))
        cursor += len(word) + 1
    pages = [
        {
            "pages": [{"size": {"width": 2000, "height": 1000}}],
            "tokens": [
                {
                    "text": words[i],
                    "doc_offset": {"start": offsets[i][0], "end": offsets[i][1]},
                    "position": {
                        "left": 200 * (i % 3),
                        "top": 100,
                        "right": 200 * (i % 3) + 100,
                        "bottom": 150,
                    },
                }
                for i in range(page * 3, page * 3 + 3)
            ],
        }
        for page in range(2)
    ]
    (root / "nda" / "ocr").mkdir(parents=True)
    with gzip.open(root / "nda" / "ocr" / "doc.json.gz", "wt") as handle:
        json.dump(pages, handle)
    with (root / "nda" / "val.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["document_path", "labels", "ocr"])
        writer.writeheader()
        writer.writerow(
            {
                "document_path": "nda/files/doc.pdf",
                "labels": json.dumps(labels),
                "ocr": "nda/ocr/doc.json.gz",
            }
        )


def test_realkie_spans_become_exact_entities_on_per_page_boxes(tmp_path, monkeypatch) -> None:
    """
    Two adjacent spans of one label stay two entities, and page two is not stacked below page one

    SROIE's mirror merged adjacent entities, which character spans make avoidable here.
    """
    monkeypatch.setattr(realkie, "ROOT", tmp_path)
    write_realkie(
        tmp_path,
        [
            {"label": "Party", "start": 4, "end": 13, "text": "Acme Corp"},
            {"label": "Party", "start": 14, "end": 22, "text": "Acme Ltd"},
        ],
    )
    document = next(realkie.NDA.load("validation"))
    names = realkie.NDA.describe().label_names
    tags = [names[tag] for tag in document.word_labels or []]
    assert tags == ["O", "B-party", "I-party", "B-party", "I-party", "O"]
    assert document.bboxes[0] == document.bboxes[3] == [0, 100, 50, 150]


def test_realkie_unknown_label_fails_the_build(tmp_path, monkeypatch) -> None:
    """A label outside the shipped set must stop the build, not become the outside class."""
    monkeypatch.setattr(realkie, "ROOT", tmp_path)
    write_realkie(tmp_path, [{"label": "Signatory", "start": 0, "end": 3, "text": "Pay"}])
    with pytest.raises(ValueError, match="unknown label"):
        next(realkie.NDA.load("validation"))


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

    Joined, the citation "1011-1019." would read as the Bates-length number "10111019".
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

    CORD and SROIE are independent of IIT-CDIP, so a corpus with no overlap passes through whole.
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
