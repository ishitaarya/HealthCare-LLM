from data.deduplication.deduplicate import (
    deduplicate_documents,
    deduplicate_records,
    normalize_for_dedup,
    text_hash,
)


def test_normalization_ignores_formatting_and_case():
    assert normalize_for_dedup("  Patient\n HEALTH  ") == "patient health"


def test_text_hash_is_stable_for_equivalent_text():
    assert text_hash("Medical report") == text_hash(" medical\nreport ")


def test_deduplicate_documents_keeps_first_occurrence():
    documents = ["Patient has fever.", "Patient has fever.", "Different report."]
    result = deduplicate_documents(documents)

    assert result.unique_documents == [documents[0], documents[2]]
    assert result.duplicate_indices == [1]
    assert result.duplicate_of == {1: 0}


def test_deduplicate_records_preserves_metadata():
    records = [
        {"text": "Patient has fever.", "source": "a"},
        {"text": " patient has FEVER ", "source": "b"},
    ]
    unique, duplicates = deduplicate_records(records)

    assert unique == [records[0]]
    assert duplicates[0]["source"] == "b"
    assert duplicates[0]["duplicate_of_index"] == 0


def test_empty_documents_are_deterministically_deduplicated():
    result = deduplicate_documents(["", "   ", "clinical report"])

    assert result.unique_documents == ["", "clinical report"]
    assert result.duplicate_indices == [1]
