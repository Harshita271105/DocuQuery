
from unittest.mock import patch

from rag_pipeline import (
    retrieve_passages,
    generate_answer,
    answer_question,
)


def test_retrieve_passages_returns_top_results():
    """Retrieval should return passages with page numbers and scores."""

    chunks = [
        {"page_number": 1, "text": "ONGC produces crude oil."},
        {"page_number": 2, "text": "ONGC operates offshore fields."},
    ]

    class FakeModel:
        def encode(self, texts, **kwargs):
            import numpy as np
            return np.array([[1.0, 0.0]], dtype="float32")

    class FakeIndex:
        ntotal = 2

        def search(self, query_embedding, k):
            import numpy as np
            return (
                np.array([[0.9, 0.7]], dtype="float32"),
                np.array([[0, 1]], dtype="int64"),
            )

    results = retrieve_passages(
        "What does ONGC produce?",
        chunks,
        FakeModel(),
        FakeIndex(),
    )

    assert len(results) == 2
    assert results[0]["page_number"] == 1
    assert abs(results[0]["score"] - 0.9) < 1e-5
    assert "crude oil" in results[0]["text"]


def test_retrieve_passages_empty_question():
    """An empty question should not trigger retrieval."""

    results = retrieve_passages(
        "   ",
        [],
        None,
        None,
    )

    assert results == []


def test_generate_answer_without_passages():
    """No retrieved passages should produce a fallback answer."""

    result = generate_answer(
        "What is ONGC's production?",
        [],
    )

    assert "couldn't find relevant passages" in result["answer"]
    assert result["sources"] == []


def test_answer_question_retrieves_only_once():
    """The pipeline should retrieve once and pass those results to generation."""

    passages = [
        {
            "page_number": 5,
            "text": "Sample document passage.",
            "score": 0.85,
        }
    ]

    expected_result = {
        "answer": "Sample answer. [Page 5]",
        "sources": passages,
    }

    with (
        patch("rag_pipeline.retrieve_passages", return_value=passages) as mock_retrieve,
        patch("rag_pipeline.generate_answer", return_value=expected_result) as mock_generate,
    ):
        result = answer_question(
            "Sample question?",
            [{"page_number": 5, "text": "Sample document passage."}],
            object(),
            object(),
        )

    mock_retrieve.assert_called_once()
    mock_generate.assert_called_once_with("Sample question?", passages)
    assert result == expected_result


def test_retrieve_passages_filters_low_similarity():
    """Passages below the configured threshold should be excluded."""

    import numpy as np
    from rag_pipeline import MIN_SIMILARITY

    chunks = [
        {"page_number": 1, "text": "Relevant report passage."},
        {"page_number": 2, "text": "Irrelevant report passage."},
    ]

    class FakeModel:
        def encode(self, texts, **kwargs):
            return np.array([[1.0, 0.0]], dtype="float32")

    class FakeIndex:
        ntotal = 2

        def search(self, query_embedding, k):
            return (
                np.array([[0.75, MIN_SIMILARITY - 0.10]], dtype="float32"),
                np.array([[0, 1]], dtype="int64"),
            )

    results = retrieve_passages(
        "Question about the report",
        chunks,
        FakeModel(),
        FakeIndex(),
    )

    assert len(results) == 1
    assert results[0]["page_number"] == 1
    assert results[0]["score"] >= MIN_SIMILARITY


def test_retrieve_passages_returns_empty_when_all_scores_are_low():
    """No passages should remain if every score is below the threshold."""

    import numpy as np
    from rag_pipeline import MIN_SIMILARITY

    chunks = [
        {"page_number": 1, "text": "Unrelated passage one."},
        {"page_number": 2, "text": "Unrelated passage two."},
    ]

    class FakeModel:
        def encode(self, texts, **kwargs):
            return np.array([[1.0, 0.0]], dtype="float32")

    class FakeIndex:
        ntotal = 2

        def search(self, query_embedding, k):
            return (
                np.array(
                    [[MIN_SIMILARITY - 0.10, MIN_SIMILARITY - 0.15]],
                    dtype="float32",
                ),
                np.array([[0, 1]], dtype="int64"),
            )

    results = retrieve_passages(
        "Unrelated question",
        chunks,
        FakeModel(),
        FakeIndex(),
    )

    assert results == []
