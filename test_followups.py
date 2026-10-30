
from types import SimpleNamespace

import rag_pipeline


def test_rewrite_question_without_history():
    """A first question should not require rewriting."""
    question = "What was ONGC's revenue?"

    result = rag_pipeline.rewrite_follow_up_question(
        question,
        [],
    )

    assert result == question


def test_rewrite_follow_up_question(monkeypatch):
    """A follow-up should be rewritten using conversation history."""

    def mock_generate_content(**kwargs):
        assert "What about the previous year?" in kwargs["contents"]
        assert "ONGC's revenue" in kwargs["contents"]

        return SimpleNamespace(
            text="What was ONGC's revenue in the previous year?"
        )

    monkeypatch.setattr(
        rag_pipeline.client.models,
        "generate_content",
        mock_generate_content,
    )

    history = [
        {
            "question": "What was ONGC's revenue?",
            "answer": "The revenue was X."
        }
    ]

    result = rag_pipeline.rewrite_follow_up_question(
        "What about the previous year?",
        history,
    )

    assert result == "What was ONGC's revenue in the previous year?"


def test_rewrite_falls_back_when_gemini_fails(monkeypatch):
    """If rewriting fails, the original question should be retained."""

    def mock_generate_content(**kwargs):
        raise RuntimeError("Gemini unavailable")

    monkeypatch.setattr(
        rag_pipeline.client.models,
        "generate_content",
        mock_generate_content,
    )

    question = "What about the previous year?"

    result = rag_pipeline.rewrite_follow_up_question(
        question,
        [{"question": "What was the revenue?", "answer": "X"}],
    )

    assert result == question


def test_answer_question_uses_rewritten_query_and_history(monkeypatch):
    """Retrieval uses the rewritten query; generation gets original context."""

    history = [
        {
            "question": "What was ONGC's revenue?",
            "answer": "The revenue was X."
        }
    ]

    retrieved = [
        {
            "page_number": 20,
            "text": "Revenue details",
            "score": 0.85,
        }
    ]

    expected_result = {
        "answer": "The previous year's revenue was Y.",
        "sources": retrieved,
    }

    monkeypatch.setattr(
        rag_pipeline,
        "rewrite_follow_up_question",
        lambda question, chat_history: "Previous year's ONGC revenue",
    )

    def mock_retrieve(question, chunks, model, index):
        assert question == "Previous year's ONGC revenue"
        return retrieved

    def mock_generate(question, passages, chat_history=None):
        assert question == "What about the previous year?"
        assert passages == retrieved
        assert chat_history == history
        return expected_result

    monkeypatch.setattr(
        rag_pipeline,
        "retrieve_passages",
        mock_retrieve,
    )
    monkeypatch.setattr(
        rag_pipeline,
        "generate_answer",
        mock_generate,
    )

    result = rag_pipeline.answer_question(
        "What about the previous year?",
        [],
        None,
        None,
        chat_history=history,
    )

    assert result == expected_result
