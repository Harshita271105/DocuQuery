import os
from types import SimpleNamespace
from unittest.mock import Mock

# Use a placeholder API key during tests.
os.environ.setdefault("GEMINI_API_KEY", "test-key")

import rag_pipeline


def test_rewrite_returns_original_without_history(monkeypatch):

    generate = Mock()

    monkeypatch.setattr(
        rag_pipeline.client.models,
        "generate_content",
        generate,
    )

    question = "What about its net profit?"

    assert (
        rag_pipeline.rewrite_follow_up_question(question, [])
        == question
    )

    generate.assert_not_called()


def test_rewrite_uses_history_and_returns_standalone_question(monkeypatch):

    generate = Mock(
        return_value=SimpleNamespace(
            text="What was ONGC's net profit?"
        )
    )

    monkeypatch.setattr(
        rag_pipeline.client.models,
        "generate_content",
        generate,
    )

    history = [
        {
            "question": "What was ONGC's total revenue?",
            "answer": "Revenue was ...",
        }
    ]

    rewritten = rag_pipeline.rewrite_follow_up_question(
        "What about its net profit?",
        history,
    )

    assert rewritten == "What was ONGC's net profit?"

    prompt = generate.call_args.kwargs["contents"]

    assert "ONGC's total revenue" in prompt
    assert "What about its net profit?" in prompt


def test_rewrite_falls_back_when_gemini_fails(monkeypatch):

    generate = Mock(
        side_effect=RuntimeError("API unavailable")
    )

    monkeypatch.setattr(
        rag_pipeline.client.models,
        "generate_content",
        generate,
    )

    question = "What about its net profit?"

    history = [
        {
            "question": "What was ONGC's revenue?",
            "answer": "...",
        }
    ]

    assert (
        rag_pipeline.rewrite_follow_up_question(
            question,
            history,
        )
        == question
    )


def test_rewrite_falls_back_when_response_is_empty(monkeypatch):

    generate = Mock(
        return_value=SimpleNamespace(text="  ")
    )

    monkeypatch.setattr(
        rag_pipeline.client.models,
        "generate_content",
        generate,
    )

    question = "What about its net profit?"

    history = [
        {
            "question": "What was ONGC's revenue?",
            "answer": "...",
        }
    ]

    assert (
        rag_pipeline.rewrite_follow_up_question(
            question,
            history,
        )
        == question
    )


def test_answer_question_retrieves_rewritten_query_but_answers_original(
    monkeypatch,
):

    history = [
        {
            "question": "What was ONGC's total revenue?",
            "answer": "Revenue was ...",
        }
    ]

    passages = [
        {
            "page_number": 12,
            "text": "Net profit details",
            "score": 0.82,
        }
    ]

    rewrite = Mock(
        return_value="What was ONGC's net profit?"
    )

    retrieve = Mock(return_value=passages)

    generate = Mock(
        return_value={
            "answer": "Net profit answer",
            "sources": passages,
        }
    )

    monkeypatch.setattr(
        rag_pipeline,
        "rewrite_follow_up_question",
        rewrite,
    )

    monkeypatch.setattr(
        rag_pipeline,
        "retrieve_passages",
        retrieve,
    )

    monkeypatch.setattr(
        rag_pipeline,
        "generate_answer",
        generate,
    )

    result = rag_pipeline.answer_question(
        "What about its net profit?",
        [],
        None,
        None,
        history,
    )

    rewrite.assert_called_once_with(
        "What about its net profit?",
        history,
    )

    retrieve.assert_called_once_with(
        "What was ONGC's net profit?",
        [],
        None,
        None,
    )

    generate.assert_called_once_with(
        "What about its net profit?",
        passages,
        chat_history=history,
    )

    assert result["answer"] == "Net profit answer"


def test_history_context_uses_only_last_four_turns():

    history = [
        {
            "question": f"Question {i}",
            "answer": f"Answer {i}",
        }
        for i in range(6)
    ]

    formatted = rag_pipeline._format_history(history)

    assert "Question 0" not in formatted
    assert "Question 1" not in formatted

    assert "Question 2" in formatted
    assert "Question 5" in formatted