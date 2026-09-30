import pytest

from app.data import load_tickets
from app.llm import LLMClient
from app.nl_query import NLQueryEngine, UnsafeExpression, validate_expression


def test_validator_blocks_import_and_dunder():
    for expression in [
        "__import__('os').system('whoami')",
        "df.__class__",
        "open('secret')",
    ]:
        with pytest.raises((UnsafeExpression, SyntaxError)):
            validate_expression(expression)


def test_mock_query_returns_correct_open_count():
    engine = NLQueryEngine(load_tickets("data/support_tickets.csv"), LLMClient("mock"))
    result = engine.answer("How many tickets are currently open?")
    assert result["raw_result"] == {"count": 111}


def test_mock_query_handles_filters_aggregates_and_ranking():
    engine = NLQueryEngine(load_tickets("data/support_tickets.csv"), LLMClient("mock"))

    assert engine.answer("How many tickets are in the Billing category?")["raw_result"] == {
        "count": 159
    }
    assert engine.answer("What is the average response time?")["raw_result"] == {
        "average": 2.6206
    }
    ranking = engine.answer("Which agent resolved the most tickets this month?")["raw_result"]
    assert len(ranking) == 1
    assert next(iter(ranking.values())) > 0


def test_anomaly_intent_bypasses_llm_code_generation():
    engine = NLQueryEngine(load_tickets("data/support_tickets.csv"), LLMClient("mock"))
    result = engine.answer("Are there any anomalies this week?")
    assert result["generated_code"] is None
