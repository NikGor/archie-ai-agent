"""
Unit tests for llm_parser.parse_llm_response().

No real LLM calls — raw provider responses are faked via SimpleNamespace.
Tests verify the parser contract: given a provider-specific raw response,
return a ParsedLLMResponse with correct types and token counts.
"""

import json
from types import SimpleNamespace

import pytest

from app.models.output_models import PlainResponse, SGROutput
from app.utils.llm_parser import ParsedLLMResponse, parse_llm_response


# ---------------------------------------------------------------------------
# Helpers — minimal fake raw responses matching each provider's structure
# ---------------------------------------------------------------------------


def _sgr_dict() -> dict:
    return {
        "fact_checks": [],
        "ui_reasoning": "plain text response",
        "reasoning": "answered directly from knowledge",
    }


def _plain_response_obj() -> PlainResponse:
    return PlainResponse(
        text="Hello",
        sgr=SGROutput(**_sgr_dict()),
    )


def _fake_openai_response(parsed_obj=None) -> SimpleNamespace:
    if parsed_obj is None:
        parsed_obj = _plain_response_obj()
    return SimpleNamespace(
        id="resp_test",
        model="gpt-4.1",
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(parsed=parsed_obj)],
            )
        ],
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            input_tokens_details=SimpleNamespace(cached_tokens=10),
            output_tokens_details=SimpleNamespace(reasoning_tokens=0),
        ),
    )


def _fake_openrouter_response(
    content_dict: dict | None = None, usage=True
) -> SimpleNamespace:
    if content_dict is None:
        content_dict = {"text": "Hello", "sgr": _sgr_dict()}
    return SimpleNamespace(
        id="chatcmpl_test",
        model="some-model",
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=json.dumps(content_dict)))
        ],
        usage=(
            SimpleNamespace(
                prompt_tokens=80,
                completion_tokens=40,
                total_tokens=120,
                prompt_tokens_details=None,
                completion_tokens_details=None,
            )
            if usage
            else None
        ),
    )


# ---------------------------------------------------------------------------
# OpenAI parser
# ---------------------------------------------------------------------------


def test_parse_openai_returns_parsed_llm_response():
    raw = _fake_openai_response()
    result = parse_llm_response(raw, "openai", PlainResponse)
    assert isinstance(result, ParsedLLMResponse)


def test_parse_openai_parsed_content_type():
    raw = _fake_openai_response()
    result = parse_llm_response(raw, "openai", PlainResponse)
    assert isinstance(result.parsed_content, PlainResponse)


def test_parse_openai_token_counts():
    raw = _fake_openai_response()
    result = parse_llm_response(raw, "openai", PlainResponse)
    assert result.llm_trace.input_tokens == 100
    assert result.llm_trace.output_tokens == 50
    assert result.llm_trace.total_tokens == 150


def test_parse_openai_extracts_response_id():
    raw = _fake_openai_response()
    result = parse_llm_response(raw, "openai", PlainResponse)
    assert result.response_id == "resp_test"


# ---------------------------------------------------------------------------
# OpenRouter parser
# ---------------------------------------------------------------------------


def test_parse_openrouter_returns_parsed_llm_response():
    raw = _fake_openrouter_response()
    result = parse_llm_response(raw, "openrouter", PlainResponse)
    assert isinstance(result, ParsedLLMResponse)


def test_parse_openrouter_parsed_content_type():
    raw = _fake_openrouter_response()
    result = parse_llm_response(raw, "openrouter", PlainResponse)
    assert isinstance(result.parsed_content, PlainResponse)


def test_parse_openrouter_token_counts():
    raw = _fake_openrouter_response()
    result = parse_llm_response(raw, "openrouter", PlainResponse)
    assert result.llm_trace.input_tokens == 80
    assert result.llm_trace.output_tokens == 40
    assert result.llm_trace.total_tokens == 120


def test_parse_openrouter_missing_usage_returns_zeros():
    """Parser must not crash when usage is None — returns zeros gracefully."""
    raw = _fake_openrouter_response(usage=False)
    result = parse_llm_response(raw, "openrouter", PlainResponse)
    assert result.llm_trace.input_tokens == 0
    assert result.llm_trace.total_tokens == 0


# ---------------------------------------------------------------------------
# Unknown provider
# ---------------------------------------------------------------------------


def test_parse_unknown_provider_raises_value_error():
    raw = _fake_openai_response()
    with pytest.raises(ValueError, match="Unsupported provider"):
        parse_llm_response(raw, "nonexistent_provider", PlainResponse)
