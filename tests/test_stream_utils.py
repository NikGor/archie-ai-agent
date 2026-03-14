"""Unit tests for app/utils/stream_utils.py — JsonTextExtractor and JsonReasoningExtractor."""

import pytest
from app.utils.stream_utils import JsonReasoningExtractor, JsonTextExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract(json_str: str, chunk_size: int = 1) -> str:
    """Feed *json_str* to a fresh extractor chunk_size chars at a time."""
    extractor = JsonTextExtractor()
    result = []
    for i in range(0, len(json_str), chunk_size):
        chunk = json_str[i : i + chunk_size]
        result.append(extractor.feed(chunk))
    return "".join(result)


# ---------------------------------------------------------------------------
# Basic extraction
# ---------------------------------------------------------------------------


def test_text_first_field():
    """text field appears before sgr."""
    json_str = '{"text": "hello world", "sgr": {"reasoning": "ok", "ui_reasoning": "ok", "fact_checks": [], "orchestration_summary": null}}'
    assert _extract(json_str) == "hello world"


def test_text_second_field():
    """text field appears after sgr."""
    json_str = '{"sgr": {"reasoning": "ok"}, "text": "hello second"}'
    assert _extract(json_str) == "hello second"


def test_empty_text():
    json_str = '{"text": "", "sgr": {}}'
    assert _extract(json_str) == ""


def test_text_with_spaces_and_punctuation():
    json_str = '{"text": "Hi! How are you? Fine, thanks.", "sgr": {}}'
    assert _extract(json_str) == "Hi! How are you? Fine, thanks."


# ---------------------------------------------------------------------------
# Escape sequences
# ---------------------------------------------------------------------------


def test_escaped_newline_in_text():
    json_str = r'{"text": "line1\nline2", "sgr": {}}'
    assert _extract(json_str) == "line1\nline2"


def test_escaped_quote_in_text():
    json_str = r'{"text": "say \"hi\"", "sgr": {}}'
    assert _extract(json_str) == 'say "hi"'


def test_escaped_backslash_in_text():
    json_str = r'{"text": "path\\to\\file", "sgr": {}}'
    assert _extract(json_str) == "path\\to\\file"


# ---------------------------------------------------------------------------
# Chunk size resilience
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 13, 50, 1000])
def test_various_chunk_sizes(chunk_size: int):
    json_str = '{"text": "streaming works", "sgr": {"info": "test"}}'
    assert _extract(json_str, chunk_size) == "streaming works"


# ---------------------------------------------------------------------------
# Nested objects do NOT confuse the extractor
# ---------------------------------------------------------------------------


def test_text_key_inside_nested_object_ignored():
    """A 'text' key inside sgr should NOT be extracted."""
    json_str = '{"sgr": {"text": "inner", "reasoning": "r"}, "text": "outer"}'
    assert _extract(json_str) == "outer"


def test_deeply_nested_json():
    json_str = '{"sgr": {"a": {"b": {"text": "deep"}}, "text": "mid"}, "text": "top"}'
    assert _extract(json_str) == "top"


# ---------------------------------------------------------------------------
# is_done property
# ---------------------------------------------------------------------------


def test_is_done_after_extraction():
    extractor = JsonTextExtractor()
    extractor.feed('{"text": "done", "sgr": {}}')
    assert extractor.is_done


def test_is_done_false_before_text_field():
    extractor = JsonTextExtractor()
    extractor.feed('{"sgr": {"reasoning":')
    assert not extractor.is_done


# ---------------------------------------------------------------------------
# No text field — extractor stays in SCANNING, returns empty string
# ---------------------------------------------------------------------------


def test_no_text_field_returns_empty():
    json_str = '{"sgr": {"reasoning": "no text here"}, "other": "value"}'
    assert _extract(json_str) == ""


# ===========================================================================
# JsonReasoningExtractor tests
# ===========================================================================


def _extract_reasoning(json_str: str, chunk_size: int = 1) -> str:
    """Feed json_str to a fresh JsonReasoningExtractor chunk_size chars at a time."""
    extractor = JsonReasoningExtractor()
    result = []
    for i in range(0, len(json_str), chunk_size):
        chunk = json_str[i : i + chunk_size]
        result.append(extractor.feed(chunk))
    return "".join(result)


# ---------------------------------------------------------------------------
# Basic extraction
# ---------------------------------------------------------------------------


def test_reasoning_sgr_first():
    """sgr (with reasoning) comes before main content."""
    json_str = '{"sgr": {"reasoning": "step by step", "ui_reasoning": "use cards"}, "ui_answer": {}}'
    assert _extract_reasoning(json_str) == "step by step"


def test_reasoning_sgr_second():
    """sgr comes after main content."""
    json_str = '{"ui_answer": {"items": []}, "sgr": {"reasoning": "because", "ui_reasoning": "cards"}}'
    assert _extract_reasoning(json_str) == "because"


def test_reasoning_key_before_other_sgr_fields():
    """reasoning is first field inside sgr object."""
    json_str = '{"sgr": {"reasoning": "first!", "fact_checks": [], "ui_reasoning": "ok"}}'
    assert _extract_reasoning(json_str) == "first!"


def test_reasoning_key_after_other_sgr_fields():
    """reasoning is last field inside sgr object."""
    json_str = '{"sgr": {"fact_checks": [], "ui_reasoning": "ok", "reasoning": "last"}}'
    assert _extract_reasoning(json_str) == "last"


def test_reasoning_empty_string():
    json_str = '{"sgr": {"reasoning": "", "ui_reasoning": "x"}}'
    assert _extract_reasoning(json_str) == ""


def test_reasoning_with_escape_sequences():
    json_str = r'{"sgr": {"reasoning": "step 1\nstep 2", "ui_reasoning": "x"}}'
    assert _extract_reasoning(json_str) == "step 1\nstep 2"


def test_reasoning_with_escaped_quote():
    json_str = r'{"sgr": {"reasoning": "say \"hi\"", "ui_reasoning": "x"}}'
    assert _extract_reasoning(json_str) == 'say "hi"'


# ---------------------------------------------------------------------------
# Chunk size resilience
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 13, 50, 1000])
def test_reasoning_various_chunk_sizes(chunk_size: int):
    json_str = '{"sgr": {"reasoning": "reasoning text here", "ui_reasoning": "x"}, "ui_answer": {}}'
    assert _extract_reasoning(json_str, chunk_size) == "reasoning text here"


# ---------------------------------------------------------------------------
# Nested objects don't confuse the extractor
# ---------------------------------------------------------------------------


def test_reasoning_key_in_nested_does_not_leak():
    """reasoning key deep inside ui_answer should not be extracted."""
    json_str = (
        '{"sgr": {"reasoning": "correct", "ui_reasoning": "x"}, '
        '"ui_answer": {"items": [{"reasoning": "wrong"}]}}'
    )
    assert _extract_reasoning(json_str) == "correct"


# ---------------------------------------------------------------------------
# is_done property
# ---------------------------------------------------------------------------


def test_reasoning_is_done_after_extraction():
    extractor = JsonReasoningExtractor()
    extractor.feed('{"sgr": {"reasoning": "done", "ui_reasoning": "x"}}')
    assert extractor.is_done


def test_reasoning_is_done_false_before_sgr():
    extractor = JsonReasoningExtractor()
    extractor.feed('{"ui_answer": {')
    assert not extractor.is_done


# ---------------------------------------------------------------------------
# No reasoning field
# ---------------------------------------------------------------------------


def test_no_reasoning_returns_empty():
    json_str = '{"sgr": {"ui_reasoning": "only this"}, "ui_answer": {}}'
    assert _extract_reasoning(json_str) == ""
