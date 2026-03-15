"""Unit tests for app/utils/stream_utils.py — JsonTextExtractor and JsonReasoningExtractor."""

import pytest
from app.utils.stream_utils import JsonReasoningExtractor, JsonTextExtractor, JsonUIIntroTextExtractor


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


# ===========================================================================
# JsonLevel2TextExtractor tests
# ===========================================================================


def _extract_l2(json_str: str, chunk_size: int = 1) -> str:
    """Feed json_str to a fresh JsonLevel2TextExtractor chunk_size chars at a time."""
    from app.utils.stream_utils import JsonLevel2TextExtractor

    extractor = JsonLevel2TextExtractor()
    result = []
    for i in range(0, len(json_str), chunk_size):
        chunk = json_str[i : i + chunk_size]
        result.append(extractor.feed(chunk))
    return "".join(result)


def _l2_json(text: str, text_type: str = "markdown") -> str:
    """Build a minimal Level2Response JSON string."""
    return (
        f'{{"sgr": {{"reasoning": "r", "ui_reasoning": "u", "fact_checks": [], "orchestration_summary": null}}, '
        f'"level2_answer": {{"text": {{"type": "{text_type}", "text": "{text}"}}, '
        f'"quick_action_buttons": {{"buttons": []}}}}}}'
    )


# ---------------------------------------------------------------------------
# Basic extraction
# ---------------------------------------------------------------------------


def test_l2_basic_extraction():
    assert _extract_l2(_l2_json("hello world")) == "hello world"


def test_l2_sgr_first():
    """sgr comes before level2_answer (typical ordering)."""
    assert _extract_l2(_l2_json("sgr is first")) == "sgr is first"


def test_l2_level2_first():
    """level2_answer before sgr."""
    json_str = (
        '{"level2_answer": {"text": {"type": "plain", "text": "content first"}, '
        '"quick_action_buttons": {"buttons": []}}, "sgr": {"reasoning": "r"}}'
    )
    assert _extract_l2(json_str) == "content first"


def test_l2_text_key_after_type():
    """type field comes before text inside TextAnswer — type value must be skipped."""
    json_str = (
        '{"sgr": {}, "level2_answer": {"text": {"type": "markdown", "text": "actual"}, '
        '"quick_action_buttons": {}}}'
    )
    assert _extract_l2(json_str) == "actual"


def test_l2_quick_actions_before_text():
    """quick_action_buttons field appears before text in level2_answer."""
    json_str = (
        '{"sgr": {}, "level2_answer": {"quick_action_buttons": {"buttons": []}, '
        '"text": {"type": "plain", "text": "after buttons"}}}'
    )
    assert _extract_l2(json_str) == "after buttons"


def test_l2_empty_text():
    assert _extract_l2(_l2_json("")) == ""


def test_l2_spaces_and_punctuation():
    assert _extract_l2(_l2_json("Hi! How are you?")) == "Hi! How are you?"


# ---------------------------------------------------------------------------
# Escape sequences
# ---------------------------------------------------------------------------


def test_l2_escaped_newline():
    json_str = _l2_json(r"line1\nline2")
    assert _extract_l2(json_str) == "line1\nline2"


def test_l2_escaped_quote():
    json_str = '{"sgr": {}, "level2_answer": {"text": {"type": "plain", "text": "say \\"hi\\""}, "quick_action_buttons": {}}}'
    assert _extract_l2(json_str) == 'say "hi"'


def test_l2_escaped_backslash():
    json_str = '{"sgr": {}, "level2_answer": {"text": {"type": "plain", "text": "path\\\\to\\\\file"}, "quick_action_buttons": {}}}'
    assert _extract_l2(json_str) == "path\\to\\file"


# ---------------------------------------------------------------------------
# Chunk size resilience
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 13, 50, 1000])
def test_l2_various_chunk_sizes(chunk_size: int):
    json_str = _l2_json("streaming level2 works")
    assert _extract_l2(json_str, chunk_size) == "streaming level2 works"


# ---------------------------------------------------------------------------
# Nested keys inside sgr / quick_action_buttons must not leak
# ---------------------------------------------------------------------------


def test_l2_text_key_inside_sgr_ignored():
    """A 'text' key inside sgr should not be extracted."""
    json_str = (
        '{"sgr": {"reasoning": "r", "text": "wrong"}, '
        '"level2_answer": {"text": {"type": "plain", "text": "correct"}, "quick_action_buttons": {}}}'
    )
    assert _extract_l2(json_str) == "correct"


def test_l2_text_key_inside_buttons_ignored():
    """A 'text' key inside quick_action_buttons should not be extracted."""
    json_str = (
        '{"sgr": {}, "level2_answer": {"text": {"type": "plain", "text": "right"}, '
        '"quick_action_buttons": {"buttons": [{"text": "Click me"}]}}}'
    )
    assert _extract_l2(json_str) == "right"


# ---------------------------------------------------------------------------
# is_done property
# ---------------------------------------------------------------------------


def test_l2_is_done_after_extraction():
    from app.utils.stream_utils import JsonLevel2TextExtractor

    extractor = JsonLevel2TextExtractor()
    extractor.feed(_l2_json("done"))
    assert extractor.is_done


def test_l2_is_done_false_before_level2():
    from app.utils.stream_utils import JsonLevel2TextExtractor

    extractor = JsonLevel2TextExtractor()
    extractor.feed('{"sgr": {')
    assert not extractor.is_done


# ---------------------------------------------------------------------------
# No level2_answer field
# ---------------------------------------------------------------------------


def test_l2_no_level2_field_returns_empty():
    json_str = '{"sgr": {"reasoning": "nothing here"}, "other": "value"}'
    assert _extract_l2(json_str) == ""


# ===========================================================================
# JsonUIIntroTextExtractor tests
# ===========================================================================


def _extract_ui(json_str: str, chunk_size: int = 1) -> str:
    """Feed json_str to a fresh JsonUIIntroTextExtractor chunk_size chars at a time."""
    extractor = JsonUIIntroTextExtractor()
    result = []
    for i in range(0, len(json_str), chunk_size):
        chunk = json_str[i : i + chunk_size]
        result.append(extractor.feed(chunk))
    return "".join(result)


def _ui_json(intro_text: str | None, text_type: str = "markdown") -> str:
    """Build a minimal UIResponse JSON string."""
    if intro_text is None:
        intro_val = "null"
    else:
        intro_val = f'{{"type": "{text_type}", "text": "{intro_text}"}}'
    return (
        f'{{"sgr": {{"reasoning": "r", "ui_reasoning": "u", "fact_checks": [], "orchestration_summary": null}}, '
        f'"ui_answer": {{"intro_text": {intro_val}, "items": [], "quick_action_buttons": {{"buttons": []}}}}}}'
    )


# ---------------------------------------------------------------------------
# Basic extraction
# ---------------------------------------------------------------------------


def test_ui_basic_extraction():
    assert _extract_ui(_ui_json("hello world")) == "hello world"


def test_ui_sgr_first():
    """sgr comes before ui_answer (typical ordering)."""
    assert _extract_ui(_ui_json("sgr is first")) == "sgr is first"


def test_ui_answer_first():
    """ui_answer before sgr."""
    json_str = (
        '{"ui_answer": {"intro_text": {"type": "plain", "text": "content first"}, '
        '"items": [], "quick_action_buttons": {}}, "sgr": {"reasoning": "r"}}'
    )
    assert _extract_ui(json_str) == "content first"


def test_ui_text_key_after_type():
    """type field comes before text inside intro_text — type value must be skipped."""
    json_str = (
        '{"sgr": {}, "ui_answer": {"intro_text": {"type": "markdown", "text": "actual"}, '
        '"items": [], "quick_action_buttons": {}}}'
    )
    assert _extract_ui(json_str) == "actual"


def test_ui_items_before_intro_text():
    """items field appears before intro_text in ui_answer."""
    json_str = (
        '{"sgr": {}, "ui_answer": {"items": [], '
        '"intro_text": {"type": "plain", "text": "after items"}, "quick_action_buttons": {}}}'
    )
    assert _extract_ui(json_str) == "after items"


def test_ui_empty_intro_text():
    assert _extract_ui(_ui_json("")) == ""


def test_ui_spaces_and_punctuation():
    assert _extract_ui(_ui_json("Hi! How are you?")) == "Hi! How are you?"


# ---------------------------------------------------------------------------
# intro_text: null edge case
# ---------------------------------------------------------------------------


def test_ui_intro_text_null_returns_empty():
    """When intro_text is null, no characters should be emitted."""
    assert _extract_ui(_ui_json(None)) == ""


def test_ui_intro_text_null_is_done():
    """Extractor should be done after consuming null."""
    extractor = JsonUIIntroTextExtractor()
    extractor.feed(_ui_json(None))
    assert extractor.is_done


# ---------------------------------------------------------------------------
# Escape sequences
# ---------------------------------------------------------------------------


def test_ui_escaped_newline():
    json_str = _ui_json(r"line1\nline2")
    assert _extract_ui(json_str) == "line1\nline2"


def test_ui_escaped_quote():
    json_str = '{"sgr": {}, "ui_answer": {"intro_text": {"type": "plain", "text": "say \\"hi\\""}, "items": [], "quick_action_buttons": {}}}'
    assert _extract_ui(json_str) == 'say "hi"'


def test_ui_escaped_backslash():
    json_str = '{"sgr": {}, "ui_answer": {"intro_text": {"type": "plain", "text": "path\\\\to\\\\file"}, "items": [], "quick_action_buttons": {}}}'
    assert _extract_ui(json_str) == "path\\to\\file"


# ---------------------------------------------------------------------------
# Chunk size resilience
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 13, 50, 1000])
def test_ui_various_chunk_sizes(chunk_size: int):
    json_str = _ui_json("streaming ui_answer works")
    assert _extract_ui(json_str, chunk_size) == "streaming ui_answer works"


# ---------------------------------------------------------------------------
# Nested keys must not leak
# ---------------------------------------------------------------------------


def test_ui_text_key_inside_sgr_ignored():
    """A 'text' key inside sgr should not be extracted."""
    json_str = (
        '{"sgr": {"reasoning": "r", "text": "wrong"}, '
        '"ui_answer": {"intro_text": {"type": "plain", "text": "correct"}, '
        '"items": [], "quick_action_buttons": {}}}'
    )
    assert _extract_ui(json_str) == "correct"


def test_ui_text_key_inside_items_ignored():
    """A 'text' key inside items should not be extracted."""
    json_str = (
        '{"sgr": {}, "ui_answer": {"intro_text": {"type": "plain", "text": "right"}, '
        '"items": [{"text": "wrong item text"}], "quick_action_buttons": {}}}'
    )
    assert _extract_ui(json_str) == "right"


# ---------------------------------------------------------------------------
# is_done property
# ---------------------------------------------------------------------------


def test_ui_is_done_after_extraction():
    extractor = JsonUIIntroTextExtractor()
    extractor.feed(_ui_json("done"))
    assert extractor.is_done


def test_ui_is_done_false_before_ui_answer():
    extractor = JsonUIIntroTextExtractor()
    extractor.feed('{"sgr": {')
    assert not extractor.is_done


# ---------------------------------------------------------------------------
# No intro_text field / no ui_answer field
# ---------------------------------------------------------------------------


def test_ui_no_intro_text_field_returns_empty():
    json_str = '{"sgr": {"reasoning": "r"}, "ui_answer": {"items": [], "quick_action_buttons": {}}}'
    assert _extract_ui(json_str) == ""


def test_ui_no_ui_answer_field_returns_empty():
    json_str = '{"sgr": {"reasoning": "nothing here"}, "other": "value"}'
    assert _extract_ui(json_str) == ""