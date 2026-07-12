"""Utilities for processing streaming LLM responses."""

from collections.abc import Sequence
from enum import Enum, auto


class _Stage(Enum):
    SCANNING = auto()  # scanning for the key of the current path segment
    FOUND_KEY = auto()  # found the key, waiting for ':'
    EXPECT_VALUE = auto()  # found ':', waiting for '"' / '{' / 'n' (null)
    IN_VALUE = auto()  # inside the leaf string value — emit chars
    IN_NULL = auto()  # consuming a "null" literal for an intermediate segment
    DONE = auto()  # extraction complete (value captured, or null encountered)


_ESCAPE_MAP: dict[str, str] = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "b": "\b",
    "f": "\f",
}

_NULL_TAIL = "ull"  # remaining chars after 'n' in "null"


class JsonPathExtractor:
    """
    State machine that extracts the string value at ``path`` from a streaming
    JSON response, emitting its characters as they arrive.

    ``path`` is a sequence of object keys, e.g. ``["sgr", "reasoning"]`` for
    ``{"sgr": {"reasoning": "..."}}``, or ``["text"]`` for a top-level
    ``{"text": "..."}``. Field ordering inside each object does not matter —
    the extractor scans depth-by-depth for the next key in the path,
    ignoring unrelated sibling fields (including same-named keys nested at
    the wrong depth).

    An intermediate segment's object value may be ``null`` instead of ``{``
    (e.g. ``"intro_text": null``); in that case extraction finishes with no
    characters emitted.

    This replaces four near-identical hand-rolled extractors
    (``JsonTextExtractor``, ``JsonReasoningExtractor``,
    ``JsonLevel2TextExtractor``, ``JsonUIIntroTextExtractor``) that only
    differed in their target key path (ARCHIE-154). Those names are kept as
    thin subclasses below for call-site and test-suite compatibility.

    Usage::

        extractor = JsonPathExtractor(["sgr", "reasoning"])
        async for token in stream:
            chunk = extractor.feed(token)
            if chunk:
                await on_stream_event("stream_reasoning", chunk)
    """

    def __init__(self, path: Sequence[str]) -> None:
        if not path:
            raise ValueError("path must not be empty")
        self._path = tuple(path)
        self._stage = _Stage.SCANNING
        self._depth = 0  # {} brace nesting depth
        self._seg = 0  # index into self._path of the segment being matched
        self._in_string = False  # currently inside a JSON string (not IN_VALUE)
        self._esc = False  # next char is escaped
        self._char_buf: list[str] = []  # accumulate key chars at the target depth
        self._next_is_key = True  # upcoming string at target depth is a key
        self._null_pos = 0  # position within "ull" when consuming null

    @property
    def is_done(self) -> bool:
        """True once the target path's value has been fully extracted (or
        skipped, if an intermediate segment was ``null``)."""
        return self._stage == _Stage.DONE

    def feed(self, chunk: str) -> str:
        """
        Process a chunk of JSON tokens.

        Returns any text characters to emit to the user.
        Returns an empty string if nothing to emit yet.
        """
        out: list[str] = []
        for ch in chunk:
            result = self._step(ch)
            if result:
                out.append(result)
        return "".join(out)

    def _step(self, ch: str) -> str:  # noqa: PLR0912, PLR0911
        if self._stage == _Stage.DONE:
            return ""

        # ── Consuming a "null" literal for an intermediate segment ────────────
        if self._stage == _Stage.IN_NULL:
            if ch == _NULL_TAIL[self._null_pos]:
                self._null_pos += 1
                if self._null_pos == len(_NULL_TAIL):
                    self._stage = _Stage.DONE
            return ""

        target_depth = self._seg + 1
        last_segment = self._seg == len(self._path) - 1

        # ── Handle pending escape ──────────────────────────────────────────────
        if self._esc:
            self._esc = False
            if self._stage == _Stage.IN_VALUE:
                return _ESCAPE_MAP.get(ch, ch)
            if (
                self._in_string
                and self._depth == target_depth
                and self._next_is_key
                and self._stage == _Stage.SCANNING
            ):
                self._char_buf.append(ch)
            return ""

        # ── Escape character ───────────────────────────────────────────────────
        if ch == "\\" and (self._in_string or self._stage == _Stage.IN_VALUE):
            self._esc = True
            return ""

        # ── Inside the leaf value — emit char or close ──────────────────────────
        if self._stage == _Stage.IN_VALUE:
            if ch == '"':
                self._stage = _Stage.DONE
                return ""
            return ch

        # ── Inside a regular JSON string ───────────────────────────────────────
        if self._in_string:
            if ch == '"':
                self._in_string = False
                if (
                    self._depth == target_depth
                    and self._next_is_key
                    and self._stage == _Stage.SCANNING
                ):
                    key = "".join(self._char_buf)
                    self._char_buf.clear()
                    if key == self._path[self._seg]:
                        self._stage = _Stage.FOUND_KEY
                    self._next_is_key = False
            elif (
                self._depth == target_depth
                and self._next_is_key
                and self._stage == _Stage.SCANNING
            ):
                self._char_buf.append(ch)
            return ""

        # ── Outside strings ────────────────────────────────────────────────────
        if ch == '"':
            if self._stage == _Stage.EXPECT_VALUE and last_segment:
                self._stage = _Stage.IN_VALUE
            else:
                self._in_string = True
                self._char_buf.clear()
            return ""

        if ch == "n" and self._stage == _Stage.EXPECT_VALUE and not last_segment:
            self._stage = _Stage.IN_NULL
            self._null_pos = 0
            return ""

        if ch == "{":
            self._depth += 1
            if self._depth == 1:
                if self._seg == 0 and self._stage == _Stage.SCANNING:
                    self._next_is_key = True
            elif (
                self._stage == _Stage.EXPECT_VALUE
                and not last_segment
                and self._depth == target_depth + 1
            ):
                self._seg += 1
                self._stage = _Stage.SCANNING
                self._next_is_key = True
            return ""

        if ch == "}":
            self._depth -= 1
            return ""

        if ch == ":":
            if self._stage == _Stage.FOUND_KEY:
                self._stage = _Stage.EXPECT_VALUE
            elif self._depth == target_depth and self._stage == _Stage.SCANNING:
                self._next_is_key = False
            return ""

        if ch == ",":
            if self._depth == target_depth and self._stage == _Stage.SCANNING:
                self._next_is_key = True
            return ""

        return ""


class JsonTextExtractor(JsonPathExtractor):
    """Extracts the top-level ``text`` field, e.g. ``{"text": "...", "sgr": {...}}``."""

    def __init__(self) -> None:
        super().__init__(["text"])


class JsonReasoningExtractor(JsonPathExtractor):
    """Extracts ``sgr.reasoning``, e.g. ``{"sgr": {"reasoning": "...", ...}}``."""

    def __init__(self) -> None:
        super().__init__(["sgr", "reasoning"])


class JsonLevel2TextExtractor(JsonPathExtractor):
    """Extracts ``level2_answer.text.text`` from a streaming Level2Response."""

    def __init__(self) -> None:
        super().__init__(["level2_answer", "text", "text"])


class JsonUIIntroTextExtractor(JsonPathExtractor):
    """Extracts ``ui_answer.intro_text.text`` from a streaming UIResponse.
    Handles ``intro_text: null`` gracefully — no characters are emitted."""

    def __init__(self) -> None:
        super().__init__(["ui_answer", "intro_text", "text"])
