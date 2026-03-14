"""Utilities for processing streaming LLM responses."""

from enum import Enum, auto


class _State(Enum):
    SCANNING = auto()  # scanning JSON, looking for "text" key at depth 1
    AFTER_TEXT_KEY = auto()  # found "text" key, waiting for ':'
    AFTER_COLON = auto()  # found ':', waiting for opening '"'
    IN_VALUE = auto()  # inside the text string value — emit chars
    DONE = auto()  # extraction complete


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


class JsonTextExtractor:
    """
    State machine that extracts the top-level ``text`` field value from a
    streaming JSON response, emitting its characters as they arrive.

    Handles both field orderings::

        {"text": "Hello world", "sgr": {...}}
        {"sgr": {...}, "text": "Hello world"}

    Usage::

        extractor = JsonTextExtractor()
        async for token in stream:
            chunk = extractor.feed(token)
            if chunk:
                await on_stream(chunk)
    """

    def __init__(self) -> None:
        self._state = _State.SCANNING
        self._depth = 0  # {} brace nesting depth
        self._in_string = False  # currently inside a JSON string (not IN_VALUE)
        self._esc = False  # next char is escaped
        self._char_buf: list[str] = []  # accumulate key chars at depth 1
        self._next_is_key = True  # upcoming string at depth 1 is a key, not value

    @property
    def is_done(self) -> bool:
        """True once the ``text`` field value has been fully extracted."""
        return self._state == _State.DONE

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

    def _step(self, ch: str) -> str:  # noqa: PLR0912
        if self._state == _State.DONE:
            return ""

        # ── Handle pending escape ──────────────────────────────────────────────
        if self._esc:
            self._esc = False
            if self._state == _State.IN_VALUE:
                return _ESCAPE_MAP.get(ch, ch)
            if self._in_string and self._depth == 1 and self._next_is_key:
                self._char_buf.append(ch)
            return ""

        # ── Escape character ───────────────────────────────────────────────────
        if ch == "\\" and (self._in_string or self._state == _State.IN_VALUE):
            self._esc = True
            return ""

        # ── Inside text value — emit char or close ─────────────────────────────
        if self._state == _State.IN_VALUE:
            if ch == '"':
                self._state = _State.DONE
                return ""
            return ch

        # ── Inside a regular JSON string ───────────────────────────────────────
        if self._in_string:
            if ch == '"':
                self._in_string = False
                if self._depth == 1 and self._next_is_key:
                    key = "".join(self._char_buf)
                    self._char_buf.clear()
                    if key == "text" and self._state == _State.SCANNING:
                        self._state = _State.AFTER_TEXT_KEY
                    self._next_is_key = False
            elif self._depth == 1 and self._next_is_key:
                self._char_buf.append(ch)
            return ""

        # ── Outside strings ────────────────────────────────────────────────────
        if ch == '"':
            if self._state == _State.AFTER_COLON:
                self._state = _State.IN_VALUE
            else:
                self._in_string = True
                self._char_buf.clear()
            return ""

        if ch == "{":
            self._depth += 1
            if self._depth == 1:
                self._next_is_key = True
            return ""

        if ch == "}":
            self._depth -= 1
            return ""

        if ch == ":":
            if self._state == _State.AFTER_TEXT_KEY:
                self._state = _State.AFTER_COLON
            elif self._depth == 1:
                self._next_is_key = False
            return ""

        if ch == "," and self._depth == 1:
            self._next_is_key = True
            return ""

        return ""
