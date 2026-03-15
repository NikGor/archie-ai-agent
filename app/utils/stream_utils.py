"""Utilities for processing streaming LLM responses."""

from enum import Enum, auto


class _L2State(Enum):
    SCANNING_D1 = auto()     # depth 1, looking for "level2_answer" key
    FOUND_L2_KEY = auto()    # found it, waiting for ':'
    EXPECT_L2_OBJ = auto()   # found ':', waiting for '{'
    IN_L2 = auto()           # depth 2, looking for "text" key (TextAnswer object)
    FOUND_TEXT_KEY = auto()  # found "text" at d2, waiting for ':'
    EXPECT_TEXT_OBJ = auto() # found ':', waiting for '{'
    IN_TEXT_OBJ = auto()     # depth 3, looking for "text" key (the string value)
    FOUND_STR_KEY = auto()   # found "text" at d3, waiting for ':'
    EXPECT_VALUE = auto()    # found ':', waiting for '"'
    IN_VALUE = auto()        # streaming chars
    DONE = auto()


class _State(Enum):
    SCANNING = auto()  # scanning JSON, looking for "text" key at depth 1
    AFTER_TEXT_KEY = auto()  # found "text" key, waiting for ':'
    AFTER_COLON = auto()  # found ':', waiting for opening '"'
    IN_VALUE = auto()  # inside the text string value — emit chars
    DONE = auto()  # extraction complete


class _RState(Enum):
    SCANNING_D1 = auto()  # depth 1, looking for "sgr" key
    FOUND_SGR_KEY = auto()  # found "sgr", waiting for ':'
    EXPECT_SGR_OBJ = auto()  # found ':', waiting for '{'
    IN_SGR = auto()  # depth 2 inside sgr, looking for "reasoning" key
    FOUND_R_KEY = auto()  # found "reasoning", waiting for ':'
    EXPECT_R_VALUE = auto()  # found ':', waiting for opening '"'
    IN_VALUE = auto()  # extracting reasoning chars
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


class JsonReasoningExtractor:
    """
    State machine that extracts ``sgr.reasoning`` from a streaming JSON response,
    emitting its characters as they arrive.

    Handles both sgr-first and sgr-second orderings. Works with any ordering
    of keys inside the sgr object::

        {"sgr": {"reasoning": "...", ...}, "ui_answer": {...}}
        {"ui_answer": {...}, "sgr": {"reasoning": "...", ...}}

    Usage::

        extractor = JsonReasoningExtractor()
        async for token in stream:
            chunk = extractor.feed(token)
            if chunk:
                await on_stream_event("stream_reasoning", chunk)
    """

    def __init__(self) -> None:
        self._state = _RState.SCANNING_D1
        self._depth = 0
        self._in_string = False
        self._esc = False
        self._char_buf: list[str] = []
        self._next_is_key_d1 = True  # next string at depth 1 is a key
        self._next_is_key_d2 = True  # next string at depth 2 is a key

    @property
    def is_done(self) -> bool:
        """True once ``sgr.reasoning`` has been fully extracted."""
        return self._state == _RState.DONE

    def feed(self, chunk: str) -> str:
        """Process a chunk, return any reasoning characters to emit."""
        out: list[str] = []
        for ch in chunk:
            result = self._step(ch)
            if result:
                out.append(result)
        return "".join(out)

    def _step(self, ch: str) -> str:  # noqa: PLR0912, PLR0911
        if self._state == _RState.DONE:
            return ""

        # ── Handle pending escape ──────────────────────────────────────────────
        if self._esc:
            self._esc = False
            if self._state == _RState.IN_VALUE:
                return _ESCAPE_MAP.get(ch, ch)
            if self._in_string:
                if (
                    self._depth == 1
                    and self._next_is_key_d1
                    or (
                        self._depth == 2
                        and self._next_is_key_d2
                        and self._state == _RState.IN_SGR
                    )
                ):
                    self._char_buf.append(ch)
            return ""

        # ── Escape character ───────────────────────────────────────────────────
        if ch == "\\" and (self._in_string or self._state == _RState.IN_VALUE):
            self._esc = True
            return ""

        # ── Inside reasoning value — emit char or close ────────────────────────
        if self._state == _RState.IN_VALUE:
            if ch == '"':
                self._state = _RState.DONE
                return ""
            return ch

        # ── Inside a regular JSON string ───────────────────────────────────────
        if self._in_string:
            if ch == '"':
                self._in_string = False
                if self._depth == 1 and self._next_is_key_d1:
                    key = "".join(self._char_buf)
                    self._char_buf.clear()
                    if key == "sgr" and self._state == _RState.SCANNING_D1:
                        self._state = _RState.FOUND_SGR_KEY
                    self._next_is_key_d1 = False
                elif (
                    self._depth == 2
                    and self._next_is_key_d2
                    and self._state == _RState.IN_SGR
                ):
                    key = "".join(self._char_buf)
                    self._char_buf.clear()
                    if key == "reasoning":
                        self._state = _RState.FOUND_R_KEY
                    self._next_is_key_d2 = False
            elif (
                self._depth == 1
                and self._next_is_key_d1
                or (
                    self._depth == 2
                    and self._next_is_key_d2
                    and self._state == _RState.IN_SGR
                )
            ):
                self._char_buf.append(ch)
            return ""

        # ── Outside strings ────────────────────────────────────────────────────
        if ch == '"':
            if self._state == _RState.EXPECT_R_VALUE:
                self._state = _RState.IN_VALUE
            else:
                self._in_string = True
                self._char_buf.clear()
            return ""

        if ch == "{":
            self._depth += 1
            if self._depth == 1:
                self._next_is_key_d1 = True
            elif self._depth == 2 and self._state == _RState.EXPECT_SGR_OBJ:
                self._state = _RState.IN_SGR
                self._next_is_key_d2 = True
            return ""

        if ch == "}":
            self._depth -= 1
            return ""

        if ch == ":":
            if self._state == _RState.FOUND_SGR_KEY:
                self._state = _RState.EXPECT_SGR_OBJ
            elif self._state == _RState.FOUND_R_KEY:
                self._state = _RState.EXPECT_R_VALUE
            elif self._depth == 1:
                self._next_is_key_d1 = False
            elif self._depth == 2 and self._state == _RState.IN_SGR:
                self._next_is_key_d2 = False
            return ""

        if ch == ",":
            if self._depth == 1:
                self._next_is_key_d1 = True
            elif self._depth == 2 and self._state == _RState.IN_SGR:
                self._next_is_key_d2 = True
            return ""

        return ""


class JsonLevel2TextExtractor:
    """
    State machine that extracts ``level2_answer.text.text`` from a streaming
    JSON Level2Response, emitting its characters as they arrive.

    Input JSON structure::

        {"sgr": {...}, "level2_answer": {"text": {"type": "markdown", "text": "..."}, ...}}

    Usage::

        extractor = JsonLevel2TextExtractor()
        async for token in stream:
            chunk = extractor.feed(token)
            if chunk:
                await on_stream_event("stream_delta", chunk)
    """

    def __init__(self) -> None:
        self._state = _L2State.SCANNING_D1
        self._depth = 0
        self._in_string = False
        self._esc = False
        self._char_buf: list[str] = []
        self._next_is_key_d1 = True
        self._next_is_key_d2 = True
        self._next_is_key_d3 = True

    @property
    def is_done(self) -> bool:
        """True once ``level2_answer.text.text`` has been fully extracted."""
        return self._state == _L2State.DONE

    def feed(self, chunk: str) -> str:
        """Process a chunk, return any text characters to emit."""
        out: list[str] = []
        for ch in chunk:
            result = self._step(ch)
            if result:
                out.append(result)
        return "".join(out)

    def _step(self, ch: str) -> str:  # noqa: PLR0912, PLR0911
        if self._state == _L2State.DONE:
            return ""

        # ── Handle pending escape ──────────────────────────────────────────────
        if self._esc:
            self._esc = False
            if self._state == _L2State.IN_VALUE:
                return _ESCAPE_MAP.get(ch, ch)
            if self._in_string:
                if self._depth == 1 and self._next_is_key_d1:
                    self._char_buf.append(ch)
                elif self._depth == 2 and self._next_is_key_d2 and self._state == _L2State.IN_L2:
                    self._char_buf.append(ch)
                elif self._depth == 3 and self._next_is_key_d3 and self._state == _L2State.IN_TEXT_OBJ:
                    self._char_buf.append(ch)
            return ""

        # ── Escape character ───────────────────────────────────────────────────
        if ch == "\\" and (self._in_string or self._state == _L2State.IN_VALUE):
            self._esc = True
            return ""

        # ── Inside text value — emit char or close ─────────────────────────────
        if self._state == _L2State.IN_VALUE:
            if ch == '"':
                self._state = _L2State.DONE
                return ""
            return ch

        # ── Inside a regular JSON string ───────────────────────────────────────
        if self._in_string:
            if ch == '"':
                self._in_string = False
                if self._depth == 1 and self._next_is_key_d1:
                    key = "".join(self._char_buf)
                    self._char_buf.clear()
                    if key == "level2_answer" and self._state == _L2State.SCANNING_D1:
                        self._state = _L2State.FOUND_L2_KEY
                    self._next_is_key_d1 = False
                elif self._depth == 2 and self._next_is_key_d2 and self._state == _L2State.IN_L2:
                    key = "".join(self._char_buf)
                    self._char_buf.clear()
                    if key == "text":
                        self._state = _L2State.FOUND_TEXT_KEY
                    self._next_is_key_d2 = False
                elif self._depth == 3 and self._next_is_key_d3 and self._state == _L2State.IN_TEXT_OBJ:
                    key = "".join(self._char_buf)
                    self._char_buf.clear()
                    if key == "text":
                        self._state = _L2State.FOUND_STR_KEY
                    self._next_is_key_d3 = False
            else:
                if self._depth == 1 and self._next_is_key_d1:
                    self._char_buf.append(ch)
                elif self._depth == 2 and self._next_is_key_d2 and self._state == _L2State.IN_L2:
                    self._char_buf.append(ch)
                elif self._depth == 3 and self._next_is_key_d3 and self._state == _L2State.IN_TEXT_OBJ:
                    self._char_buf.append(ch)
            return ""

        # ── Outside strings ────────────────────────────────────────────────────
        if ch == '"':
            if self._state == _L2State.EXPECT_VALUE:
                self._state = _L2State.IN_VALUE
            else:
                self._in_string = True
                self._char_buf.clear()
            return ""

        if ch == "{":
            self._depth += 1
            if self._depth == 1:
                self._next_is_key_d1 = True
            elif self._depth == 2 and self._state == _L2State.EXPECT_L2_OBJ:
                self._state = _L2State.IN_L2
                self._next_is_key_d2 = True
            elif self._depth == 3 and self._state == _L2State.EXPECT_TEXT_OBJ:
                self._state = _L2State.IN_TEXT_OBJ
                self._next_is_key_d3 = True
            return ""

        if ch == "}":
            self._depth -= 1
            return ""

        if ch == ":":
            if self._state == _L2State.FOUND_L2_KEY:
                self._state = _L2State.EXPECT_L2_OBJ
            elif self._state == _L2State.FOUND_TEXT_KEY:
                self._state = _L2State.EXPECT_TEXT_OBJ
            elif self._state == _L2State.FOUND_STR_KEY:
                self._state = _L2State.EXPECT_VALUE
            elif self._depth == 1:
                self._next_is_key_d1 = False
            elif self._depth == 2 and self._state == _L2State.IN_L2:
                self._next_is_key_d2 = False
            elif self._depth == 3 and self._state == _L2State.IN_TEXT_OBJ:
                self._next_is_key_d3 = False
            return ""

        if ch == ",":
            if self._depth == 1:
                self._next_is_key_d1 = True
            elif self._depth == 2 and self._state == _L2State.IN_L2:
                self._next_is_key_d2 = True
            elif self._depth == 3 and self._state == _L2State.IN_TEXT_OBJ:
                self._next_is_key_d3 = True
            return ""

        return ""
