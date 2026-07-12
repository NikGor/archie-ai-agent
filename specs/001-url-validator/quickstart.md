# Quickstart: Validating Response URL Validation

Prerequisites: repo checked out on branch `001-url-validator`, dependencies
installed (`poetry install` / existing `.venv`), no extra env vars required
(feature adds no new external service).

## 1. Run the unit tests for this feature

```bash
./execute_tests.sh -m "not llm" -k url_validator
```

Expected: all tests in `tests/test_url_validator.py` pass without network
access, covering (see [contracts/url_validator.md](./contracts/url_validator.md)
and [data-model.md](./data-model.md)):
- a valid, reachable URL passes through unchanged
- a URL returning 404 (or any non-2xx) is treated as broken
- a broken markdown link in `TextAnswer.text` is reduced to plain text
- a broken `FrontendButton.url` with no replacement becomes an
  `AssistantButton`
- a broken `FrontendButton.url` with a Google Search fallback available is
  replaced with the fallback URL

## 2. Run the full non-LLM suite to confirm no regression

```bash
./execute_tests.sh -m "not llm"
```

Expected: 100% of previously passing tests still pass (per FR-008 / SC-004 —
existing behavior for valid/no-link responses is unchanged).

## 3. Manual smoke check (optional, requires `.env` with API keys)

```bash
make api-test
```

Send a request that would normally produce a `url_to` or
`open_on_youtube_video` button, and inspect the returned JSON:
- If the model's URL happens to be valid, the button is unchanged.
- If you want to force the broken path, temporarily point
  `app/tools/create_output_tool.py`'s call to `validate_and_fix_urls` at a
  content object with a deliberately invalid URL (e.g. via a scratch script
  in `scripts/`, not committed) and confirm:
  - a `url_validator_NNN:` WARNING log line appears, and
  - the returned button is either an `assistant_button` or points to a
    working replacement URL.

## 4. Confirm latency is not affected

Compare response time for a multi-link response (e.g. a `dashboard` or
`ui_answer` with several buttons + a location card) before and after the
change; per SC-003 the difference should be within normal request-to-request
variance, since all checks run concurrently with a 3s worst-case bound per
check, not per response.
