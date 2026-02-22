---
description: Run the project test suite
---

Run `./execute_tests.sh` with one of these patterns:

- **All tests**: `./execute_tests.sh`
- **Unit tests only** (no API keys needed): `./execute_tests.sh -m "not llm"`
- **Smoke tests** (requires `.env` with API keys): `./execute_tests.sh -m llm`
- **Specific test by name**: `./execute_tests.sh -v -k <test_name>`

Report: total passed/failed, tracebacks for any failures.
