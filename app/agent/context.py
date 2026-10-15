"""Conversation threading context shared across the 3-stage pipeline."""

import logging
from dataclasses import dataclass


logger = logging.getLogger(__name__)

OPENAI_RESPONSE_ID_PREFIX = "resp_"


@dataclass(frozen=True)
class ConversationContext:
    """Carries conversation continuity across LLM calls.

    OpenAI threads a conversation via `previous_response_id` and must not
    receive the inlined chat history again. Every other provider has no such
    mechanism, so `chat_history` is inlined into the prompt instead. This is
    the single owner of that provider-routing rule, previously duplicated at
    each call site.
    """

    previous_response_id: str | None = None
    chat_history: str | None = None

    def for_provider(self, provider: str) -> "ConversationContext":
        """Return the context fields actually valid for `provider`."""
        if provider == "openai":
            if self.previous_response_id and self.previous_response_id.startswith(
                OPENAI_RESPONSE_ID_PREFIX
            ):
                return ConversationContext(
                    previous_response_id=self.previous_response_id
                )
            # ARCHIE-210: a non-OpenAI id (e.g. our generated message-...) cannot
            # be threaded, so fall back to inlined history instead of losing it.
            if self.previous_response_id:
                logger.warning(
                    f"context_001: previous_response_id \033[33m{self.previous_response_id}\033[0m "
                    "is not an OpenAI response id, falling back to chat_history"
                )
        return ConversationContext(chat_history=self.chat_history)
