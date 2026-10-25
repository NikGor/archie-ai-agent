"""Conversation threading context shared by every LLM call (ARCHIE-303)."""

import logging
from dataclasses import dataclass
from .registry import get_provider


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConversationContext:
    """Carries conversation continuity across LLM calls.

    A provider that threads by response id (OpenAI) continues from
    `previous_response_id` and must not receive the inlined chat history again.
    Every other provider gets `chat_history` inlined into the prompt instead.
    """

    previous_response_id: str | None = None
    chat_history: str | None = None

    def for_provider(self, provider: str) -> "ConversationContext":
        """Return the context fields actually valid for `provider`."""
        adapter = get_provider(provider)
        if adapter.threads_by_response_id:
            if self.previous_response_id and adapter.accepts_response_id(
                self.previous_response_id
            ):
                return ConversationContext(
                    previous_response_id=self.previous_response_id
                )
            # ARCHIE-210: a foreign id (e.g. our generated message-...) cannot
            # be threaded, so fall back to inlined history instead of losing it.
            if self.previous_response_id:
                logger.warning(
                    f"context_001: previous_response_id \033[33m{self.previous_response_id}\033[0m "
                    f"is not a {provider} response id, falling back to chat_history"
                )
        return ConversationContext(chat_history=self.chat_history)
