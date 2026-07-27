"""Conversation threading context shared across the 3-stage pipeline."""

from dataclasses import dataclass


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
            return ConversationContext(previous_response_id=self.previous_response_id)
        return ConversationContext(chat_history=self.chat_history)
