"""The chat as a supervised service (ARCHIE-298): it serves requests, so it has
nothing to start; it shows in the runtime status next to the background kinds."""


class ChatService:
    name = "chat"

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def healthy(self) -> bool:
        return True
