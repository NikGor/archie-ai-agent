"""HTTP client for backend API communication."""

import logging
import os
from typing import Any, Dict, List, Optional

import httpx
from archie_shared.chat.models import ChatMessage
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class BackendClient:
    """HTTP client for backend API communication."""
    
    def __init__(self):
        self.base_url = os.getenv("BACKEND_API_URL", "http://0.0.0.0:8002")
        self.timeout = 30.0
        
    async def create_conversation(
        self, 
        conversation_id: Optional[str] = None,
        title: str = "New Conversation"
    ) -> str:
        """Create a new conversation via API."""
        try:
            logger.info("backend_001: Creating conversation via API")
            logger.info(f"backend_002: Backend URL: \033[36m{self.base_url}\033[0m")
            
            payload = {
                "conversation_id": conversation_id,
                "title": title
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/conversations",
                    json=payload
                )
                response.raise_for_status()
                
                data = response.json()
                created_conversation_id = data["conversation_id"]
                
                logger.info(f"backend_003: Created conversation: \033[32m{created_conversation_id}\033[0m")
                return created_conversation_id
                
        except Exception as e:
            logger.error(f"backend_error_001: Conversation creation failed: \033[31m{e!s}\033[0m")
            raise
    
    async def create_message(self, message: ChatMessage) -> ChatMessage:
        """Create a message via API."""
        try:
            logger.info("backend_004: Creating message via API")
            logger.info(f"backend_005: Message ID: \033[36m{message.message_id or 'AUTO'}\033[0m")
            payload = message.model_dump(mode='json')
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/messages",
                    json=payload
                )
                response.raise_for_status()
                
                data = response.json()
                message.message_id = data["message_id"]
                message.conversation_id = data["conversation_id"]
                
                logger.info(f"backend_006: Created message: \033[32m{message.message_id}\033[0m")
                return message
                
        except Exception as e:
            logger.error(f"backend_error_002: Message creation failed: \033[31m{e!s}\033[0m")
            raise
    
    async def get_messages(
        self, 
        conversation_id: Optional[str] = None,
        limit: int = 50
    ) -> List[ChatMessage]:
        """Get messages from API."""
        try:
            logger.info("backend_007: Getting messages via API")
            
            params = {"limit": limit}
            if conversation_id:
                params["conversation_id"] = conversation_id
                logger.info(f"backend_008: Conversation filter: \033[36m{conversation_id}\033[0m")
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/messages",
                    params=params
                )
                response.raise_for_status()
                
                data = response.json()
                messages = [ChatMessage(**msg) for msg in data]
                
                logger.info(f"backend_009: Retrieved messages: \033[33m{len(messages)}\033[0m")
                return messages
                
        except Exception as e:
            logger.error(f"backend_error_003: Message retrieval failed: \033[31m{e!s}\033[0m")
            return []
