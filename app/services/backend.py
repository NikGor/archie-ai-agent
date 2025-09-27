"""
SQLite backend for storing chat messages and conversations.
"""

import os
import sqlite3
import json
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from ..models import ChatMessage, Conversation

load_dotenv()
logger = logging.getLogger(__name__)


class ChatDatabase:
    """SQLite database handler for chat messages and conversations."""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.getenv("DATABASE_PATH", "data/chat.db")
        # Ensure directory exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    llm_trace TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    text_format TEXT DEFAULT 'plain',
                    text TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    llm_trace TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_id 
                ON messages (conversation_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_created_at 
                ON messages (created_at)
            """)
            
            conn.commit()
            logger.info(f"Database initialized: {self.db_path}")
    
    def save_message(self, message: ChatMessage) -> None:
        """Save a chat message to the database."""
        with sqlite3.connect(self.db_path) as conn:
            # Ensure conversation exists
            if message.conversation_id:
                self._ensure_conversation_exists(conn, message.conversation_id)
            
            conn.execute("""
                INSERT OR REPLACE INTO messages 
                (message_id, conversation_id, role, text_format, text, metadata, created_at, llm_trace)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                message.message_id,
                message.conversation_id,
                message.role,
                message.text_format,
                message.text,
                json.dumps(message.metadata.dict()) if message.metadata else None,
                message.created_at.isoformat(),
                json.dumps(message.llm_trace.dict()) if message.llm_trace else None
            ))
            conn.commit()
            logger.debug(f"Saved message {message.message_id}")
    
    def save_conversation(self, conversation: Conversation) -> None:
        """Save a conversation and all its messages."""
        with sqlite3.connect(self.db_path) as conn:
            # Save conversation
            conn.execute("""
                INSERT OR REPLACE INTO conversations 
                (conversation_id, created_at, llm_trace)
                VALUES (?, ?, ?)
            """, (
                conversation.conversation_id,
                conversation.created_at.isoformat(),
                json.dumps(conversation.llm_trace.dict()) if conversation.llm_trace else None
            ))
            
            # Save all messages
            for message in conversation.messages:
                message.conversation_id = conversation.conversation_id
                conn.execute("""
                    INSERT OR REPLACE INTO messages 
                    (message_id, conversation_id, role, text_format, text, metadata, created_at, llm_trace)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    message.message_id,
                    message.conversation_id,
                    message.role,
                    message.text_format,
                    message.text,
                    json.dumps(message.metadata.dict()) if message.metadata else None,
                    message.created_at.isoformat(),
                    json.dumps(message.llm_trace.dict()) if message.llm_trace else None
                ))
            
            conn.commit()
            logger.debug(f"Saved conversation {conversation.conversation_id} with {len(conversation.messages)} messages")
    
    def get_conversation_history(self, conversation_id: str, order_desc: bool = False) -> List[ChatMessage]:
        """Load conversation history ordered by creation time."""
        order_clause = "ORDER BY created_at DESC" if order_desc else "ORDER BY created_at ASC"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(f"""
                SELECT * FROM messages 
                WHERE conversation_id = ? 
                {order_clause}
            """, (conversation_id,))
            
            messages = []
            for row in cursor.fetchall():
                message_dict = dict(row)
                
                # Parse JSON fields
                if message_dict['metadata']:
                    message_dict['metadata'] = json.loads(message_dict['metadata'])
                
                if message_dict['llm_trace']:
                    message_dict['llm_trace'] = json.loads(message_dict['llm_trace'])
                
                # Convert datetime string back to datetime object
                message_dict['created_at'] = datetime.fromisoformat(message_dict['created_at'])
                
                messages.append(ChatMessage(**message_dict))
            
            logger.debug(f"Loaded {len(messages)} messages for conversation {conversation_id}")
            return messages
    
    def get_conversation_history_for_agent(self, conversation_id: str) -> List[Dict[str, str]]:
        """Get conversation history in OpenAI API compatible format (chronological order)."""
        messages = self.get_conversation_history(conversation_id, order_desc=False)
        return [
            {"role": msg.role, "content": msg.text}
            for msg in messages
        ]
    
    def _ensure_conversation_exists(self, conn: sqlite3.Connection, conversation_id: str) -> None:
        """Ensure conversation record exists."""
        cursor = conn.execute(
            "SELECT 1 FROM conversations WHERE conversation_id = ?", 
            (conversation_id,)
        )
        
        if not cursor.fetchone():
            conn.execute("""
                INSERT INTO conversations (conversation_id, created_at)
                VALUES (?, ?)
            """, (conversation_id, datetime.now(timezone.utc).isoformat()))
            logger.debug(f"Created conversation {conversation_id}")
    
    def list_conversations(self, limit: int = 50) -> List[str]:
        """List recent conversation IDs."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT conversation_id FROM conversations 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (limit,))
            
            return [row[0] for row in cursor.fetchall()]
    
    def get_all_conversations(self, limit: int = 50) -> List[Conversation]:
        """Get all conversations with their basic info (without messages)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT conversation_id, created_at, llm_trace 
                FROM conversations 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (limit,))
            
            conversations = []
            for row in cursor.fetchall():
                conversation_dict = dict(row)
                
                # Parse JSON fields
                if conversation_dict['llm_trace']:
                    conversation_dict['llm_trace'] = json.loads(conversation_dict['llm_trace'])
                
                # Convert datetime string back to datetime object
                conversation_dict['created_at'] = datetime.fromisoformat(conversation_dict['created_at'])
                
                # Create conversation without messages (empty list)
                conversation_dict['messages'] = []
                
                conversations.append(Conversation(**conversation_dict))
            
            logger.debug(f"Loaded {len(conversations)} conversations")
            return conversations
    
    def get_conversation_with_messages(self, conversation_id: str) -> Optional[Conversation]:
        """Get a complete conversation with all its messages."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get conversation info
            cursor = conn.execute("""
                SELECT conversation_id, created_at, llm_trace 
                FROM conversations 
                WHERE conversation_id = ?
            """, (conversation_id,))
            
            conv_row = cursor.fetchone()
            if not conv_row:
                return None
            
            conversation_dict = dict(conv_row)
            
            # Parse JSON fields
            if conversation_dict['llm_trace']:
                conversation_dict['llm_trace'] = json.loads(conversation_dict['llm_trace'])
            
            # Convert datetime string back to datetime object
            conversation_dict['created_at'] = datetime.fromisoformat(conversation_dict['created_at'])
            
            # Get messages for this conversation (sorted by newest first for API response)
            messages = self.get_conversation_history(conversation_id, order_desc=True)
            conversation_dict['messages'] = messages
            
            return Conversation(**conversation_dict)
    
    def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation and all its messages."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            conn.execute("DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,))
            conn.commit()
            logger.info(f"Deleted conversation {conversation_id}")


# Global database instance
_db_instance = None

def get_database() -> ChatDatabase:
    """Get global database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = ChatDatabase()
    return _db_instance