import os
import uuid
import logging
from dotenv import load_dotenv
from agents import Runner
from .agent_builder import build_main_agent
from .state import get_state
from .services.backend import get_database
from .models import ChatMessage
from .utils import generate_message_id, generate_conversation_id

logger = logging.getLogger(__name__)
load_dotenv()

# Get configuration from environment variables
PERSONA = os.getenv("DEFAULT_PERSONA", "business")
USER_NAME = os.getenv("DEFAULT_USER_NAME", "Николай")
DEFAULT_CONVERSATION_ID = os.getenv("DEFAULT_CONVERSATION_ID", "default")

logger.info(f"Initializing agent with persona: {PERSONA}, user: {USER_NAME}")
main_agent = build_main_agent()
logger.info("Main agent initialized successfully")

# Database instance
db = get_database()
logger.info("Database connection established")

async def handle_chat(user_message: ChatMessage) -> ChatMessage:
    """Handle chat message with database persistence."""
    
    # Generate conversation_id if not provided
    conversation_id = user_message.conversation_id or generate_conversation_id()
    logger.info(f"Handling chat message for conversation: {conversation_id}")
    logger.debug(f"User message: {user_message.text}")
    
    # Update user message with conversation_id and generate message_id
    user_message.conversation_id = conversation_id
    user_message.message_id = generate_message_id()
    
    # Load conversation history from database (in chronological order for agent)
    conversation_history = db.get_conversation_history_for_agent(conversation_id)
    logger.debug(f"Loaded {len(conversation_history)} messages from history")
    
    # Add user message to history
    conversation_history.append({"role": "user", "content": user_message.text})
    
    # Save user message to database
    db.save_message(user_message)
    logger.debug("User message saved to database")
    
    # Process with agent
    logger.info("Processing message with AI agent...")
    
    # Create agent
    format_aware_agent = build_main_agent()
    
    result = await Runner.run(
        format_aware_agent, 
        conversation_history
    )
    logger.debug(f"Full agent result: {result}")
    
    # Extract response text and metadata from the structured output
    response_text = result.final_output.response
    metadata = result.final_output.metadata
    
    # Save assistant response to database
    assistant_message = ChatMessage(
        message_id=generate_message_id(),
        role="assistant",
        text=response_text,
        text_format=user_message.text_format,
        conversation_id=conversation_id,
        metadata=metadata
    )
    db.save_message(assistant_message)
    logger.debug("Assistant message saved to database")
    
    logger.info("Chat message handling completed successfully")
    return assistant_message
