import uuid
from agents import Runner
from .agent_builder import build_main_agent
from .state import get_state
from .services.backend import get_database
from .models import ChatMessage

# Initialize agent
PERSONA = 'business'
USER_NAME = 'Николай'
app_state = get_state(user_name=USER_NAME, persona=PERSONA)
main_agent = build_main_agent(app_state)

# Database instance
db = get_database()

# Default conversation ID (you might want to make this per-session)
DEFAULT_CONVERSATION_ID = "default"

async def handle_chat(message: str, conversation_id: str = DEFAULT_CONVERSATION_ID):
    """Handle chat message with database persistence."""
    
    # Load conversation history from database
    conversation_history = db.get_conversation_history_for_agent(conversation_id)
    
    # Add user message to history
    conversation_history.append({"role": "user", "content": message})
    
    # Save user message to database
    user_message = ChatMessage(
        message_id=str(uuid.uuid4()),
        role="user",
        text=message,
        conversation_id=conversation_id
    )
    db.save_message(user_message)
    
    # Process with agent
    result = await Runner.run(main_agent, conversation_history)
    assistant_reply = result.final_output if hasattr(result, "final_output") else str(result)
    
    # Save assistant response to database
    assistant_message = ChatMessage(
        message_id=str(uuid.uuid4()),
        role="assistant",
        text=assistant_reply,
        conversation_id=conversation_id
    )
    db.save_message(assistant_message)
    
    return {"response": assistant_reply}
