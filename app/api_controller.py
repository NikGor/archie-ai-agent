from agents import Runner
from .agent_builder import build_main_agent
from .state import get_state

# Initialize agent
PERSONA = 'business'
USER_NAME = 'Николай'
app_state = get_state(user_name=USER_NAME, persona=PERSONA)
main_agent = build_main_agent(app_state)

# History for conversation continuity
conversation_history = []

async def handle_chat(message: str):
    global conversation_history
    
    conversation_history.append({"role": "user", "content": message})
    
    result = await Runner.run(main_agent, conversation_history)
    assistant_reply = result.final_output if hasattr(result, "final_output") else str(result)
    
    conversation_history.append({"role": "assistant", "content": assistant_reply})
    
    return {"response": assistant_reply}
