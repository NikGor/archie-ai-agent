"""Test full agent pipeline with updated parser and reasoning models."""
import asyncio
import json
import logging
from app.agent.agent_factory import AgentFactory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_openai_pipeline():
    """Test with OpenAI GPT-4.1 (non-thinking model)."""
    logger.info("\n" + "="*60)
    logger.info("Testing OpenAI GPT-4.1 Pipeline")
    logger.info("="*60)
    
    agent = AgentFactory()
    messages = [
        {"role": "user", "content": "Create a dinner appointment for tomorrow at 7 PM with John at Restaurant XYZ"}
    ]
    
    response = await agent.arun(
        messages=messages,
        model="gpt-4.1",
        response_format="plain",
        user_name="Test User"
    )
    
    print("\n=== OpenAI GPT-4.1 Response ===")
    print(f"Content: {response.content}")
    print(f"\nLLM Trace: {json.dumps(response.llm_trace.model_dump(), indent=2)}")


async def test_openai_thinking_pipeline():
    """Test with OpenAI GPT-5 (thinking model)."""
    logger.info("\n" + "="*60)
    logger.info("Testing OpenAI GPT-5 (Thinking) Pipeline")
    logger.info("="*60)
    
    agent = AgentFactory()
    messages = [
        {"role": "user", "content": "Create a dinner appointment for tomorrow at 7 PM with John at Restaurant XYZ"}
    ]
    
    response = await agent.arun(
        messages=messages,
        model="gpt-5",
        response_format="plain",
        user_name="Test User"
    )
    
    print("\n=== OpenAI GPT-5 (Thinking) Response ===")
    print(f"Content: {response.content}")
    print(f"\nLLM Trace: {json.dumps(response.llm_trace.model_dump(), indent=2)}")


async def test_gemini_pipeline():
    """Test with Gemini 2.5 Pro (non-thinking model)."""
    logger.info("\n" + "="*60)
    logger.info("Testing Gemini 2.5 Pro Pipeline")
    logger.info("="*60)
    
    agent = AgentFactory()
    messages = [
        {"role": "user", "content": "Create a dinner appointment for tomorrow at 7 PM with John at Restaurant XYZ"}
    ]
    
    response = await agent.arun(
        messages=messages,
        model="gemini-2.5-pro",
        response_format="plain",
        user_name="Test User"
    )
    
    print("\n=== Gemini 2.5 Pro Response ===")
    print(f"Content: {response.content}")
    print(f"\nLLM Trace: {json.dumps(response.llm_trace.model_dump(), indent=2)}")


async def test_gemini_thinking_pipeline():
    """Test with Gemini 2.0 Flash Thinking (thinking model)."""
    logger.info("\n" + "="*60)
    logger.info("Testing Gemini 2.0 Flash Thinking Pipeline")
    logger.info("="*60)
    
    agent = AgentFactory()
    messages = [
        {"role": "user", "content": "Create a dinner appointment for tomorrow at 7 PM with John at Restaurant XYZ"}
    ]
    
    response = await agent.arun(
        messages=messages,
        model="gemini-2.0-flash-thinking-exp",
        response_format="plain",
        user_name="Test User"
    )
    
    print("\n=== Gemini 2.0 Flash Thinking Response ===")
    print(f"Content: {response.content}")
    print(f"\nLLM Trace: {json.dumps(response.llm_trace.model_dump(), indent=2)}")


async def main():
    """Run all pipeline tests."""
    try:
        await test_openai_pipeline()
    except Exception as e:
        logger.error(f"OpenAI GPT-4.1 test failed: {e}", exc_info=True)
    
    try:
        await test_openai_thinking_pipeline()
    except Exception as e:
        logger.error(f"OpenAI GPT-5 test failed: {e}", exc_info=True)
    
    try:
        await test_gemini_pipeline()
    except Exception as e:
        logger.error(f"Gemini 2.5 Pro test failed: {e}", exc_info=True)
    
    try:
        await test_gemini_thinking_pipeline()
    except Exception as e:
        logger.error(f"Gemini Thinking test failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())

# Run with: PYTHONPATH=. poetry run python app/samples/full_pipeline_test.py
