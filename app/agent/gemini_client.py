import json
import logging
import os
import uuid
from typing import Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import asyncio
from archie_shared.chat.models import LllmTrace, InputTokensDetails, OutputTokensDetails


logger = logging.getLogger(__name__)
load_dotenv()


class GeminiResponseWrapper:
    """Wrapper to make Gemini response compatible with AgentFactory expectations."""

    def __init__(self, parsed_result: BaseModel, llm_trace: LllmTrace, response_id: str):
        self.parsed_result = parsed_result
        self.llm_trace = llm_trace
        self.response_id = response_id


class GeminiClient:
    """Client for Google Gemini API interactions."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key)
        logger.info("gemini_client_001: Initialized Gemini client")

    async def create_completion(
        self,
        messages: list[dict[str, Any]],
        model: str,
        response_format: type[BaseModel],
        tools: list[dict[str, Any]] | None = None,
        previous_response_id: str | None = None,
    ) -> Any:
        """
        Create a completion using Gemini API with structured outputs.

        IMPORTANT LIMITATION: Gemini's structured outputs only work reliably with
        simple, flat schemas. Complex nested objects (like AgentResponse with
        Content/SGRTrace/LlmTrace) are returned as strings instead of objects.

        For production use with complex schemas, consider:
        1. Flattening the response schema
        2. Using JSON mode without schema enforcement
        3. Post-processing to parse nested string fields
        """
        msg_breakdown = {}
        for msg in messages:
            role = msg.get("role", "unknown")
            msg_breakdown[role] = msg_breakdown.get(role, 0) + 1
        logger.info(
            f"gemini_client_002: Calling \033[36m{model}\033[0m with \033[33m{len(messages)}\033[0m msgs "
            f"(system: {msg_breakdown.get('system', 0)}, user: {msg_breakdown.get('user', 0)}, assistant: {msg_breakdown.get('assistant', 0)})"
        )
        try:
            # Extract system instruction from first message
            system_instruction = None
            if messages and messages[0]["role"] == "system":
                system_instruction = [types.Part.from_text(text=messages[0]["content"])]
                messages = messages[1:]

            # Convert messages to Gemini format
            contents = []
            for msg in messages:
                if msg["role"] == "user":
                    contents.append(
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=msg["content"])],
                        )
                    )
                elif msg["role"] == "assistant":
                    contents.append(
                        types.Content(
                            role="model",
                            parts=[types.Part.from_text(text=msg["content"])],
                        )
                    )

            # Convert Pydantic schema to Gemini schema format
            pydantic_schema = response_format.model_json_schema()
            gemini_schema = self._convert_schema_to_gemini(pydantic_schema)

            # Configure generation
            generate_content_config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=gemini_schema,
                system_instruction=system_instruction,
            )

            # Generate response
            logger.info("gemini_client_003: Calling generate_content")
            response = self.client.models.generate_content(
                model=model,
                contents=contents,
                config=generate_content_config,
            )

            # Parse the response
            parsed_data = json.loads(response.text)
            logger.info(f"gemini_client_004: Raw response data: \033[32m{json.dumps(parsed_data, indent=2, ensure_ascii=False)}\033[0m")

            # Debug: save to file
            with open("/tmp/gemini_ui_answer.json", "w") as f:
                json.dump(parsed_data, f, indent=2, ensure_ascii=False)
            logger.info("gemini_client_debug: Saved to /tmp/gemini_ui_answer.json")

            # Fix Gemini's issue: content, sgr and llm_trace often come back as plain strings
            # Defensive parsing: try JSON decode, otherwise wrap strings into minimal structures
            def _safe_json_load(value: str):
                try:
                    return json.loads(value)
                except Exception:
                    return None

            # content: if it's a simple string, wrap into a minimal content dict so Pydantic
            # Content model (which expects an object) can accept it. Prefer JSON if possible.
            if "content" in parsed_data and isinstance(parsed_data["content"], str):
                parsed_obj = _safe_json_load(parsed_data["content"])
                if parsed_obj is not None:
                    parsed_data["content"] = parsed_obj
                    logger.info("gemini_client_004c: Parsed content JSON string to object")
                else:
                    # Wrap string into minimal object with a `text` field
                    logger.warning("gemini_client_warning_002: content is plain string, wrapping into {'text': ...}")
                    parsed_data["content"] = {"text": parsed_data["content"]}

            # sgr: if Gemini returned a textual SGR trace, create a safe fallback SGR structure
            if "sgr" in parsed_data and isinstance(parsed_data["sgr"], str):
                parsed_obj = _safe_json_load(parsed_data["sgr"])
                if parsed_obj is not None:
                    parsed_data["sgr"] = parsed_obj
                    logger.info("gemini_client_004a: Parsed sgr JSON string to object")
                else:
                    logger.warning("gemini_client_warning_003: sgr is plain string, creating fallback SGR structure")
                    sgr_text = parsed_data["sgr"]
                    # Create minimal SGRTrace-compatible dict with the raw text placed into routing.rationale
                    parsed_data["sgr"] = {
                        "routing": {"intent": "answer_general", "rationale": sgr_text[:1000]},
                        "slots": {"needed": [], "filled": [], "pending": []},
                        "evidence": [],
                        "sources": [],
                        "verification": {"level": "unverified", "confidence_pct": 0},
                        "pre_action": {"summary": "Fallback SGR generated from raw string", "decision": "none"},
                    }

            # llm_trace: we won't try to inject Gemini's freeform trace into the structured llm_trace
            # instead we build a safe LllmTrace below from usage metadata. Remove any textual llm_trace
            # so it doesn't confuse model validation of parsed_data.
            if "llm_trace" in parsed_data and isinstance(parsed_data["llm_trace"], str):
                logger.warning("gemini_client_warning_004: llm_trace present as string; discarding textual llm_trace and building structured LllmTrace from usage metadata")
            parsed_data.pop("llm_trace", None)

            logger.info(f"gemini_client_005: Processed data: \033[32m{json.dumps(parsed_data, indent=2, ensure_ascii=False)}\033[0m")

            # Create proper LllmTrace
            llm_trace = self._create_llm_trace_from_gemini_response(response, model)

            # Parse the data without llm_trace, then create AgentResponse manually
            if response_format.__name__ == "AgentResponse":
                # For AgentResponse, we need to construct it manually with our llm_trace
                temp_data = {**parsed_data, "llm_trace": llm_trace.model_dump()}
                parsed_result = response_format.model_validate(temp_data)
            else:
                # For other response formats, parse normally
                parsed_result = response_format.model_validate(parsed_data)

            # Log usage
            self._log_usage(response)

            # Generate response ID for tracking
            response_id = str(uuid.uuid4())

            # Return wrapper instead of raising NotImplementedError
            return GeminiResponseWrapper(
                parsed_result=parsed_result, 
                llm_trace=llm_trace,
                response_id=response_id
            )

        except Exception as e:
            logger.error(f"gemini_client_error_001: \033[31m{e!s}\033[0m")
            raise

    def _convert_schema_to_gemini(
        self, pydantic_schema: dict[str, Any]
    ) -> genai.types.Schema:
        """Convert Pydantic JSON schema to Gemini Schema format following the reference pattern."""

        def convert_type(json_type: str | list) -> genai.types.Type:
            """Convert JSON schema type to Gemini Type."""
            # Handle union types like ["string", "null"]
            if isinstance(json_type, list):
                json_type = next((t for t in json_type if t != "null"), "string")

            type_mapping = {
                "string": genai.types.Type.STRING,
                "number": genai.types.Type.NUMBER,
                "integer": genai.types.Type.INTEGER,
                "boolean": genai.types.Type.BOOLEAN,
                "array": genai.types.Type.ARRAY,
                "object": genai.types.Type.OBJECT,
            }
            return type_mapping.get(json_type, genai.types.Type.STRING)

        def convert_property(prop_schema: dict[str, Any]) -> genai.types.Schema:
            """Convert a single property schema recursively."""
            # Handle anyOf/allOf/oneOf by taking first option
            if "anyOf" in prop_schema:
                prop_schema = prop_schema["anyOf"][0]

            prop_type = prop_schema.get("type", "string")

            # Handle array type
            if prop_type == "array" and "items" in prop_schema:
                return genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    items=convert_property(prop_schema["items"]),
                )

            # Handle object type with properties
            if prop_type == "object" and "properties" in prop_schema:
                properties = {
                    k: convert_property(v) for k, v in prop_schema["properties"].items()
                }
                required = prop_schema.get("required", [])
                return genai.types.Schema(
                    type=genai.types.Type.OBJECT,
                    properties=properties,
                    required=required if required else None,
                )

            # Simple type
            return genai.types.Schema(type=convert_type(prop_type))

        # Convert root schema properties
        properties = {}
        if "properties" in pydantic_schema:
            properties = {
                k: convert_property(v) for k, v in pydantic_schema["properties"].items()
            }

        required = pydantic_schema.get("required", [])

        return genai.types.Schema(
            type=genai.types.Type.OBJECT,
            properties=properties,
            required=required if required else None,
        )

    def _create_llm_trace_from_gemini_response(
        self, response: Any, model: str
    ) -> LllmTrace:
        """Create proper LllmTrace from Gemini response."""
        usage = getattr(response, "usage_metadata", None)

        def _safe_int(val, default=0):
            try:
                if val is None:
                    return default
                return int(val)
            except Exception:
                return default

        def _safe_float(val, default=0.0):
            try:
                if val is None:
                    return default
                return float(val)
            except Exception:
                return default

        prompt_tokens = _safe_int(getattr(usage, "prompt_token_count", None))
        cached_tokens = _safe_int(getattr(usage, "cached_content_token_count", None))
        candidates_tokens = _safe_int(getattr(usage, "candidates_token_count", None))
        total_tokens = _safe_int(getattr(usage, "total_token_count", None))

        return LllmTrace(
            model=model,
            input_tokens=prompt_tokens,
            input_tokens_details=InputTokensDetails(
                cached_tokens=cached_tokens
            ),
            output_tokens=candidates_tokens,
            output_tokens_details=OutputTokensDetails(
                reasoning_tokens=0  # Gemini doesn't provide reasoning tokens
            ),
            total_tokens=total_tokens,
            total_cost=_safe_float(getattr(usage, "total_cost", None)) if usage else 0.0,
        )

    def _log_usage(self, response: Any) -> None:
        """Log token usage from response."""
        try:
            usage = getattr(response, "usage_metadata", None)
            if usage:
                in_tok = getattr(usage, "prompt_token_count", None)
                out_tok = getattr(usage, "candidates_token_count", None)
                total = getattr(usage, "total_token_count", None)
                cached = getattr(usage, "cached_content_token_count", None)
            else:
                in_tok = out_tok = total = cached = None
            logger.info(
                f"gemini_client_004: Usage - Input: \033[33m{in_tok}\033[0m | "
                f"Output: \033[33m{out_tok}\033[0m | "
                f"Total: \033[33m{total}\033[0m | "
                f"Cached: \033[33m{cached}\033[0m"
            )
            logger.info(
                f"gemini_client_005: Status: completed | Model: \033[36m{response.model}\033[0m"
            )
        except Exception as e:
            logger.warning(f"gemini_client_warning_002: Could not log usage: {e}")
