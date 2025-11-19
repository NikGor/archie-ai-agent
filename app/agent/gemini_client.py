import json
import logging
import os
import uuid
import ast
from typing import Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import asyncio
from archie_shared.chat.models import LllmTrace, InputTokensDetails, OutputTokensDetails


logger = logging.getLogger(__name__)
load_dotenv()


class MockInputTokensDetails(BaseModel):
    cached_tokens: int = 0


class MockOutputTokensDetails(BaseModel):
    reasoning_tokens: int = 0


class MockUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_tokens_details: MockInputTokensDetails = Field(
        default_factory=MockInputTokensDetails
    )
    output_tokens_details: MockOutputTokensDetails = Field(
        default_factory=MockOutputTokensDetails
    )


class MockContent(BaseModel):
    parsed: Any


class MockOutputItem(BaseModel):
    type: str = "message"
    content: list[MockContent]


class MockOpenAIResponse(BaseModel):
    id: str
    model: str
    usage: MockUsage
    output: list[MockOutputItem]


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

            # Remove fields that should be handled by the client, not the model
            if "properties" in pydantic_schema:
                pydantic_schema["properties"].pop("llm_trace", None)
                pydantic_schema["properties"].pop("response_id", None)
                if "required" in pydantic_schema:
                    pydantic_schema["required"] = [
                        field
                        for field in pydantic_schema["required"]
                        if field not in ["llm_trace", "response_id"]
                    ]

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
            elif "content" not in parsed_data and "response" in parsed_data:
                # Fallback: if Gemini returns "response" instead of "content", map it
                logger.warning("gemini_client_warning_005: 'content' missing, mapping 'response' to 'content'")
                parsed_data["content"] = parsed_data.pop("response")

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

            # Fix structural issues in parsed_data
            parsed_data = self._fix_gemini_response_structure(parsed_data)

            logger.info(f"gemini_client_005: Processed data: \033[32m{json.dumps(parsed_data, indent=2, ensure_ascii=False)}\033[0m")

            # Fix common structural issues in Gemini response to match Pydantic models
            parsed_data = self._fix_gemini_response_structure(parsed_data)

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
            self._log_usage(response, model)

            # Generate response ID for tracking
            response_id = str(uuid.uuid4())

            # Create mock usage
            mock_usage = self._create_mock_usage_from_gemini_response(response)

            # Return MockOpenAIResponse
            return MockOpenAIResponse(
                id=response_id,
                model=model,
                usage=mock_usage,
                output=[MockOutputItem(content=[MockContent(parsed=parsed_result)])],
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
                # Try to find a non-null type in anyOf
                non_null_option = next(
                    (opt for opt in prop_schema["anyOf"] if opt.get("type") != "null"),
                    prop_schema["anyOf"][0]
                )
                # If it's a reference, we need to resolve it or treat it as object if we can't
                if "$ref" in non_null_option:
                    # For now, assume refs are objects. 
                    # Ideally we should resolve refs but Gemini client might handle refs if we pass definitions.
                    # But here we are building schema manually.
                    # Let's try to find the definition in $defs if available in root schema
                    ref_name = non_null_option["$ref"].split("/")[-1]
                    if "$defs" in pydantic_schema and ref_name in pydantic_schema["$defs"]:
                         return convert_property(pydantic_schema["$defs"][ref_name])
                    
                    # Fallback if we can't resolve
                    return genai.types.Schema(type=genai.types.Type.OBJECT)
                
                prop_schema = non_null_option

            prop_type = prop_schema.get("type", "string")

            # Handle array type
            if prop_type == "array" and "items" in prop_schema:
                return genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    items=convert_property(prop_schema["items"]),
                )

            # Handle object type with properties
            if prop_type == "object":
                properties = {}
                if "properties" in prop_schema:
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

    def _create_mock_usage_from_gemini_response(self, response: Any) -> MockUsage:
        usage = getattr(response, "usage_metadata", None)

        def _safe_int(val, default=0):
            try:
                if val is None:
                    return default
                return int(val)
            except Exception:
                return default

        prompt_tokens = _safe_int(getattr(usage, "prompt_token_count", None))
        cached_tokens = _safe_int(getattr(usage, "cached_content_token_count", None))
        candidates_tokens = _safe_int(getattr(usage, "candidates_token_count", None))
        total_tokens = _safe_int(getattr(usage, "total_token_count", None))

        return MockUsage(
            input_tokens=prompt_tokens,
            output_tokens=candidates_tokens,
            total_tokens=total_tokens,
            input_tokens_details=MockInputTokensDetails(cached_tokens=cached_tokens),
            output_tokens_details=MockOutputTokensDetails(reasoning_tokens=0),
        )

    def _log_usage(self, response: Any, model: str) -> None:
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
                f"gemini_client_005: Status: completed | Model: \033[36m{model}\033[0m"
            )
        except Exception as e:
            logger.warning(f"gemini_client_warning_002: Could not log usage: {e}")

    def _fix_gemini_response_structure(self, parsed_data: dict[str, Any]) -> dict[str, Any]:
        """
        Fix common structural issues in Gemini response to match Pydantic models.
        Gemini often returns strings instead of objects for nested fields.
        """
        def _safe_json_load(value: str) -> Any | None:
            try:
                return json.loads(value)
            except Exception:
                try:
                    # Fallback for Python-style dict strings (single quotes) which Gemini sometimes returns
                    return ast.literal_eval(value)
                except Exception:
                    return None

        # Fix SGR structure
        if "sgr" in parsed_data:
            sgr_data = parsed_data["sgr"]
            # If sgr is a string, we already parsed it in create_completion, but let's be safe
            if isinstance(sgr_data, str):
                parsed_obj = _safe_json_load(sgr_data)
                if parsed_obj:
                    sgr_data = parsed_obj
                    parsed_data["sgr"] = sgr_data

            # Check if sgr is missing required fields (routing, verification, pre_action)
            # If so, try to map from flat structure or create defaults
            if isinstance(sgr_data, dict):
                required_fields = ["routing", "verification", "pre_action"]
                missing_fields = [f for f in required_fields if f not in sgr_data]
                
                if missing_fields:
                    logger.warning(f"gemini_client_warning_006: SGR missing fields: {missing_fields}. Attempting to fix.")
                    
                    # Map flat fields to structured fields if they exist
                    # Ensure routing exists and has required fields
                    if "routing" not in sgr_data:
                        intent = sgr_data.get("intent", "answer_general")
                        valid_intents = ["answer_general", "weather", "sports_score", "web_search", "clarify", "out_of_scope"]
                        if intent not in valid_intents:
                            intent = "answer_general"
                        sgr_data["routing"] = {"intent": intent}
                    
                    if "rationale" not in sgr_data["routing"]:
                        sgr_data["routing"]["rationale"] = str(sgr_data.get("rationale", "Auto-generated rationale"))
                    
                    # Ensure verification exists and has required fields
                    if "verification" not in sgr_data:
                        sgr_data["verification"] = {}

                    if "level" not in sgr_data["verification"]:
                        # Try to find it in 'status' field of verification object, or 'verification_status' of root
                        sgr_data["verification"]["level"] = sgr_data["verification"].get("status") or sgr_data.get("verification_status", "unverified")
                        
                    # Ensure level is valid enum
                    valid_levels = ["verified", "partially_verified", "unverified"]
                    if sgr_data["verification"]["level"] not in valid_levels:
                         sgr_data["verification"]["level"] = "unverified"

                    if "confidence_pct" not in sgr_data["verification"]:
                        sgr_data["verification"]["confidence_pct"] = sgr_data.get("confidence_score", 0)

                    if "pre_action" not in sgr_data:
                        sgr_data["pre_action"] = {
                            "summary": "Auto-generated pre-action check",
                            "decision": "none"
                        }
                    
                    # Ensure slots exists
                    if "slots" not in sgr_data:
                        sgr_data["slots"] = {"needed": [], "filled": [], "pending": []}
                    
                    # Ensure evidence exists
                    if "evidence" not in sgr_data:
                        # Try to map from "claims" if present
                        if "claims" in sgr_data and isinstance(sgr_data["claims"], list):
                            evidence_list = []
                            for claim in sgr_data["claims"]:
                                if isinstance(claim, dict):
                                    evidence_list.append({
                                        "claim": claim.get("claim", "Unknown claim"),
                                        "support": "supported", # Assume supported if listed
                                        "source_ids": claim.get("source_ids", [])
                                    })
                            sgr_data["evidence"] = evidence_list
                        else:
                            sgr_data["evidence"] = []
                            
                    # Ensure sources exists
                    if "sources" not in sgr_data:
                        sgr_data["sources"] = []

        # Fix Content structure
        if "content" in parsed_data and isinstance(parsed_data["content"], dict):
            content = parsed_data["content"]
            
            # Handle UI Answer
            if "ui_answer" in content and isinstance(content["ui_answer"], dict):
                ui_answer = content["ui_answer"]
                
                # Fix intro_text: string -> TextAnswer
                if "intro_text" in ui_answer and isinstance(ui_answer["intro_text"], str):
                    ui_answer["intro_text"] = {
                        "type": "plain",
                        "text": ui_answer["intro_text"]
                    }
                
                # Fix items: list[str] -> list[dict]
                if "items" in ui_answer and isinstance(ui_answer["items"], list):
                    fixed_items = []
                    for idx, item in enumerate(ui_answer["items"]):
                        # If item is a string, try to parse it
                        if isinstance(item, str):
                            parsed_item = _safe_json_load(item)
                            if parsed_item:
                                item = parsed_item
                            else:
                                # Fallback for plain string
                                item = {
                                    "type": "text",
                                    "order": idx + 1,
                                    "content": {
                                        "type": "plain",
                                        "text": item
                                    }
                                }
                                fixed_items.append(item)
                                continue

                        # Now item is a dict. We need to ensure it matches AdvancedAnswerItem structure.
                        # AdvancedAnswerItem expects: type, order, content

                        # Special handling: Gemini sometimes wraps the real UI component inside a text_answer string
                        # e.g. type="text_answer", content={"text": "{'type': 'card_grid', ...}"}
                        if item.get("type") == "text_answer" and "content" in item and isinstance(item["content"], dict):
                            text_val = item["content"].get("text", "")
                            if isinstance(text_val, str) and (text_val.strip().startswith("{") or text_val.strip().startswith("[")):
                                parsed_inner = _safe_json_load(text_val)
                                if isinstance(parsed_inner, dict):
                                    logger.info("gemini_client_004d: Unwrapped hidden UI component from text_answer")
                                    # If the parsed object looks like a full item (has type), use it
                                    if "type" in parsed_inner:
                                        # Preserve order if present in original but not in inner
                                        if "order" in item and "order" not in parsed_inner:
                                            parsed_inner["order"] = item["order"]
                                        item = parsed_inner
                                    else:
                                        # It might be a raw card object, let it fall through to the wrapper logic
                                        if "order" in item and "order" not in parsed_inner:
                                            parsed_inner["order"] = item["order"]
                                        item = parsed_inner

                        # Check if it's already in the correct format (has 'content' wrapper)
                        if "content" in item and "type" in item and item["type"] in ["card", "elements", "table", "text"]:
                            # Ensure order is present
                            if "order" not in item:
                                item["order"] = idx + 1
                            
                            # Fix inner content fields if needed
                            inner_content = item["content"]
                            if isinstance(inner_content, dict):
                                # Fix card_type -> type
                                if "card_type" in inner_content:
                                    inner_content["type"] = inner_content.pop("card_type")
                                
                                # Fix buttons inside content
                                if "buttons" in inner_content and isinstance(inner_content["buttons"], list):
                                    fixed_buttons = []
                                    for btn in inner_content["buttons"]:
                                        if isinstance(btn, dict):
                                            # Fix button_text -> text
                                            if "button_text" in btn:
                                                btn["text"] = btn.pop("button_text")
                                            # Fix name -> assistant_request (for assistant buttons)
                                            if "name" in btn:
                                                btn["assistant_request"] = btn.pop("name")
                                            # Infer type if missing
                                            if "type" not in btn:
                                                if "action" in btn:
                                                    btn["type"] = "frontend_button"
                                                else:
                                                    btn["type"] = "assistant_button"
                                        fixed_buttons.append(btn)
                                    inner_content["buttons"] = fixed_buttons
                            
                            # Normalize top-level item type to Agent schema expected literals
                            type_map = {
                                "card": "card_grid",
                                "elements": "card_grid",
                                "text": "text_answer",
                            }
                            if item.get("type") in type_map:
                                item["type"] = type_map[item["type"]]
                            fixed_items.append(item)
                        else:
                            # It's likely a raw card/table object that needs wrapping
                            wrapper = {"order": idx + 1}

                            # Determine type and wrap content to match Agent schema
                            if "card_type" in item or "location" in item or "track_title" in item or "movie_title" in item or item.get("type", "").lower() == "movie":
                                # Normalize to card_grid with cards list
                                if "card_type" in item:
                                    item["type"] = item.pop("card_type")

                                # Fix buttons inside the card
                                if "buttons" in item and isinstance(item["buttons"], list):
                                    fixed_buttons = []
                                    for btn in item["buttons"]:
                                        if isinstance(btn, dict):
                                            if "button_text" in btn:
                                                btn["text"] = btn.pop("button_text")
                                            if "label" in btn and "text" not in btn:
                                                btn["text"] = btn.pop("label")
                                            if "name" in btn:
                                                btn["assistant_request"] = btn.pop("name")
                                            if "type" not in btn:
                                                if "action" in btn or "value" in btn:
                                                    btn["type"] = "frontend_button"
                                                else:
                                                    btn["type"] = "assistant_button"
                                        fixed_buttons.append(btn)
                                    item["buttons"] = fixed_buttons

                                wrapper["type"] = "card_grid"
                                wrapper["content"] = {"cards": [item]}

                            elif "headers" in item and "rows" in item:
                                wrapper["type"] = "table"
                                wrapper["content"] = item
                            elif "elements" in item:
                                wrapper["type"] = "card_grid"
                                wrapper["content"] = item
                            elif "text" in item and ("type" in item and item["type"] in ["plain", "markdown", "html"]):
                                wrapper["type"] = "text_answer"
                                wrapper["content"] = item
                            else:
                                # Fallback: treat as text if possible, or generic card
                                logger.warning(f"gemini_client_warning_007: Could not determine item type for {item}. Defaulting to text_answer.")
                                wrapper["type"] = "text_answer"
                                wrapper["content"] = {
                                    "type": "plain",
                                    "text": str(item)
                                }

                            fixed_items.append(wrapper)
                            
                    ui_answer["items"] = fixed_items
                
                # Fix quick_action_buttons: list[str] -> list[dict]
                if "quick_action_buttons" in ui_answer and isinstance(ui_answer["quick_action_buttons"], dict):
                    qa_buttons = ui_answer["quick_action_buttons"]
                    if "buttons" in qa_buttons and isinstance(qa_buttons["buttons"], list):
                        fixed_buttons = []
                        for btn in qa_buttons["buttons"]:
                            if isinstance(btn, str):
                                fixed_buttons.append({
                                    "type": "assistant_button",
                                    "text": btn,
                                    "assistant_request": btn
                                })
                            elif isinstance(btn, dict):
                                # Fix button_text -> text
                                if "button_text" in btn:
                                    btn["text"] = btn.pop("button_text")
                                # Fix name -> assistant_request
                                if "name" in btn:
                                    btn["assistant_request"] = btn.pop("name")
                                # Ensure type exists
                                if "type" not in btn:
                                    btn["type"] = "assistant_button"
                                fixed_buttons.append(btn)
                            else:
                                fixed_buttons.append(btn)
                        qa_buttons["buttons"] = fixed_buttons

        return parsed_data
