"""Prompt spec rules a stored turn is judged against (ARCHIE-213).

A turn is judged by the rules of the prompts that produced it: cmd_prompt
(Stage 1), assistant_prompt and the response format spec (Stage 3). A rule is
judged only if its effect is visible in what the monitor has: the stored
response, the conversation and, for tool rules, the Langfuse trace.

Not judged:
- templated rules: state facts like "Timezone: {{ state.user_timezone }}";
- rules in EXCLUDED_RULES: Stage 1 internals, output schema instructions,
  image generation prompts (the monitor drops image_prompt), reference tables,
  examples and definitions;
- rules and sections whose `when` is false for the turn's format, or depends on
  the user state, which the monitor does not have for old turns;
- persona specs: the persona of an old turn is unknown.

GUIDANCE_RULES are style advice: breaking one makes the turn orange, never red,
so the turn grade stays meaningful with 30+ rules per turn.
"""

import functools
import re
from dataclasses import dataclass
from app.agent.prompt_builder import PromptBuilder
from app.agent.prompt_compiler import PromptCompiler, PromptLibrary, load_library


COMMAND_SPEC = "cmd_prompt"
ASSISTANT_SPEC = "assistant_prompt"
# Formats that skip Stage 1, so cmd_prompt did not take part
DIRECT_FORMATS = frozenset({"dashboard", "widget"})

EXCLUDED_RULES = frozenset(
    {
        # cmd_prompt: Stage 1 role, decision mechanics and output schema
        "command_role",
        "decide_next_action",
        "action_function_call",
        "action_parameters_request",
        "action_final_response",
        "explain_reasoning",
        "run_tools_in_parallel",
        "stop_when_results_complete",
        "fill_ui_intents",
        "required_ui_intents",
        "other_ui_intents",
        "ui_intents_scope",
        "follow_decision_schema",
        # format specs: fill-the-field instructions, definitions, references, examples
        "ui_fill",
        "level2_fill",
        "fill_text_field",
        "ui_card_grid_example",
        "markdown_element_guide",
        "assistant_button_follow_up",
        "frontend_button_ui_action",
        "ui_email_form",
        "ui_event_form",
        "ui_note_form",
        # image prompts: the monitor does not see image_prompt
        "image_pick_visual_format",
        "image_constraints",
        "image_visual_library",
        "image_styles",
        "card_image_one_style",
        "card_image_prompt_rules",
        "ui_data_driven",
        # output schema instructions: the agent maps the schema to the stored Content
        "set_content_format_formatted_text",
        "leave_ui_answer_null",
        "remind_ui_answer_null",
        "dashboard_fill",
        "level3_fill",
        "widget_fill",
        "tts_fill_script",
        # permissive: "can include multiple widgets" cannot be broken
        "level3_multiple_widgets",
        # purpose statements, structure descriptions and examples
        "level2_purpose",
        "level3_purpose",
        "widget_purpose",
        "dashboard_purpose",
        "dashboard_parts",
        "dashboard_tiles",
        "dashboard_example",
        "dashboard_tiles_from_redis",
        "tts_script_structure",
        "widget_weather",
        "widget_light",
        "widget_climate",
        "widget_football",
        "widget_music",
        "widget_documents",
        # device-level design advice not visible in the response
        "ux_one_handed",
        "ui_touch_friendly",
    }
)

GUIDANCE_RULES = frozenset(
    {
        # cmd_prompt
        "search_when_in_doubt",
        "request_only_required_parameters",
        "ask_when_uncertain",
        "confirm_important_operations",
        "watch_current_date",
        # format_ui_answer
        "ui_progressive_disclosure",
        "ui_contextual_actions",
        "ui_scannable_content",
        "ui_open_with_impact",
        "ui_information_flow",
        "ui_anticipate_next_steps",
        "ui_visual_grouping",
        "ui_interactive_first",
        "ui_no_walls_of_text",
        "ui_rich_depth",
        "ui_text_answers",
        "ui_images",
        "markdown_full_rendering",
        "markdown_headings_for_sections",
        "markdown_mermaid_for_flows",
        "ux_active_voice",
        "ux_user_context",
        # level2 / level3 / plain
        "markdown_structure",
        "markdown_no_excess",
        "level2_scannable",
        "level2_paragraphs_and_lists",
        "level2_buttons_relevant",
        "quick_actions_action_labels",
        "level3_text_complements_widget",
        "level3_concise_text",
        "level3_widget_when_useful",
        "level3_buttons_relate",
        "plain_concise",
        "plain_natural_paragraphs",
        "plain_straightforward",
        # dashboard / widget / tts
        "dashboard_common_scenarios",
        "dashboard_use_user_input",
        "dashboard_simple_language",
        "widget_pick_best",
        "widget_complete_data",
        "widget_self_contained",
        "tts_natural_short",
    }
)


# Rules that concern only some requests: when the condition is false for the turn,
# the judge must answer "not applicable" instead of "violated" (false reds, ARCHIE-213)
RULE_CONDITIONS = {
    "load_matching_skill": "a skill in the agent catalog clearly matches the user's request",
    "gmail_read_only": "the user asks about emails",
    "defer_actions_with_cron": "the user asks to do an action later or on a schedule",
    "form_before_create": "the user asks to create a calendar event, a note or an email",
    "smarthome_act_without_confirmation": "the user asks to control a smart-home device now",
    "smarthome_infer_parameters": "the user asks to control a smart-home device and leaves a parameter open",
    "never_offer_search": "the assistant asks the user whether to search, or offers to search",
    "search_for_live_data": "the request needs live, recent or date-bound facts",
    "respond_in_message_language": 'the user message is more than an interjection like "Mhm" or "ok" and has a clear language',
    "answer_from_knowledge_cases": "the assistant answered without calling a tool",
}

# User messages that only acknowledge the previous answer: no request, so the
# Stage 1 tool and skill rules have nothing to judge
_ACKNOWLEDGEMENTS = frozenset(
    "mhm hmm hm ok okay yes no yep thanks thank you cool nice great sure "
    "угу ага ок окей да нет спасибо благодарю понял поняла ясно хорошо ладно круто супер "
    "ну отлично здорово понятно".split()
)


def is_acknowledgement(text: str) -> bool:
    """True for a message like "Mhm." or "ок, спасибо" that carries no request."""
    words = re.findall(r"[\w']+", text.lower())
    return (
        0 < len(words) <= 4
        and "?" not in text
        and all(w in _ACKNOWLEDGEMENTS for w in words)
    )


@dataclass(frozen=True)
class SpecRule:
    """One rule the judge checks."""

    spec: str
    rule_id: str
    text: str
    guidance: bool = False

    @property
    def condition(self) -> str | None:
        return RULE_CONDITIONS.get(self.rule_id)

    @property
    def key(self) -> str:
        return f"{self.spec}:{self.rule_id}"


@functools.cache
def _compiler() -> PromptCompiler:
    return PromptCompiler(load_library())


def _library() -> PromptLibrary:
    return _compiler().library


def _is_templated(rule_id: str) -> bool:
    rule = _library().rules[rule_id]
    return any("{{" in part or "{%" in part for part in [rule.text, *rule.items])


def specs_for_turn(content_format: str, trace_found: bool) -> list[str]:
    """Specs whose rules apply to a turn of this format."""
    specs = []
    # cmd_prompt rules are about tool and skill calls: judged only with the trace
    if content_format not in DIRECT_FORMATS and trace_found:
        specs.append(COMMAND_SPEC)
    specs.append(ASSISTANT_SPEC)
    format_spec = PromptBuilder.format_spec_name(content_format)
    if format_spec in _library().specs:
        specs.append(format_spec)
    return specs


def _applies(expression: str | None, content_format: str) -> bool:
    """A `when` condition for this format; conditions on the user state are unknown."""
    if expression is None:
        return True
    if "state" in expression:
        return False
    return _compiler().check(expression, response_format=content_format)


@functools.cache
def spec_rules(spec: str, content_format: str) -> tuple[SpecRule, ...]:
    """Judged rules of one spec for a format, in spec order, each rule once."""
    rules: list[SpecRule] = []
    seen: set[str] = set()
    for section in _library().specs[spec].sections:
        if not _applies(section.when, content_format):
            continue
        for ref in section.rules + section.fallback:
            rule_id = ref.rule
            if (
                rule_id in seen
                or rule_id in EXCLUDED_RULES
                or _is_templated(rule_id)
                or not _applies(ref.when, content_format)
            ):
                continue
            seen.add(rule_id)
            rules.append(
                SpecRule(
                    spec=spec,
                    rule_id=rule_id,
                    text=_compiler().render_rule(rule_id),
                    guidance=rule_id in GUIDANCE_RULES,
                )
            )
    return tuple(rules)


def rules_for_turn(
    content_format: str, trace_found: bool, user_message: str = ""
) -> list[SpecRule]:
    """Every judged rule of the turn's specs (outcome checks live in outcome.py)."""
    specs = specs_for_turn(content_format, trace_found)
    if is_acknowledgement(user_message):
        specs = [spec for spec in specs if spec != COMMAND_SPEC]
    rules: list[SpecRule] = []
    for spec in specs:
        rules += spec_rules(spec, content_format)
    return rules


def spec_versions(content_format: str, trace_found: bool) -> dict[str, str]:
    """Spec name -> fingerprint of the spec version the turn was judged against."""
    return {
        spec: _compiler().fingerprint(spec)
        for spec in specs_for_turn(content_format, trace_found)
    }
