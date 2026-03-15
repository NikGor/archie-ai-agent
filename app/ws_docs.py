"""Auto-generated WebSocket protocol documentation endpoint."""

import json
from archie_shared.chat.models import ChatMessage, ChatRequest
from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from .models.ws_models import StatusUpdate


router = APIRouter()


def _resolve_ref(ref: str, defs: dict) -> dict:
    name = ref.split("/")[-1]
    return defs.get(name, {})


def _type_str(field_schema: dict, defs: dict) -> str:
    if "$ref" in field_schema:
        field_schema = _resolve_ref(field_schema["$ref"], defs)

    if "anyOf" in field_schema:
        parts = []
        for part in field_schema["anyOf"]:
            if "$ref" in part:
                parts.append(part["$ref"].split("/")[-1])
            elif part.get("type") == "null":
                pass  # skip null, shown via optional badge
            else:
                parts.append(part.get("type", "any"))
        return " | ".join(parts) if parts else "any"

    if "enum" in field_schema:
        return " | ".join(f'"{v}"' for v in field_schema["enum"])

    if "allOf" in field_schema and len(field_schema["allOf"]) == 1:
        return _type_str(field_schema["allOf"][0], defs)

    t = field_schema.get("type", "")
    if t == "array":
        items = field_schema.get("items", {})
        return f"{_type_str(items, defs)}[]"
    if t == "object":
        return "object"
    return t or "any"


def _render_fields(schema: dict) -> str:
    defs = schema.get("$defs", {})
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    if not properties:
        return "<tr><td colspan='4'><em>No fields</em></td></tr>"

    rows = []
    for name, field in properties.items():
        resolved = _resolve_ref(field["$ref"], defs) if "$ref" in field else field
        type_label = _type_str(field, defs)
        description = resolved.get("description", field.get("description", ""))
        default = field.get("default", resolved.get("default", "—"))
        req = name in required

        badge = (
            '<span class="badge req">required</span>'
            if req
            else '<span class="badge opt">optional</span>'
        )
        default_cell = f"<code>{json.dumps(default)}</code>" if default != "—" else "—"

        rows.append(
            f"<tr><td><code>{name}</code></td>"
            f"<td><code>{type_label}</code></td>"
            f"<td>{badge}</td>"
            f"<td>{description}</td>"
            f"<td>{default_cell}</td></tr>"
        )

    return "\n".join(rows)


def _section(title: str, subtitle: str, envelope: str, schema: dict, color: str) -> str:
    fields_html = _render_fields(schema)
    return f"""
    <section>
        <div class="section-header" style="border-left: 4px solid {color}">
            <h2>{title}</h2>
            <div class="envelope">Envelope: <code>{envelope}</code></div>
            <p class="subtitle">{subtitle}</p>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Field</th><th>Type</th><th></th><th>Description</th><th>Default</th>
                </tr>
            </thead>
            <tbody>{fields_html}</tbody>
        </table>
    </section>
    """


def _build_html() -> str:
    chat_request_schema = ChatRequest.model_json_schema()
    chat_message_schema = ChatMessage.model_json_schema()
    status_update_schema = StatusUpdate.model_json_schema()

    s_request = _section(
        "1 · Client → Server",
        "Send once after connecting. The server processes the request and streams status updates.",
        "ChatRequest (raw JSON, no wrapper)",
        chat_request_schema,
        "#4f9cf9",
    )

    s_status = _section(
        "2 · Server → Client: status",
        "Zero or more messages while the agent is working. Stream these to show progress.",
        '{"type": "status", ...StatusUpdate fields}',
        status_update_schema,
        "#f9a84f",
    )

    s_final = _section(
        "3 · Server → Client: final",
        "Exactly one message when processing is complete. Contains the full ChatMessage.",
        '{"type": "final", "data": ChatMessage}',
        chat_message_schema,
        "#4fcf7a",
    )

    error_section = """
    <section>
        <div class="section-header" style="border-left: 4px solid #f94f4f">
            <h2>Error · Server → Client</h2>
            <div class="envelope">Envelope: <code>{"type": "error", "message": "&lt;string&gt;"}</code></div>
            <p class="subtitle">Sent on validation or unhandled exception. Connection closes after.</p>
        </div>
        <table>
            <thead><tr><th>Field</th><th>Type</th><th></th><th>Description</th><th>Default</th></tr></thead>
            <tbody>
                <tr><td><code>type</code></td><td><code>"error"</code></td><td><span class="badge req">required</span></td><td>Message discriminator</td><td>—</td></tr>
                <tr><td><code>message</code></td><td><code>string</code></td><td><span class="badge req">required</span></td><td>Human-readable error description</td><td>—</td></tr>
            </tbody>
        </table>
    </section>
    """

    flow_html = """
    <section class="flow">
        <h2>Protocol Flow</h2>
        <div class="flow-diagram">
            <div class="flow-row"><span class="actor client">Client</span><span class="arrow">──── ChatRequest ────────────────────────────────▶</span><span class="actor server">Server</span></div>
            <div class="flow-row"><span class="actor client">Client</span><span class="arrow">◀──── {type:"status", step, message, detail} ────</span><span class="actor server">Server</span></div>
            <div class="flow-row"><span class="actor client">Client</span><span class="arrow">◀──── {type:"status", ...} ─────────────────────</span><span class="actor server">Server</span></div>
            <div class="flow-row"><span class="actor client">Client</span><span class="arrow">◀──── {type:"final", data: ChatMessage} ─────────</span><span class="actor server">Server</span></div>
            <div class="flow-note">On error: {type:"error", message} — connection closed</div>
        </div>
    </section>
    """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Archie WebSocket API — /ws_chat</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #0f1117; color: #e2e8f0; padding: 2rem; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 0.25rem; }}
  .meta {{ color: #64748b; font-size: 0.9rem; margin-bottom: 2rem; }}
  h2 {{ font-size: 1.1rem; font-weight: 600; color: #f1f5f9; margin-bottom: 0.4rem; }}
  section {{ margin-bottom: 2.5rem; }}
  .section-header {{ padding: 1rem 1.2rem; background: #1e2433; border-radius: 8px 8px 0 0; }}
  .envelope {{ font-size: 0.8rem; color: #94a3b8; margin-top: 0.3rem; }}
  .subtitle {{ font-size: 0.85rem; color: #64748b; margin-top: 0.4rem; }}
  table {{ width: 100%; border-collapse: collapse; background: #161b2e; border-radius: 0 0 8px 8px; overflow: hidden; }}
  th {{ text-align: left; padding: 0.6rem 1rem; font-size: 0.75rem; text-transform: uppercase; color: #64748b; background: #1a2035; }}
  td {{ padding: 0.65rem 1rem; font-size: 0.85rem; border-top: 1px solid #1e2a3a; vertical-align: top; }}
  code {{ font-family: "JetBrains Mono", "Fira Code", monospace; font-size: 0.82rem; color: #7dd3fc; }}
  .badge {{ display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.72rem; font-weight: 600; }}
  .badge.req {{ background: #1e3a4a; color: #38bdf8; }}
  .badge.opt {{ background: #1e2a1e; color: #4ade80; }}
  .flow {{ background: #1e2433; border-radius: 8px; padding: 1.2rem 1.5rem; }}
  .flow-diagram {{ margin-top: 0.8rem; font-family: monospace; font-size: 0.85rem; }}
  .flow-row {{ display: flex; align-items: center; gap: 0.5rem; margin: 0.4rem 0; }}
  .actor {{ display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-weight: 600; font-size: 0.8rem; min-width: 60px; text-align: center; }}
  .actor.client {{ background: #1e3a5f; color: #4f9cf9; }}
  .actor.server {{ background: #1e3a3a; color: #4fcf9f; }}
  .arrow {{ color: #475569; flex: 1; }}
  .flow-note {{ margin-top: 0.6rem; font-size: 0.8rem; color: #94a3b8; padding-left: 0.5rem; }}
</style>
</head>
<body>
<h1>WebSocket API — <code>/ws_chat</code></h1>
<p class="meta">Auto-generated from Pydantic models · refreshes on server restart</p>

{flow_html}
{s_request}
{s_status}
{s_final}
{error_section}
</body>
</html>"""


@router.get("/ws_docs", response_class=HTMLResponse, include_in_schema=False)
async def ws_docs():
    """Auto-generated WebSocket protocol documentation."""
    return _build_html()
