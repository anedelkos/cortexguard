"""Prompt construction helpers for cloud remediation planning."""

from __future__ import annotations

from cortexguard.cloud.planner.llm_client import PlannerRequest


def _xml_block(tag: str, body: str, indent: int = 0) -> str:
    pad = "  " * indent
    return f"{pad}<{tag}>\n{body}\n{pad}</{tag}>"


def build_planner_prompt(request: PlannerRequest) -> str:
    parts: list[str] = []

    parts.append(
        "<system>\n"
        "You are a safety planner for a hardware device.\n"
        "\n"
        "HARD RULES:\n"
        "1. Only use actions listed in the capability catalog.\n"
        "2. Prefer safe, reversible actions over irreversible ones.\n"
        "3. If no safe plan is possible, set needs_human_review=true and explain in rationale.\n"
        "4. Do not invent capabilities.\n"
        "</system>"
    )

    ctx: list[str] = []

    anomaly_body = (
        f"  <key>{request.anomaly_key}</key>\n" f"  <severity>{request.severity}</severity>"
    )
    if request.anomaly_details:
        anomaly_body += f"\n  <details>\n{_indent(request.anomaly_details, 2)}\n  </details>"
    ctx.append(_xml_block("anomaly", anomaly_body))

    ctx.append(_xml_block("state", _indent(request.state_summary)))

    if request.reasoning_trace:
        ctx.append(_xml_block("edge_reasoning", _indent(request.reasoning_trace)))

    if request.last_actions:
        ctx.append(_xml_block("actions_tried", _indent(request.last_actions)))

    plan_body_parts: list[str] = []
    if request.current_plan_id:
        plan_body_parts.append(f"  <id>{request.current_plan_id}</id>")
    if request.current_step:
        plan_body_parts.append(f"  <current_step>{request.current_step}</current_step>")
    if request.current_plan_compact:
        plan_body_parts.append(f"  <plan>\n{_indent(request.current_plan_compact, 2)}\n  </plan>")
    if plan_body_parts:
        ctx.append(_xml_block("active_plan", "\n".join(plan_body_parts)))

    if request.scene_graph:
        ctx.append(_xml_block("vision", _indent(request.scene_graph)))

    if request.system_health:
        ctx.append(_xml_block("system_health", _indent(request.system_health)))

    if request.remediation_policy:
        ctx.append(_xml_block("remediation_policy", _indent(request.remediation_policy)))

    if request.retrieved_summaries:
        lines = "\n".join(
            f"  <incident>\n{_indent(s, 2)}\n  </incident>" for s in request.retrieved_summaries
        )
        ctx.append(_xml_block("retrieved_incidents", lines))

    ctx.append(_xml_block("capability_catalog", _indent(request.capability_catalog_json)))

    parts.append(_xml_block("context", "\n\n".join(ctx)))
    parts.append("<instruction>\nGenerate a remediation plan as structured JSON.\n</instruction>")

    return "\n\n".join(parts)


def _indent(text: str, level: int = 1) -> str:
    pad = "  " * level
    return "\n".join(f"{pad}{line}" for line in text.split("\n"))
