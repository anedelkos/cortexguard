"""Prompt construction helpers for cloud remediation planning."""

from __future__ import annotations

from cortexguard.cloud.planner.llm_client import PlannerRequest

_HARD_RULES = """\
HARD RULES (non-negotiable):
1. Do not invent capabilities — only use actions listed in the capability catalog below.
2. Prefer safe, reversible actions over irreversible ones.
3. Output structured JSON only — no prose outside the JSON schema.
4. Do not recommend actions outside the provided capability catalog.
5. If no safe plan is possible, set needs_human_review=true and explain in rationale.
"""


def build_planner_prompt(request: PlannerRequest) -> str:
    retrieved_section = ""
    if request.retrieved_summaries:
        lines = "\n".join(f"  - {s}" for s in request.retrieved_summaries)
        retrieved_section = f"\nSimilar past incidents:\n{lines}\n"

    return (
        f"You are a cloud safety planner for a collaborative robot.\n"
        f"{_HARD_RULES}\n"
        f"Anomaly key: {request.anomaly_key}\n"
        f"Severity: {request.severity}\n"
        f"Escalation summary: {request.escalation_summary}\n"
        f"State summary: {request.state_summary}\n"
        f"{retrieved_section}"
        f"Capability catalog:\n{request.capability_catalog_json}\n"
        f"\nGenerate a remediation plan as structured JSON."
    )
