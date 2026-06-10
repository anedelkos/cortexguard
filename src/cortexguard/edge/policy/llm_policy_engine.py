from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

from cortexguard.core.interfaces.base_policy_engine import BasePolicyEngine
from cortexguard.edge.models.agent_tool_call import AgentToolCall
from cortexguard.edge.models.anomaly_event import AnomalyEvent
from cortexguard.edge.models.plan import PlanStep, StepStatus
from cortexguard.edge.models.remediation_policy import PolicySource, RemediationPolicy
from cortexguard.edge.models.state_estimate import StateEstimate

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID: str = os.getenv("POLICY_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
_STRUCTURAL_DELIMITERS = ("[/INST]", "[INST]", "<|im_start|>", "<|im_end|>")


def _load_action_catalog(action_catalog_json: str) -> list[dict[str, Any]]:
    try:
        return cast("list[dict[str, Any]]", json.loads(action_catalog_json))
    except (json.JSONDecodeError, TypeError):
        return []


class LLMPolicyEngine(BasePolicyEngine):
    """
    A policy engine implementation using a large language model (Qwen2.5-7B)
    to dynamically generate remediation steps, integrated with existing domain models.

    This version includes the 'action_catalog_json' in the prompt to provide the LLM
    with a list of available action primitives (capabilities and arguments).
    """

    def __init__(
        self, use_mock: bool = False, model_id: str = _DEFAULT_MODEL_ID, llm_timeout_s: float = 30.0
    ):
        self._use_mock = use_mock
        self._model_name = model_id
        self._llm_timeout_s = llm_timeout_s
        self.device: str = "cpu"
        self.tokenizer: Any = None
        self.model: Any = None

        if not self._use_mock:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            # Verify CUDA availability based on your installed PyTorch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            if self.device == "cpu":
                logger.warning(
                    "CUDA not available. Loading model on CPU will be extremely slow. "
                    "Please ensure PyTorch/CUDA are linked correctly."
                )

            logger.info(f"Loading LLM {model_id} onto {self.device}...")

            # --- Configuration for RTX 3060 VRAM Efficiency (4-bit Quantization) ---
            bnb_config = BitsAndBytesConfig(  # type: ignore [no-untyped-call]
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",  # Use NF4 for best results
                bnb_4bit_compute_dtype=torch.bfloat16,  # Recommended compute dtype
            )

            self.tokenizer = AutoTokenizer.from_pretrained(  # type: ignore [no-untyped-call]
                model_id, revision="main"
            )  # nosec B615 # type: ignore [no-untyped-call]
            self.model = AutoModelForCausalLM.from_pretrained(  # nosec B615 # type: ignore [no-untyped-call]
                model_id,
                revision="main",
                device_map="auto",  # Automatically maps layers across GPU/CPU
                # Apply quantization only if running on CUDA
                quantization_config=bnb_config if self.device == "cuda" else None,
            )
            logger.info("LLM loading complete.")
        else:
            logger.warning("LLM is running in MOCK mode.")

    def model_name(self) -> str:
        return self._model_name

    def _run_real_llm_call(self, prompt: str) -> str:
        """
        Executes the LLM inference using the loaded model.
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model and tokenizer must be loaded for real inference.")

        # 1. Tokenize the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # 2. Generate the response
        import torch

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=512,  # Limit the response length
                do_sample=False,  # Use greedy decoding for structured output
                temperature=0.1,  # Low temperature for deterministic JSON output
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # 3. Decode only the newly generated tokens (skip the input prompt)
        generated_ids = output[0][input_ids.shape[1] :]
        raw_response = cast(
            str, self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        )

        return raw_response

    def _mock_llm_call(self, prompt: str, event: AnomalyEvent) -> str:
        """
        Mocks the LLM's response for demonstration and testing.
        This mock output now STRICTLY follows the required JSON schema
        (nested 'action' object with 'action_name').
        The trace now reflects a simple rule summary.
        """
        logger.info("Performing mock LLM inference (replace with self._model.generate()).")

        context_data = {
            "temperature": event.metadata.get("current_temp", "N/A"),
            "target_temp": event.metadata.get("target_temp", "N/A"),
        }

        if event.key == "TEMP_HIGH":
            return json.dumps(
                {
                    "reasoning_trace": (
                        "Rule 1.A/B was TRUE: Anomaly is TEMP_HIGH, SET_POWER_LEVEL exists, and "
                        "Cooling_Unit_001 is below max power. Primary mitigation engaged."
                    ),
                    "risk_assessment": "MEDIUM",
                    "escalation_required": False,
                    "corrective_steps": [
                        {
                            "description": "Engage the emergency cooling unit at 100% power.",
                            "action": {
                                "action_name": "SET_POWER_LEVEL",
                                "arguments": {"device_id": "Cooling_Unit_001", "level": 1.0},
                            },
                        },
                        {
                            "description": "Send a critical alert notification to the maintenance supervisor.",
                            "action": {
                                "action_name": "SEND_NOTIFICATION",
                                "arguments": {
                                    "recipient": "supervisor",
                                    "message": f"Critical temp anomaly: {context_data['temperature']}C.",
                                },
                            },
                        },
                    ],
                }
            )
        else:
            return json.dumps(
                {
                    "reasoning_trace": "Rule 2 was applied: Event is unknown/low-priority. Standard logging initiated.",
                    "risk_assessment": "LOW",
                    "escalation_required": True,
                    "corrective_steps": [
                        {
                            "description": f"Log unknown anomaly: {event.key}.",
                            "action": {
                                "action_name": "LOG_EVENT",
                                "arguments": {"event_key": event.key},
                            },
                        }
                    ],
                }
            )

    def _parse_llm_response(self, raw_response: str, event: AnomalyEvent) -> RemediationPolicy:
        """Parses the LLM response into a RemediationPolicy.

        Handles three formats:
        1. The original policy JSON schema (with corrective_steps, reasoning_trace, etc.)
        2. Single function call: {"name": "...", "arguments": {...}}
        3. Multiple JSON objects in text output (supports text+tool calls mixed)
        """
        try:
            clean_text = raw_response.strip()
            if clean_text.startswith(("```json", "```")):
                clean_text = clean_text.removeprefix("```json").removeprefix("```")
            if clean_text.endswith("```"):
                clean_text = clean_text.removesuffix("```")

            data = json.loads(clean_text.strip())

            # Handle single function call format: {"name": "ACTION", "arguments": {...}}
            if "name" in data and "arguments" in data:
                return RemediationPolicy(
                    policy_id="llm-" + str(uuid4()),
                    source=PolicySource.LLM,
                    reasoning_trace=f"Tool call: {data['name']}",
                    risk_assessment="MEDIUM",
                    corrective_steps=[
                        PlanStep(
                            id=f"plan-step-{uuid4().hex[:6]}-0",
                            description=f"Execute {data['name']} remediation.",
                            action=AgentToolCall(
                                action_name=data["name"],
                                arguments=data["arguments"],
                            ),
                            status=StepStatus.PENDING,
                        )
                    ],
                    escalation_required=False,
                    trigger_event=event,
                    created_at=datetime.now(UTC),
                )

            # Handle the original RemediationPolicy JSON schema
            corrective_steps: list[PlanStep] = []
            for i, step_dict in enumerate(data.get("corrective_steps", [])):
                action_dict = step_dict.get("action", {})
                action_name = action_dict.get("action_name", "NO_OP")
                arguments = action_dict.get("arguments", {})

                if not action_name or not isinstance(arguments, dict):
                    logger.warning(f"LLM produced malformed action in step {i}: {step_dict}")
                    continue

                step = PlanStep(
                    id=f"plan-step-{uuid4().hex[:6]}-{i}",
                    description=step_dict.get("description", "No description provided"),
                    action=AgentToolCall(action_name=action_name, arguments=arguments),
                    status=StepStatus.PENDING,
                )
                corrective_steps.append(step)

            return RemediationPolicy(
                policy_id="llm-" + str(uuid4()),
                source=PolicySource.LLM,
                reasoning_trace=data.get("reasoning_trace", "No reasoning provided by LLM."),
                risk_assessment=data.get("risk_assessment", "UNKNOWN"),
                corrective_steps=corrective_steps,
                escalation_required=data.get("escalation_required", False),
                trigger_event=event,
                created_at=datetime.now(UTC),
            )

        except (json.JSONDecodeError, ValueError):
            # Text with embedded tool calls, try extracting JSON objects
            return self._parse_tool_calls_from_text(raw_response, event)

    def _parse_tool_calls_from_text(self, text: str, event: AnomalyEvent) -> RemediationPolicy:
        """Extract tool call JSON objects from free-form text output.

        Handles Qwen's <tool_call>...</tool_call> XML tag format and
        bare JSON objects on separate lines.
        """

        tool_calls = []
        seen_tool_json: set[str] = set()

        # Extract JSON from <tool_call>...</tool_call> tags first
        for match in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL):
            fragment = match.group(1).strip()
            try:
                obj = json.loads(fragment)
                if "name" in obj and "arguments" in obj:
                    key = json.dumps(obj, sort_keys=True)
                    if key not in seen_tool_json:
                        tool_calls.append(obj)
                        seen_tool_json.add(key)
                    continue
            except json.JSONDecodeError:
                pass

        # Also try line-by-line for bare JSON objects
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "name" in obj and "arguments" in obj:
                    key = json.dumps(obj, sort_keys=True)
                    if key not in seen_tool_json:
                        tool_calls.append(obj)
                        seen_tool_json.add(key)
                    continue
            except json.JSONDecodeError:
                pass

        reasoning_lines = [
            ln.strip()
            for ln in text.split("\n")
            if ln.strip()
            and re.search(r"<tool_call>", ln) is None
            and not ln.strip().startswith(("{", "["))
        ]
        reasoning_trace = " ".join(reasoning_lines).strip() if reasoning_lines else "LLM reasoning."

        if not tool_calls:
            logger.error(f"Failed to parse LLM response. Raw text: {text}")
            return RemediationPolicy(
                policy_id="llm-" + str(uuid4()),
                source=PolicySource.FALLBACK,
                trigger_event=event,
                reasoning_trace="CRITICAL PARSING FAILURE: LLM response was invalid JSON. Error: no valid JSON found in output",
                risk_assessment="HIGH - System safety cannot be guaranteed.",
                escalation_required=True,
                corrective_steps=[
                    PlanStep(
                        id=f"step-err-{uuid4().hex[:6]}",
                        description="LLM failed to generate valid policy. Triggering emergency system halt.",
                        action=AgentToolCall(
                            action_name="EMERGENCY_STOP",
                            arguments={"device_id": "safety_manager"},
                        ),
                    )
                ],
            )

        corrective_steps = []
        for i, call in enumerate(tool_calls):
            corrective_steps.append(
                PlanStep(
                    id=f"plan-step-{uuid4().hex[:6]}-{i}",
                    description=f"Execute {call['name']} remediation.",
                    action=AgentToolCall(
                        action_name=call["name"],
                        arguments=call["arguments"],
                    ),
                    status=StepStatus.PENDING,
                )
            )

        return RemediationPolicy(
            policy_id="llm-" + str(uuid4()),
            source=PolicySource.LLM,
            reasoning_trace=reasoning_trace,
            risk_assessment="MEDIUM",
            corrective_steps=corrective_steps,
            escalation_required=False,
            trigger_event=event,
            created_at=datetime.now(UTC),
        )

    @staticmethod
    def _convert_catalog_to_openai_tools(action_catalog_json: str) -> list[dict[str, Any]]:
        """Convert the project's action catalog format to OpenAI-compatible tools for apply_chat_template."""
        raw = _load_action_catalog(action_catalog_json)

        tools = []
        for entry in raw:
            func = {}
            if "action_name" in entry:
                func["name"] = entry["action_name"]
                func["description"] = entry.get("description", "")
                func["parameters"] = entry.get("parameters") or {"type": "object", "properties": {}}
            elif "function" in entry:
                func = entry["function"]
            elif "name" in entry:
                func = entry
            else:
                continue
            tools.append({"type": "function", "function": func})
        return tools

    def _format_prompt(
        self,
        event: AnomalyEvent,
        context: StateEstimate,
        action_catalog_json: str,
        active_plan_context: str = "",
        vision_context: str | None = None,
    ) -> str:
        """
        Formats the remediation agent prompt.
        Uses apply_chat_template with tools (function calling) when the tokenizer is available,
        falls back to manual string formatting for mock mode.
        """

        def _sanitize(text: str) -> str:
            for token in _STRUCTURAL_DELIMITERS:
                text = text.replace(token, "")
            return text

        anomaly_context_json = _sanitize(event.model_dump_json(indent=2))
        state_context_json = _sanitize(context.model_dump_json(indent=2))
        plan_context = _sanitize(active_plan_context)

        system_instruction = (
            "You are the Policy Selector Agent for a safety-critical kitchen automation system.\n\n"
            "Your task: Given an anomaly event and the current system state, select the appropriate "
            "remediation action from the available tools. You may only use the tools provided.\n\n"
            "Grounding: Never target sensors, only actuators.\n"
            "Think step by step, then call the correct tool from the ones available to you."
        )

        user_content = f"""Anomaly event:
{anomaly_context_json}

Current system state:
{state_context_json}

Active plan context:
{plan_context}

Analyze this anomaly and produce a remediation plan. Explain your reasoning, then call the appropriate tool(s)."""

        if self.tokenizer is not None and not self._use_mock:
            tools = self._convert_catalog_to_openai_tools(action_catalog_json)
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content.strip()},
            ]
            return cast(
                str,
                self.tokenizer.apply_chat_template(
                    messages, tools=tools, add_generation_prompt=True, tokenize=False
                ),
            )

        # Fallback for mock mode, manual chat template
        prompt = f"""<|im_start|>system
        {system_instruction}
        <|im_end|>
        <|im_start|>user
        {user_content.strip()}

Available actions:
{action_catalog_json}
        <|im_end|>
        <|im_start|>assistant"""

        return prompt

    @staticmethod
    def _extract_valid_action_names(action_catalog_json: str) -> set[str]:
        """Extract valid action names from the catalog JSON."""
        catalog = _load_action_catalog(action_catalog_json)
        names: set[str] = set()
        for entry in catalog:
            if "action_name" in entry:
                names.add(entry["action_name"])
            elif "name" in entry:
                names.add(entry["name"])
            elif isinstance(entry, dict) and "function" in entry:
                names.add(entry["function"].get("name", ""))
        return names

    async def generate_policy(
        self,
        event: AnomalyEvent,
        context: StateEstimate,
        action_catalog_json: str,
        active_plan_context: str,
        vision_context: str | None = None,
    ) -> RemediationPolicy:
        """
        Generates the policy by calling the LLM or using the mock.
        NOTE: The synchronous LLM call is offloaded to a thread pool executor
        to prevent blocking the asyncio loop.
        """
        # Pass the action_catalog_json to the prompt formatter
        prompt = self._format_prompt(
            event, context, action_catalog_json, active_plan_context, vision_context
        )

        if self._use_mock:
            raw_response = self._mock_llm_call(prompt, event)
        else:
            loop = asyncio.get_running_loop()
            raw_response = await asyncio.wait_for(
                loop.run_in_executor(None, self._run_real_llm_call, prompt),
                timeout=self._llm_timeout_s,
            )

        policy = self._parse_llm_response(raw_response, event)

        # Safety overrides: fix hallucinated/incorrect actions
        valid_actions = self._extract_valid_action_names(action_catalog_json)
        if not valid_actions:
            return policy

        # Remove hallucinated actions (not in catalog)
        filtered_steps = []
        for step in policy.corrective_steps:
            if step.action.action_name in valid_actions:
                filtered_steps.append(step)
            else:
                logger.warning(
                    "Safety override: removing hallucinated action '%s' from policy.",
                    step.action.action_name,
                )
        # TEMP_HIGH rule: if no cooling action in catalog, must have EMERGENCY_STOP
        if event.key == "TEMP_HIGH":
            has_cooling_action = any(
                isinstance(s.action.arguments, dict)
                and ("power_level" in s.action.arguments or "level" in s.action.arguments)
                for s in filtered_steps
            ) or any(
                "SET_POWER_LEVEL" in s.action.action_name.upper()
                or "COOL" in s.action.action_name.upper()
                for s in filtered_steps
            )
            has_cooling_tool = any(
                "POWER_LEVEL" in name.upper() or "COOL" in name.upper() for name in valid_actions
            )
            has_emergency_stop = any(
                s.action.action_name == "EMERGENCY_STOP" for s in filtered_steps
            )
            has_restart_action = any(s.action.action_name == "RESET_DEVICE" for s in filtered_steps)

            if has_restart_action and not has_cooling_action and not has_emergency_stop:
                logger.warning(
                    "Safety override: MODEL chose RESET_DEVICE for TEMP_HIGH with no cooling tool. "
                    "Replacing with EMERGENCY_STOP on heat source."
                )
                filtered_steps = [
                    s for s in filtered_steps if s.action.action_name != "RESET_DEVICE"
                ]
                has_emergency_stop = False

            if not has_emergency_stop and not has_cooling_tool:
                logger.warning(
                    "Safety override: forcing EMERGENCY_STOP for TEMP_HIGH with no cooling tool available."
                )
                device_mapping: dict[str, Any] = (
                    context.flags if isinstance(context.flags, dict) else {}
                ).get("device_mapping", {})
                heat_source_id = "Grill_Station_1"
                for dev_id, info in device_mapping.items():
                    info_dict = info or {}
                    if (
                        "heat" in str(info_dict.get("purpose", "")).lower()
                        or "grill" in dev_id.lower()
                    ):
                        heat_source_id = dev_id
                        break
                filtered_steps.insert(
                    0,
                    PlanStep(
                        id=f"plan-step-{uuid4().hex[:6]}-safety",
                        description="Emergency stop triggered by safety override.",
                        action=AgentToolCall(
                            action_name="EMERGENCY_STOP",
                            arguments={"device_id": heat_source_id},
                        ),
                        status=StepStatus.PENDING,
                    ),
                )
                if not any(s.action.action_name == "SEND_ALERT" for s in filtered_steps):
                    filtered_steps.append(
                        PlanStep(
                            id=f"plan-step-{uuid4().hex[:6]}-alert",
                            description="Alert supervisor about emergency stop.",
                            action=AgentToolCall(
                                action_name="SEND_ALERT",
                                arguments={
                                    "recipient": "supervisor",
                                    "message": (
                                        f"Safety override: EMERGENCY_STOP triggered for "
                                        f"TEMP_HIGH anomaly on sensor {event.metadata.get('sensor_id', 'unknown')}."
                                    ),
                                },
                            ),
                            status=StepStatus.PENDING,
                        ),
                    )
                policy.risk_assessment = "HIGH"

        policy.corrective_steps = filtered_steps

        if any(s.action.action_name == "EMERGENCY_STOP" for s in policy.corrective_steps):
            policy.risk_assessment = "HIGH"

        return policy
