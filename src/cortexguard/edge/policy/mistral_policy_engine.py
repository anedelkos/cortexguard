from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from uuid import uuid4

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from cortexguard.core.interfaces.base_policy_engine import BasePolicyEngine
from cortexguard.edge.models.agent_tool_call import AgentToolCall
from cortexguard.edge.models.anomaly_event import AnomalyEvent
from cortexguard.edge.models.plan import PlanStep, StepStatus
from cortexguard.edge.models.remediation_policy import PolicySource, RemediationPolicy
from cortexguard.edge.models.state_estimate import StateEstimate

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"


class MistralLLMPolicyEngine(BasePolicyEngine):
    """
    A policy engine implementation using a large language model (Mistral 7B)
    to dynamically generate remediation steps, integrated with existing domain models.

    This version includes the 'action_catalog_json' in the prompt to provide the LLM
    with a list of available action primitives (capabilities and arguments).
    """

    def __init__(self, use_mock: bool = False, model_id: str = _DEFAULT_MODEL_ID):
        self._use_mock = use_mock
        self._model_name = model_id
        # Verify CUDA availability based on your installed PyTorch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None

        if not self._use_mock:
            if self.device == "cpu":
                logger.warning(
                    "CUDA not available. Loading Mistral on CPU will be extremely slow. "
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
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=512,  # Limit the response length
                do_sample=False,  # Use greedy decoding for structured output
                temperature=0.1,  # Low temperature for deterministic JSON output
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # 3. Decode and clean the output
        generated_text: str = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # We strip the input prompt (the entire [INST]...[/INST] block)
        # from the generated text, as Mistral models append the answer to the prompt.
        raw_response = generated_text.replace(prompt, "", 1).strip()

        return raw_response

    def _mock_llm_call(self, prompt: str, event: AnomalyEvent, action_catalog_json: str) -> str:
        """
        Mocks the LLM's response for demonstration and testing.
        This mock output now STRICTLY follows the required JSON schema
        (nested 'action' object with 'action_name').
        The trace now reflects a simple rule summary.
        """
        logger.info("Performing mock LLM inference (replace with self._model.generate()).")
        logger.debug(f"Action Catalog received in mock: {action_catalog_json}")

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
        """Parses the LLM's complex JSON output and converts it into the final RemediationPolicy."""
        try:
            # Simple cleanup for LLM generation artifacts (if they ignore instruction #1)
            clean_text = raw_response.strip()
            # Robustly remove common markdown/code block wrappers
            if clean_text.startswith(("```json", "```")):
                clean_text = clean_text.removeprefix("```json").removeprefix("```")
            if clean_text.endswith("```"):
                clean_text = clean_text.removesuffix("```")
            data = json.loads(clean_text.strip())

            # 1. Convert step dicts into rich PlanStep objects
            corrective_steps: list[PlanStep] = []
            for i, step_dict in enumerate(data.get("corrective_steps", [])):
                action_dict = step_dict.get("action", {})
                action_name = action_dict.get("action_name", "NO_OP")
                arguments = action_dict.get("arguments", {})

                # Basic validation for required fields from LLM
                if not action_name or not isinstance(arguments, dict):
                    logger.warning(f"LLM produced malformed action in step {i}: {step_dict}")
                    # In case of malformed data, we skip this step but continue with others
                    continue

                # Construct the AgentToolCall object
                agent_tool_call = AgentToolCall(
                    action_name=action_name,
                    arguments=arguments,
                )

                # Construct the PlanStep object
                step = PlanStep(
                    id=f"plan-step-{uuid4().hex[:6]}-{i}",
                    description=step_dict.get("description", "No description provided"),
                    action=agent_tool_call,
                    status=StepStatus.PENDING,
                )
                corrective_steps.append(step)

            # 2. Construct the final RemediationPolicy object
            return RemediationPolicy(
                # Policy Fields from LLM Output
                policy_id="llm-" + str(uuid4()),
                source=PolicySource.LLM,
                reasoning_trace=data.get("reasoning_trace", "No reasoning provided by LLM."),
                risk_assessment=data.get("risk_assessment", "UNKNOWN"),
                corrective_steps=corrective_steps,
                escalation_required=data.get("escalation_required", False),
                trigger_event=event,
                created_at=datetime.now(UTC),
            )

        except Exception as e:
            logger.error(
                f"Failed to parse LLM JSON response or construct policy: {e}. Raw text: {raw_response}"
            )
            # Fallback: Create a fail-safe policy
            return RemediationPolicy(
                policy_id=str(uuid4()),
                source=PolicySource.FALLBACK,
                trigger_event=event,
                reasoning_trace=f"CRITICAL PARSING FAILURE: LLM response was invalid JSON. Error: {e}",
                risk_assessment="HIGH - System safety cannot be guaranteed.",
                escalation_required=True,
                corrective_steps=[
                    PlanStep(
                        id=f"step-err-{uuid4().hex[:6]}",
                        description="LLM failed to generate valid policy. Triggering emergency system halt.",
                        action=AgentToolCall(
                            action_name="EMERGENCY_SHUTDOWN",
                            arguments={"scope": "local", "tool_id": "safety_manager"},
                        ),
                    )
                ],
            )

    def _format_prompt(
        self,
        event: AnomalyEvent,
        context: StateEstimate,
        action_catalog_json: str,
        vision_context: str | None = None,
    ) -> str:
        """
        Formats the remediation agent prompt with strict rules,
        hallucination guardrails, and a canonical JSON output example.
        """

        json_schema = """
        {
          "reasoning_trace": "Policy Compliance Summary: Brief, factual summary of the rule set applied (e.g., 'Rule 1.A/B was TRUE'). MUST NOT be an open-ended explanation or Chain-of-Thought.",
          "risk_assessment": "One of: 'LOW', 'MEDIUM', 'HIGH', 'UNKNOWN'",
          "escalation_required": "Boolean true/false",
          "corrective_steps": [
            {
              "description": "Short natural-language description",
              "action": {
                "action_name": "One valid action name from the Action Catalog",
                "arguments": {}
              }
            }
          ]
        }
        """

        anomaly_context_json = event.model_dump_json(indent=2)
        state_context_json = context.model_dump_json(indent=2)

        # --- SYSTEM INSTRUCTIONS (Rewritten for test-stable determinism) ---
        system_instruction = (
            "You are the Policy Selector Agent. You MUST output ONLY valid JSON matching the schema. "
            "NO natural language outside the JSON. NO commentary.\n\n"
            "=========================\n"
            "   VALID ACTION NAMES\n"
            "=========================\n"
            "The ONLY action names you are allowed to output are EXACTLY the ones included in the Action Catalog. "
            "Any action name not present in the catalog is strictly forbidden.\n\n"
            "Explicitly forbidden action names include (but are not limited to):\n"
            "- EMERGENCY_SHUTDOWN\n"
            "- SHUTDOWN\n"
            "- STOP_ALL\n"
            "- ANY name not present in the catalog\n\n"
            "=========================\n"
            "   RISK ASSESSMENT RULE\n"
            "=========================\n"
            "risk_assessment MUST be exactly one of:\n"
            "['LOW', 'MEDIUM', 'HIGH', 'UNKNOWN']\n"
            "NO extra wording. NO additional text.\n\n"
            "=========================\n"
            "   COOLING DECISION RULES\n"
            "=========================\n"
            "TEMP_HIGH anomaly logic:\n"
            "PRIMARY MITIGATION succeeds ONLY if BOTH are true:\n"
            "  A) The action 'SET_POWER_LEVEL' EXISTS in the Action Catalog.\n"
            "  B) Cooling_Unit_001 has power_level < 1.0 in the System State.\n\n"
            "If BOTH A and B are TRUE:\n"
            "  → Step 1 MUST be:\n"
            '       {"action_name": "SET_POWER_LEVEL", '
            '"arguments": {"device_id": "Cooling_Unit_001", "power_level": 1.0}}\n'
            "  → escalation_required = false\n\n"
            "If EITHER A or B is FALSE:\n"
            "  → PRIMARY mitigation FAILS\n"
            "  → You MUST produce the following EMERGENCY FALLBACK sequence:\n"
            "       Step 1: EMERGENCY_STOP on Grill_Station_1\n"
            '       Step 2: SEND_ALERT with target="supervisor"\n'
            "  → escalation_required = true\n\n"
            "Rules:\n"
            "- NEVER call SET_POWER_LEVEL if it is NOT in the Action Catalog.\n"
            "- NEVER output a power_level > 1.0.\n"
            "- NEVER emit actions on sensors.\n"
            "- IGNORE irrelevant actions.\n\n"
            "=========================\n"
            "   OUTPUT FORMAT (MANDATORY)\n"
            "=========================\n"
            "Output MUST be valid JSON. No prefix. No suffix. No natural language. "
            "Only a single JSON object that satisfies the schema."
        )

        user_query_content = f"""
            --- ANOMALY DETAILS ---
            {anomaly_context_json}

            --- SYSTEM STATE CONTEXT ---
            {state_context_json}

            --- AVAILABLE ACTION CATALOG ---
            {action_catalog_json}

            Analyze the anomaly and produce a remediation plan STRICTLY following the decision hierarchy.
            Output ONLY valid JSON following this schema:
            {json_schema}
        """

        prompt = f"""[INST]
        {system_instruction}

        {user_query_content.strip()}
        [/INST]"""

        return prompt

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
        prompt = self._format_prompt(event, context, action_catalog_json, vision_context)

        if self._use_mock:
            # Pass the action_catalog_json to the mock implementation
            raw_response = self._mock_llm_call(prompt, event, action_catalog_json)
        else:
            # Use an executor to run the synchronous LLM call without blocking the event loop
            loop = asyncio.get_running_loop()
            raw_response = await loop.run_in_executor(None, self._run_real_llm_call, prompt)

        return self._parse_llm_response(raw_response, event)
