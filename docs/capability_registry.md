# Capability Registry

The `CapabilityRegistry` is the central catalog of actions the edge system can perform. It defines what the `PolicyAgent` (LLM) can call, what arguments are valid, and the associated risk level. It is also used by `StepExecutor` and `Arbiter` to validate commands before execution.

---

## Files

| File | Purpose |
|---|---|
| `src/cortexguard/edge/models/capability_registry.yaml` | Capability definitions (source of truth) |
| `src/cortexguard/edge/models/capability_registry.py` | Loads, validates, and exposes capabilities at runtime |

---

## How It Works

At startup, `CapabilityRegistry.load_from_yaml()` reads the YAML file and validates each entry into a `FunctionSchema`. The registry is then injected into `PolicyAgent`, `StepExecutor`, and `Arbiter`.

- **`PolicyAgent`** calls `get_llm_tool_catalog()` to pass the full catalog to the LLM as available tools.
- **`Arbiter`** calls `get_function_schema()` to check that a proposed action exists before allowing it.
- **`StepExecutor`** calls `validate_call()` to verify argument types/values against the JSON Schema before executing a step.

---

## Schema: FunctionSchema

Each capability entry in the YAML has the following fields:

| Field | Type | Description |
|---|---|---|
| `description` | string | Shown to the LLM to explain the capability's purpose |
| `parameters` | JSON Schema object | Argument validation schema (properties + required) |
| `risk_level` | `LOW \| MEDIUM \| HIGH \| E-STOP` | Operational risk; used by the LLM to prefer safer paths |
| `pre_conditions` | list[string] | Conditions that must be true before calling this capability |
| `post_effects` | list[string] | Expected state changes after successful execution |

---

## Risk Levels

| Level | Meaning |
|---|---|
| `LOW` | Safe to execute autonomously |
| `MEDIUM` | Involves heat, motion, or brief unavailability — use with care |
| `HIGH` | Reserved for dangerous or destructive operations |
| `E-STOP` | Last resort; cuts power and halts all operations immediately |

---

## Registered Capabilities

### Operational Capabilities

| Capability | Risk | Description |
|---|---|---|
| `SLICE_ITEM` | LOW | Slices a target item using a cutting tool |
| `GRILL_ITEM` | MEDIUM | Processes an item on a heating station for a defined duration |
| `PLACE_ITEM` | LOW | Picks up and places an item at a target location |
| `ALIGN_AND_STACK` | LOW | Stacks two components for assembly |
| `DELIVER_ORDER` | MEDIUM | Transfers a completed item to the output area |

### Remediation Capabilities

| Capability | Risk | Description |
|---|---|---|
| `EMERGENCY_STOP` | E-STOP | Cuts power to a device on extreme hazard detection |
| `RESET_DEVICE` | MEDIUM | Power-cycles a non-critical device to clear transient errors |
| `SEND_ALERT` | LOW | Notifies a human supervisor when self-remediation fails |

---

## Adding a New Capability

Add an entry to `capability_registry.yaml`:

```yaml
MY_NEW_ACTION:
  description: "What this action does, written for the LLM."
  parameters:
    type: object
    properties:
      param_name: {type: string}
      tool_id:
        type: string
        enum: ["Tool_A", "Tool_B"]
    required: [param_name, tool_id]
  risk_level: LOW
  pre_conditions:
    - "Some precondition"
  post_effects:
    - "Some resulting state change"
```

No code changes required — the registry is loaded from YAML at startup.
