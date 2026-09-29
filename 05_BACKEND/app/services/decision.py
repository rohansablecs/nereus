import json
import os

from huggingface_hub import InferenceClient


MODEL = os.getenv(
    "NEREUS_AI_MODEL",
    "moonshotai/Kimi-K2-Instruct-0905",
)


class AIDecisionError(Exception):
    pass


def _build_prompt(decision_payload: dict) -> str:
    return f"""
You are the decision reasoning layer for NEREUS,
an intelligent bulk-cargo vessel chartering and procurement
decision-support system.

You MUST reason ONLY from the supplied NEREUS evidence.

Do not invent:
- freight rates
- vessel capabilities
- port constraints
- waiting times
- berth availability
- costs
- weather
- forecasts
- market conditions
- operational data

If information is missing, explicitly identify it as missing.

IMPORTANT:
A vessel marked FEASIBLE is physically compatible with the
evaluated constraints.

A vessel marked POTENTIALLY_FEASIBLE has unresolved physical
constraints and must NOT be presented as confirmed feasible.

A vessel marked INFEASIBLE must NOT be recommended.

The deterministic NEREUS engine remains authoritative for:
- physical feasibility
- route distance
- voyage calculations
- cost calculations
- berth compatibility
- operational observations

Your role is to:
1. interpret the evidence
2. explain the decision
3. identify trade-offs
4. identify risks
5. identify missing information
6. explain why alternatives differ

Do not perform unsupported numerical calculations.

NEREUS DECISION DATA:

{json.dumps(decision_payload, indent=2)}
""".strip()


def _schema():
    return {
        "type": "object",
        "properties": {
            "decision": {
                "type": "string"
            },
            "decision_status": {
                "type": "string",
                "enum": [
                    "confirmed",
                    "conditional",
                    "no_decision"
                ]
            },
            "summary": {
                "type": "string"
            },
            "why": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            },
            "risks": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            },
            "alternatives": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "vessel_class": {
                            "type": "string"
                        },
                        "status": {
                            "type": "string"
                        },
                        "reason": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "vessel_class",
                        "status",
                        "reason"
                    ],
                    "additionalProperties": False
                }
            },
            "missing_data": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            }
        },
        "required": [
            "decision",
            "decision_status",
            "summary",
            "why",
            "risks",
            "alternatives",
            "missing_data"
        ],
        "additionalProperties": False
    }


def analyze_with_ai(decision_payload: dict) -> dict:
    token = os.getenv("HF_TOKEN")

    if not token:
        raise AIDecisionError(
            "HF_TOKEN is not configured."
        )

    client = InferenceClient(
        provider="auto",
        api_key=token,
    )

    response = client.chat_completion(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the NEREUS procurement "
                    "decision reasoning layer."
                ),
            },
            {
                "role": "user",
                "content": _build_prompt(
                    decision_payload
                ),
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "nereus_decision",
                "schema": _schema(),
            },
        },
        temperature=0.1,
        max_tokens=1200,
    )

    content = response.choices[0].message.content

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise AIDecisionError(
            "Hugging Face returned invalid structured output."
        ) from exc