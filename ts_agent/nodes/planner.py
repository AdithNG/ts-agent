"""Unified Planner — synthesizes a temporal program Π = (Πhist, Πfm, Πfuture)."""
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from ..state import TSAgentState

_MODEL = "gpt-4o"

_SYSTEM = """You are a temporal program synthesizer for a time-series forecasting agent.

Synthesize a temporal program with three parts:

hist_ops — history reconditioning operations (list of {op, params}).
  Available ops:
    clip_outliers   params: lo_pct, hi_pct
    impute_missing  params: (none)
    scale           params: factor, offset
    apply_event_effect  params: index (int, position in series), multiplier
    inject_pattern  params: pattern (list[float]), start_index

foundation_forecast — base forecasting call spec.
  Format: {model: "moirai"|"chronos"|"lag-llama", horizon: <int>, freq: "<str>"}

future_ops — future enforcement operations (list of {op, params}).
  Available ops:
    cap_at              params: value
    floor_at            params: value
    apply_trend         params: slope (per-step additive)
    enforce_event_spike params: index, multiplier
    enforce_event_dip   params: index, multiplier

Only include ops actually needed given the intervention_mode and scenario.
If intervention_mode is "none", both hist_ops and future_ops should be empty lists."""


class _PlannerOut(BaseModel):
    hist_ops: list[dict]
    foundation_forecast: dict
    future_ops: list[dict]
    rationale: str


def run(state: TSAgentState) -> dict:
    llm = ChatOpenAI(model=_MODEL).with_structured_output(_PlannerOut, method="function_calling")

    archetypes_str = (
        "\n".join(f"- {a['description']}" for a in state.get("archetypes", []))
        or "None available."
    )
    issues_str = "; ".join(state.get("issues", [])) or "None."

    msg = HumanMessage(
        content=(
            f"Task: {state['task']}\n"
            f"Context: {state.get('context') or ''}\n"
            f"Scenario type: {state.get('scenario_type', 'unknown')}\n"
            f"Intervention mode: {state.get('intervention_mode', 'none')}\n"
            f"Archetypes:\n{archetypes_str}\n"
            f"Audit issues to address: {issues_str}"
        )
    )
    out: _PlannerOut = llm.invoke([SystemMessage(content=_SYSTEM), msg])

    return {
        "temporal_program": {
            "hist_ops": out.hist_ops,
            "foundation_forecast": out.foundation_forecast,
            "future_ops": out.future_ops,
        },
        "messages": [msg],
    }
