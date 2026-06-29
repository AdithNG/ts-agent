"""Audit Agent — validates execution result and returns ACCEPT / REVISE / FALLBACK."""
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from ..state import TSAgentState

_MODEL = "claude-sonnet-4-6"

_SYSTEM = """You are an audit agent for a time-series forecasting pipeline.

Evaluate the execution result against the original task and context.

Check:
  format_valid         — result is the right type and length
  context_aligned      — result respects the contextual scenario and constraints
  temporally_consistent — forecast is coherent (no impossible jumps, honours trends)

verdict:
  "accept"  — result is satisfactory, pipeline complete
  "revise"  — result has fixable issues; list them so the planner can address them
  "fallback" — result is fundamentally broken; return naive baseline

issues          — list of specific problems found (empty if accept)
suggested_repair — concrete one-sentence instruction for the planner (empty if accept)"""


class _AuditOut(BaseModel):
    verdict: str
    format_valid: bool
    context_aligned: bool
    temporally_consistent: bool
    issues: list[str]
    suggested_repair: str


def run(state: TSAgentState) -> dict:
    llm = ChatAnthropic(model=_MODEL).with_structured_output(_AuditOut)

    result_preview = str(state.get("execution_result", ""))[:500]
    msg = HumanMessage(
        content=(
            f"Task: {state['task']}\n"
            f"Context: {state.get('context') or ''}\n"
            f"Scenario type: {state.get('scenario_type', 'unknown')}\n"
            f"Intervention mode: {state.get('intervention_mode', 'none')}\n"
            f"Execution result: {result_preview}"
        )
    )
    out: _AuditOut = llm.invoke([SystemMessage(content=_SYSTEM), msg])

    return {
        "verdict": out.verdict,
        "issues": out.issues,
        "revisions": state.get("revisions", 0) + 1,
        "messages": [msg],
    }


def run_fallback(state: TSAgentState) -> dict:
    """Naive last-value-repeated baseline when audit exhausts revision budget."""
    series = state["series"]
    last = series[-1] if series else 0.0
    horizon = state.get("temporal_program", {}).get("foundation_forecast", {}).get("horizon", 12)
    baseline = [last] * horizon
    return {
        "execution_result": baseline,
        "answer": (
            f"Fallback baseline: last observed value ({last:.4f}) "
            f"repeated for {horizon} steps."
        ),
        "verdict": "fallback",
    }
