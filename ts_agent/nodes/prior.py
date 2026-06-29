"""Contextual Prior Agent — three sub-components run in sequence:
  1. Temporal Scenario Profiler  → classifies context into one of 7 ECLIPSE categories
  2. Temporal Archetype Bank     → bi-modal retrieval (shape + semantic, RRF)
  3. Intervention Advisor        → predicts hist / future / both / none
"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from ..state import TSAgentState
from ..archetype_bank.retrieval import retrieve_archetypes

_MODEL = "gpt-4o"

_SCENARIO_TYPES = (
    "AddFutureConstraint",
    "CommonsenseInference",
    "FutureTimedEvent",
    "PastTimedEvent",
    "PastAndFutureEvents",
    "RuleBased",
    "IndirectReasoning",
)

_PROFILER_SYSTEM = f"""Classify the temporal scenario into exactly one of:
{", ".join(_SCENARIO_TYPES)}

Definitions:
  AddFutureConstraint  — task imposes a specific constraint on future values
  CommonsenseInference — requires general world / domain knowledge
  FutureTimedEvent     — a specific future event will affect the series
  PastTimedEvent       — a specific past event already affected the series
  PastAndFutureEvents  — both past and future events matter
  RuleBased            — explicit rules govern the series behaviour
  IndirectReasoning    — causal chain is indirect or multi-hop

Return JSON: scenario_type, rationale (one sentence)."""

_ADVISOR_SYSTEM = """Given the scenario type, context, and retrieved temporal archetypes,
recommend an intervention mode:
  "hist"   — recondition the historical series before forecasting
  "future" — enforce constraints on the post-forecast values
  "both"   — apply both historical and future modifications
  "none"   — raw series is sufficient, no modification needed

Return JSON: intervention_mode, rationale (one sentence)."""


class _ProfilerOut(BaseModel):
    scenario_type: str
    rationale: str


class _AdvisorOut(BaseModel):
    intervention_mode: str
    rationale: str


def run(state: TSAgentState) -> dict:
    task = state["task"]
    context = state.get("context") or ""

    # 1. Scenario Profiler
    profiler = ChatOpenAI(model=_MODEL).with_structured_output(_ProfilerOut, method="function_calling")
    p_msg = HumanMessage(content=f"Task: {task}\nContext: {context}")
    profile: _ProfilerOut = profiler.invoke([SystemMessage(content=_PROFILER_SYSTEM), p_msg])

    # 2. Temporal Archetype Bank
    archetypes = retrieve_archetypes(state["series"], context, k=3)

    # 3. Intervention Advisor
    advisor = ChatOpenAI(model=_MODEL).with_structured_output(_AdvisorOut, method="function_calling")
    archetype_str = "\n".join(f"- {a['description']}" for a in archetypes) or "None available."
    a_msg = HumanMessage(
        content=(
            f"Scenario: {profile.scenario_type}\n"
            f"Context: {context}\n"
            f"Retrieved archetypes:\n{archetype_str}"
        )
    )
    advice: _AdvisorOut = advisor.invoke([SystemMessage(content=_ADVISOR_SYSTEM), a_msg])

    return {
        "scenario_type": profile.scenario_type,
        "archetypes": archetypes,
        "intervention_mode": advice.intervention_mode,
        "messages": [p_msg, a_msg],
    }
