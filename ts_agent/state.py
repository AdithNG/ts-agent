from __future__ import annotations

from typing import Annotated, Any
from typing_extensions import NotRequired

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import TypedDict


class TemporalProgram(TypedDict):
    hist_ops: list[dict]       # history reconditioning operations
    foundation_forecast: dict  # foundation model call spec
    future_ops: list[dict]     # future enforcement operations


class TSAgentState(TypedDict):
    # ---- Input ----
    task: str
    series: list[float]
    timestamps: list[str]
    context: str

    # ---- Router output ----
    task_family: str           # "qa" | "forecast"
    knowledge_augmented: bool

    # ---- Contextual Prior output ----
    scenario_type: str         # one of 7 ECLIPSE scenario categories
    archetypes: list[dict]     # top-k entries from the Temporal Archetype Bank
    intervention_mode: str     # "hist" | "future" | "both" | "none"

    # ---- Planner output ----
    temporal_program: TemporalProgram

    # ---- Executor output ----
    execution_result: Any

    # ---- Audit output ----
    verdict: str               # "accept" | "revise" | "fallback"
    issues: list[str]
    revisions: int

    # ---- Final answer ----
    answer: str

    # ---- LangGraph message thread ----
    messages: Annotated[list[BaseMessage], add_messages]
