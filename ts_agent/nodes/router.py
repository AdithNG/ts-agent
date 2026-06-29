from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from ..state import TSAgentState

_MODEL = "claude-sonnet-4-6"

_ROUTER_SYSTEM = """You are an intent-aware router for a time-series agent.

Classify the task into:
  task_family  — "forecast" (predict future values) or "qa" (answer a question about the series)
  knowledge_augmented — true if the task references external events, domain knowledge, or
                        contextual scenarios beyond the raw series values
  confidence   — float 0–1
  rationale    — one sentence explaining the classification"""


class _RouterOut(BaseModel):
    task_family: str
    knowledge_augmented: bool
    confidence: float
    rationale: str


def run(state: TSAgentState) -> dict:
    llm = ChatAnthropic(model=_MODEL).with_structured_output(_RouterOut)
    msg = HumanMessage(
        content=f"Task: {state['task']}\nContext: {state.get('context') or 'None'}"
    )
    out: _RouterOut = llm.invoke([SystemMessage(content=_ROUTER_SYSTEM), msg])
    return {
        "task_family": out.task_family,
        "knowledge_augmented": out.knowledge_augmented,
        "messages": [msg],
    }


_QA_SYSTEM = "Answer the question directly from the provided time series and context."


def run_qa(state: TSAgentState) -> dict:
    """Direct path for self-contained QA tasks that need no contextual intervention."""
    llm = ChatAnthropic(model=_MODEL)
    series_snippet = ", ".join(str(v) for v in state["series"][:50])
    msg = HumanMessage(
        content=(
            f"Task: {state['task']}\n"
            f"Series (first 50 values): [{series_snippet}]\n"
            f"Context: {state.get('context') or 'None'}"
        )
    )
    response = llm.invoke([SystemMessage(content=_QA_SYSTEM), msg])
    return {"answer": response.content, "messages": [msg, response]}
