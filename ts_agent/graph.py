from langgraph.graph import StateGraph, START, END

from .state import TSAgentState
from .nodes import router, prior, planner, executor, audit

MAX_REVISIONS = 3


def _route_after_router(state: TSAgentState) -> str:
    if state["task_family"] == "qa" and not state.get("knowledge_augmented", False):
        return "qa_direct"
    return "contextual_prior"


def _route_after_audit(state: TSAgentState) -> str:
    if state["verdict"] == "accept":
        return END
    if state["verdict"] == "revise" and state.get("revisions", 0) < MAX_REVISIONS:
        return "planner"
    return "fallback"


def build_graph() -> StateGraph:
    g = StateGraph(TSAgentState)

    g.add_node("router", router.run)
    g.add_node("qa_direct", router.run_qa)
    g.add_node("contextual_prior", prior.run)
    g.add_node("planner", planner.run)
    g.add_node("executor", executor.run)
    g.add_node("audit", audit.run)
    g.add_node("fallback", audit.run_fallback)

    g.add_edge(START, "router")
    g.add_conditional_edges(
        "router",
        _route_after_router,
        {"qa_direct": "qa_direct", "contextual_prior": "contextual_prior"},
    )
    g.add_edge("qa_direct", END)
    g.add_edge("contextual_prior", "planner")
    g.add_edge("planner", "executor")
    g.add_edge("executor", "audit")
    g.add_conditional_edges(
        "audit",
        _route_after_audit,
        {END: END, "planner": "planner", "fallback": "fallback"},
    )
    g.add_edge("fallback", END)

    return g.compile()


graph = build_graph()
