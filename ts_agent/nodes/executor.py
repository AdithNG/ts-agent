"""Executor — runs the temporal program against the tool library."""
from ..state import TSAgentState
from ..tools.library import apply_hist_ops, run_foundation_forecast, apply_future_ops


def run(state: TSAgentState) -> dict:
    program = state["temporal_program"]
    series = list(state["series"])

    conditioned = apply_hist_ops(series, program.get("hist_ops", []))
    forecast = run_foundation_forecast(conditioned, program.get("foundation_forecast", {}))
    enforced = apply_future_ops(forecast, program.get("future_ops", []))

    return {"execution_result": enforced}
