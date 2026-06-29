"""Quick end-to-end smoke test. Requires ANTHROPIC_API_KEY in environment."""
from ts_agent import graph

result = graph.invoke(
    {
        "task": (
            "Forecast electricity demand for the next 12 months. "
            "A major industrial plant will open in month 3, expected to increase demand by ~20%."
        ),
        "series": [
            100, 102, 98, 105, 110, 108, 112, 115, 111, 109, 114, 118,
            120, 122, 119, 125, 130, 128, 132, 135, 131, 129, 134, 138,
        ],
        "timestamps": [f"2023-{i:02d}" for i in range(1, 25)],
        "context": (
            "A major industrial plant is scheduled to open in month 3 of the forecast horizon. "
            "Historical demand has shown a steady upward trend with seasonal winter peaks."
        ),
        "messages": [],
        "revisions": 0,
        "issues": [],
    }
)

print("Verdict :", result.get("verdict"))
print("Forecast:", result.get("execution_result"))
print("Answer  :", result.get("answer"))
