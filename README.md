# TS-Agent

A LangGraph implementation of the ECLIPSE multi-agent pipeline for context-rich time-series forecasting and QA.

## Architecture

Five agents wired as a `StateGraph`, matching the ECLIPSE paper:

```
START → router → contextual_prior → planner → executor → audit → END
                                        ↑________________________| (REVISE)
         ↓ (self-contained QA)
       qa_direct → END
```

| Node | Role |
|------|------|
| `router` | Classifies task as `forecast` or `qa`, flags if knowledge-augmented |
| `contextual_prior` | Scenario profiler → Temporal Archetype Bank retrieval → Intervention Advisor |
| `planner` | Synthesizes temporal program Π = (hist\_ops, foundation\_forecast, future\_ops) |
| `executor` | Runs the tool library against the temporal program |
| `audit` | Returns ACCEPT / REVISE / FALLBACK; loops back to planner on REVISE (max 3×) |

## Setup

```bash
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=...
```

## Run

```bash
python example.py
```

## Structure

```
ts_agent/
├── state.py                # TSAgentState TypedDict
├── graph.py                # StateGraph definition and compilation
├── nodes/
│   ├── router.py           # Intent-aware routing + direct QA handler
│   ├── prior.py            # Contextual Prior Agent (3 sub-components)
│   ├── planner.py          # Temporal program synthesis
│   ├── executor.py         # Tool library execution
│   └── audit.py            # Audit + fallback
├── tools/
│   └── library.py          # Hist reconditioning, forecast stub, future enforcement ops
└── archetype_bank/
    ├── retrieval.py        # Bi-modal DTW + semantic retrieval via RRF
    └── bank.json           # Populate from TS-RAG medoid data to activate retrieval
```

## Status

- [x] Full graph wiring and typed state
- [x] All five agent nodes with LLM structured output
- [x] Tool library (clip, impute, scale, trend, event spike/dip ops)
- [x] Archetype bank retrieval (DTW + RRF) — active once `bank.json` is populated
- [ ] Replace forecast stub with real foundation model (Moirai / Chronos / Lag-LLaMA)
- [ ] Wire semantic embeddings for archetype retrieval second channel
- [ ] Populate `bank.json` from TS-RAG medoid data
